import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.block(x)


class ChunkAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, chunks: torch.Tensor):
        """
        Args:
            chunks: [B, num_chunks, embed_dim]
        Returns:
            aggregated: [B, embed_dim]
        """
        B, N, D = chunks.shape
        
        # Self-attention over chunks
        q = self.q_proj(chunks).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(chunks).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        v = self.v_proj(chunks).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted aggregation
        attn_out = torch.matmul(attn_weights, v)  # [B, H, N, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        attn_out = self.out_proj(attn_out)  # [B, N, D]
        
        # Aggregate across chunks (mean pooling with attention weights)
        chunk_weights = attn_weights.mean(dim=1).mean(dim=1)  # [B, N] - average attention per chunk
        aggregated = (attn_out * chunk_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        return self.norm(aggregated)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scale1 = nn.Linear(input_dim, hidden_dim // 4)  # Fine-grained
        self.scale2 = nn.Linear(input_dim, hidden_dim // 2)  # Medium
        self.scale3 = nn.Linear(input_dim, hidden_dim)       # Coarse
        
        self.fusion = nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4, hidden_dim)
    
    def forward(self, x):
        s1 = F.gelu(self.scale1(x))
        s2 = F.gelu(self.scale2(x))
        s3 = F.gelu(self.scale3(x))
        fused = torch.cat([s1, s2, s3], dim=-1)
        return self.fusion(fused)


class LayerConditioning(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layer_embeddings = nn.Embedding(num_layers, hidden_dim)
    
    def forward(self, features: torch.Tensor, layer_idx: int):
        if isinstance(layer_idx, int):
            layer_idx = torch.tensor([layer_idx], device=features.device)
        layer_emb = self.layer_embeddings(layer_idx)
        return features + layer_emb


class ImprovedHypernetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        module_specs,
        hidden_dim,
        rank,
        num_layers,
        use_residual_trunk: bool = True,
        use_attention_aggregation: bool = True,
        use_multiscale: bool = False,
        use_layer_conditioning: bool = False,
        num_residual_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.use_attention_aggregation = use_attention_aggregation
        self.use_multiscale = use_multiscale
        self.use_layer_conditioning = use_layer_conditioning
        
        if use_attention_aggregation:
            self.chunk_attention = ChunkAttention(input_dim, num_heads=8)
        else:
            self.chunk_aggregation = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
            )
        
        if use_multiscale:
            self.multiscale = MultiScaleFeatureExtractor(input_dim, hidden_dim)
            trunk_input_dim = hidden_dim
        else:
            trunk_input_dim = input_dim
        
        if use_residual_trunk:
            self.trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                *[ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)],
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, hidden_dim),
                nn.ReLU(),
            )
        
        if use_layer_conditioning:
            self.layer_conditioning = LayerConditioning(hidden_dim, num_layers)
        
        type_shapes = {}
        for _, _, t, in_f, out_f in module_specs:
            if t not in type_shapes:
                type_shapes[t] = (in_f, out_f)
        self.types = sorted(type_shapes.keys())
        self.type_shapes = type_shapes
        
        self.heads_A = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, rank * type_shapes[t][0])
            )
            for t in self.types
        })
        
        self.heads_B = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, type_shapes[t][1] * rank)
            )
            for t in self.types
        })
        
        self.temp_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(1e-2))
            for t in self.types
        })
        self.temp_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(1e-2))
            for t in self.types
        })
        
        self.use_gating = False
        if use_residual_trunk:
            self.gates_A = nn.ModuleDict({
                t: nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                for t in self.types
            })
            self.gates_B = nn.ModuleDict({
                t: nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                for t in self.types
            })
            self.use_gating = True
    
    def forward(self, ctx: torch.Tensor, layer_idx: Optional[int] = None):
        if ctx.dim() == 3: # if processing the data in few embeddings
            if self.use_attention_aggregation:
                # aggregate chunks
                ctx = self.chunk_attention(ctx)  # [B, num_chunks, dim] → [B, dim]
            else:
                # weighted average
                weights = F.softmax(self.chunk_aggregation(ctx).mean(dim=-1), dim=1)  # [B, num_chunks]
                ctx = (ctx * weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        elif ctx.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected ctx dimension: {ctx.dim()}, shape: {ctx.shape}")
        
        if self.use_multiscale: #for feature extraction
            ctx = self.multiscale(ctx)  # [B, dim] → [B, hidden_dim]
        
        base = self.trunk(ctx)  # [B, hidden_dim]
        
        if self.use_layer_conditioning and layer_idx is not None:
            base = self.layer_conditioning(base, layer_idx)
        
        A, B = {}, {}
        for t in self.types:
            in_f, out_f = self.type_shapes[t]
            
            A_flat = self.heads_A[t](base)  # [B, rank * in_f]
            B_flat = self.heads_B[t](base)   # [B, out_f * rank]
            
            A_t = A_flat.view(-1, self.rank, in_f)     # [B, r, in_f]
            B_t = B_flat.view(-1, out_f, self.rank)    # [B, out_f, r]
            
            temp_A_clamped = torch.clamp(self.temp_A[t], min=1e-4, max=1.0)
            temp_B_clamped = torch.clamp(self.temp_B[t], min=1e-4, max=1.0)

            A_t = A_t * temp_A_clamped
            B_t = B_t * temp_B_clamped
            
            if self.use_gating:
                gate_A = self.gates_A[t](base)  # [B, 1]
                gate_B = self.gates_B[t](base)   # [B, 1]
                A_t = A_t * gate_A.unsqueeze(-1)
                B_t = B_t * gate_B.unsqueeze(-1)
            
            A[t], B[t] = A_t, B_t
        
        return {"A": A, "B": B}