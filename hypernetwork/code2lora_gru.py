#!/usr/bin/env python3
"""
Code2LoRA-GRU: hypernetwork that generates repository-specific LoRA adapters
by processing source files sequentially through a GRU, trained over commit
history.

Four components:
  1. File encoder       -- frozen Qwen3-Embedding-0.6B, chunked + MaxPool||MeanPool
  2. RepositoryGRU      -- maintains hidden state h_t across files
  3. Mamba2Initializer  -- reads preamble files to produce h_0
  4. LoRA generator     -- PAW-style shared-basis mapper from h_t -> LoRA weights

Usage:
    python hypernetwork/code2lora_gru.py --splits-dir $SCRATCH/REPO_DATASET
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Component 1: File Encoder (pooling helpers, model loaded externally)
# ---------------------------------------------------------------------------

def pool_file_chunks_maxmean(chunk_embs: torch.Tensor) -> torch.Tensor:
    """[K, D] chunk embeddings -> [2*D] file vector via concat(MaxPool, MeanPool)."""
    mean_pool = chunk_embs.mean(dim=0)
    max_pool = chunk_embs.max(dim=0).values
    return torch.cat([max_pool, mean_pool], dim=-1)


# ---------------------------------------------------------------------------
# Component 2: Repository GRU
# ---------------------------------------------------------------------------

class RepositoryGRU(nn.Module):
    """Sequentially processes file embeddings, maintaining a recurrent hidden
    state that accumulates repository knowledge.

    Args:
        input_dim:  Dimensionality of file embeddings (2*D for MaxPool||MeanPool).
        hidden_dim: GRU hidden state size H.
        num_layers: Number of stacked GRU layers.
        dropout:    Dropout between GRU layers (only if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        file_embeddings: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            file_embeddings: [B, T, input_dim] padded sequence of file embeddings.
            h_0:             [num_layers, B, H] initial hidden state (from Mamba2 or zeros).
            lengths:         [B] actual sequence lengths for packed processing.

        Returns:
            h_final: [B, H] final hidden state from the last GRU layer.
        """
        B = file_embeddings.size(0)

        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers, B, self.hidden_dim,
                device=file_embeddings.device, dtype=file_embeddings.dtype,
            )

        x = self.input_proj(file_embeddings)
        x = self.input_norm(x)

        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)

        output, h_n = self.gru(x, h_0)

        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        return self.output_norm(h_n[-1])  # [B, H]

    def forward_streaming(
        self,
        file_embedding: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single file embedding and return updated hidden state.
        Useful for online/incremental updates.

        Args:
            file_embedding: [B, input_dim] single file vector.
            h_prev:         [num_layers, B, H] previous hidden state.

        Returns:
            h_new: [num_layers, B, H] updated hidden state.
        """
        x = self.input_proj(file_embedding).unsqueeze(1)  # [B, 1, H]
        x = self.input_norm(x)
        _, h_new = self.gru(x, h_prev)
        return h_new


# ---------------------------------------------------------------------------
# Component 3: Mamba2 Initializer
# ---------------------------------------------------------------------------

class Mamba2Initializer(nn.Module):
    """Reads a variable-length repository preamble (file embeddings from early
    commits) and produces h_0 for the GRU.

    Uses a Mamba2 SSM block if available, falls back to a bidirectional GRU.

    Args:
        input_dim:  File embedding dimension (2*D).
        hidden_dim: Output hidden dimension H (matches GRU hidden_dim).
        d_state:    Mamba2 state dimension.
        d_conv:     Mamba2 convolution width.
        expand:     Mamba2 expansion factor.
        num_gru_layers: Number of GRU layers for the initializer's output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.use_mamba = MAMBA2_AVAILABLE

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        if self.use_mamba:
            self.encoder = Mamba2(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * num_gru_layers),
        )

    def forward(
        self,
        preamble_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            preamble_embeddings: [B, T_preamble, input_dim]
            lengths:             [B] actual preamble lengths.

        Returns:
            h_0: [num_gru_layers, B, hidden_dim] initial state for RepositoryGRU.
        """
        B = preamble_embeddings.size(0)
        x = self.input_proj(preamble_embeddings)
        x = self.input_norm(x)

        if self.use_mamba:
            x = self.encoder(x)  # [B, T, H]
        else:
            if lengths is not None:
                lengths_cpu = lengths.cpu().clamp(min=1)
                x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            x, _ = self.encoder(x)
            if lengths is not None:
                x, _ = pad_packed_sequence(x, batch_first=True)

        if lengths is not None:
            idx = (lengths - 1).clamp(min=0).long()
            h_last = x[torch.arange(B, device=x.device), idx]  # [B, H]
        else:
            h_last = x[:, -1, :]  # [B, H]

        out = self.output_proj(h_last)  # [B, H * num_gru_layers]
        h_0 = out.view(self.num_gru_layers, B, self.hidden_dim)
        return h_0


class MeanPoolInitializer(nn.Module):
    """Simple alternative to Mamba2: mean-pool preamble files, project to h_0."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024, num_gru_layers: int = 1):
        super().__init__()
        self.num_gru_layers = num_gru_layers
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * num_gru_layers),
        )

    def forward(
        self,
        preamble_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = preamble_embeddings.size(0)
        if lengths is not None:
            mask = torch.arange(preamble_embeddings.size(1), device=preamble_embeddings.device)
            mask = mask.unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (preamble_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = preamble_embeddings.mean(dim=1)
        out = self.proj(pooled)
        return out.view(self.num_gru_layers, B, self.hidden_dim)


# ---------------------------------------------------------------------------
# Component 4: LoRA Generator (PAW-style shared-basis)
# ---------------------------------------------------------------------------

class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class LoraGenerator(nn.Module):
    """PAW-style LoRA generator: shared basis vectors combined via
    input-conditioned mixing coefficients from the GRU hidden state.

    Args:
        input_dim:   GRU hidden_dim H.
        num_layers:  Number of transformer layers in the target LLM.
        module_dims: {module_name: (in_features, out_features)} for each LoRA target.
        hidden_dim:  Trunk hidden dimension.
        rank:        LoRA rank r.
        alpha:       LoRA alpha for scaling.
        num_bases:   Number of shared basis vectors per module type.
        trunk_depth: Number of residual MLP blocks in the trunk.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        module_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int = 512,
        rank: int = 16,
        alpha: float = 32.0,
        num_bases: int = 16,
        trunk_depth: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.lora_scaling = alpha / rank
        self.module_names = sorted(module_dims.keys())
        self.module_dims = module_dims
        self.num_bases = num_bases

        layers: list = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, trunk_depth - 1)):
            layers.append(_ResidualMLP(hidden_dim))
        self.trunk = nn.Sequential(*layers)

        for m in self.module_names:
            d_in, d_out = module_dims[m]
            self.register_parameter(
                f"A_bases_{m}",
                nn.Parameter(torch.randn(num_bases, rank, d_in) * 0.02),
            )
            self.register_parameter(
                f"B_bases_{m}",
                nn.Parameter(torch.zeros(num_bases, d_out, rank)),
            )

        M = len(self.module_names)
        num_coeff = num_layers * M * num_bases * 2
        self.coeff_head = nn.Linear(hidden_dim, num_coeff)
        nn.init.normal_(self.coeff_head.weight, std=0.01)
        nn.init.zeros_(self.coeff_head.bias)

    def forward(self, h_t: torch.Tensor) -> Dict:
        """
        Args:
            h_t: [B, input_dim] GRU hidden state (or pooled repo embedding).

        Returns:
            {(layer_idx, module_name): (A, B)}
            where A: [B, rank, d_in], B: [B, d_out, rank]
        """
        L = self.num_layers
        M = len(self.module_names)
        N = self.num_bases

        h = self.trunk(h_t.float())
        coeffs = self.coeff_head(h)
        BK = coeffs.shape[0]
        coeffs = coeffs.view(BK, L, M, N, 2)

        lora_params: Dict = {}
        for mi, m in enumerate(self.module_names):
            A_bases = getattr(self, f"A_bases_{m}")
            B_bases = getattr(self, f"B_bases_{m}")
            for layer_idx in range(L):
                alpha_A = coeffs[:, layer_idx, mi, :, 0]
                alpha_B = coeffs[:, layer_idx, mi, :, 1]
                A = torch.einsum("bn,nrd->brd", alpha_A, A_bases)
                B = torch.einsum("bn,ndr->bdr", alpha_B, B_bases)
                lora_params[(layer_idx, m)] = (A, B)

        return lora_params

    def config_dict(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "lora_scaling": self.lora_scaling,
            "module_names": self.module_names,
            "module_dims": {k: list(v) for k, v in self.module_dims.items()},
            "num_bases": self.num_bases,
        }


# ---------------------------------------------------------------------------
# Full Code2LoRA-GRU model (combines all 4 components)
# ---------------------------------------------------------------------------

class Code2LoRAGRU(nn.Module):
    """Complete Code2LoRA-GRU system combining:
    - RepositoryGRU for sequential file processing
    - Mamba2Initializer (or fallback) for preamble -> h_0
    - LoraGenerator (PAW-style) for h_t -> LoRA weights

    The file encoder (Qwen3-Embedding-0.6B) is external and frozen.

    Args:
        file_embed_dim:  Dimension of file embeddings (2*D from MaxPool||MeanPool).
        gru_hidden_dim:  GRU hidden state dimension H.
        gru_num_layers:  Number of stacked GRU layers.
        num_target_layers: Number of transformer layers in the target LLM.
        module_dims:     {module_name: (in_features, out_features)} for LoRA.
        lora_hidden_dim: Hidden dim for the LoRA generator trunk.
        lora_rank:       LoRA rank.
        lora_alpha:      LoRA alpha scaling.
        lora_num_bases:  Number of shared basis vectors.
        lora_trunk_depth: Depth of the LoRA generator trunk.
        init_type:       "mamba2", "meanpool", or "zeros".
        gru_dropout:     Dropout for multi-layer GRU.
        bptt_window:     Truncated BPTT window (detach every K steps). None = full BPTT.
    """

    def __init__(
        self,
        file_embed_dim: int,
        gru_hidden_dim: int = 1024,
        gru_num_layers: int = 1,
        num_target_layers: int = 28,
        module_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        lora_hidden_dim: int = 512,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_num_bases: int = 16,
        lora_trunk_depth: int = 2,
        init_type: str = "mamba2",
        gru_dropout: float = 0.0,
        bptt_window: Optional[int] = 32,
    ):
        super().__init__()
        self.file_embed_dim = file_embed_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.init_type = init_type
        self.bptt_window = bptt_window

        self.gru = RepositoryGRU(
            input_dim=file_embed_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
        )

        if init_type == "mamba2":
            self.initializer = Mamba2Initializer(
                input_dim=file_embed_dim,
                hidden_dim=gru_hidden_dim,
                num_gru_layers=gru_num_layers,
            )
        elif init_type == "meanpool":
            self.initializer = MeanPoolInitializer(
                input_dim=file_embed_dim,
                hidden_dim=gru_hidden_dim,
                num_gru_layers=gru_num_layers,
            )
        else:
            self.initializer = None

        if module_dims is not None:
            self.lora_generator = LoraGenerator(
                input_dim=gru_hidden_dim,
                num_layers=num_target_layers,
                module_dims=module_dims,
                hidden_dim=lora_hidden_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                num_bases=lora_num_bases,
                trunk_depth=lora_trunk_depth,
            )
        else:
            self.lora_generator = None

    def compute_h0(
        self,
        preamble_embeddings: Optional[torch.Tensor] = None,
        preamble_lengths: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Compute initial hidden state h_0 from the preamble.

        Returns:
            h_0: [gru_num_layers, B, gru_hidden_dim]
        """
        if preamble_embeddings is not None and self.initializer is not None:
            return self.initializer(preamble_embeddings, preamble_lengths)

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        return torch.zeros(
            self.gru_num_layers, batch_size, self.gru_hidden_dim,
            device=device, dtype=dtype,
        )

    def encode_repository(
        self,
        file_embeddings: torch.Tensor,
        h_0: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the GRU over the file sequence with optional truncated BPTT.

        Args:
            file_embeddings: [B, T, file_embed_dim]
            h_0:             [num_layers, B, H]
            lengths:         [B] actual lengths

        Returns:
            h_final: [B, H]
        """
        if self.bptt_window is None or not self.training:
            return self.gru(file_embeddings, h_0, lengths)

        B, T, D = file_embeddings.shape
        K = self.bptt_window
        h = h_0

        if lengths is not None:
            max_len = lengths.max().item()
        else:
            max_len = T

        for start in range(0, max_len, K):
            end = min(start + K, max_len)
            chunk = file_embeddings[:, start:end, :]

            chunk_lengths = None
            if lengths is not None:
                chunk_lengths = (lengths - start).clamp(min=0, max=end - start)
                active = chunk_lengths > 0
                if not active.any():
                    break

            h_out = self.gru(chunk, h, chunk_lengths)

            if end < max_len:
                h = h.detach()
                h[-1] = h_out.detach()
            else:
                return h_out

        return self.gru.output_norm(h[-1])

    def forward(
        self,
        file_embeddings: torch.Tensor,
        file_lengths: Optional[torch.Tensor] = None,
        preamble_embeddings: Optional[torch.Tensor] = None,
        preamble_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Full forward pass: preamble -> h_0 -> GRU -> LoRA.

        Args:
            file_embeddings:    [B, T, file_embed_dim]
            file_lengths:       [B]
            preamble_embeddings: [B, T_pre, file_embed_dim] or None
            preamble_lengths:   [B] or None

        Returns:
            h_final:    [B, H] final GRU hidden state
            lora_params: {(layer_idx, module_name): (A, B)} or None if no generator
        """
        B = file_embeddings.size(0)

        h_0 = self.compute_h0(
            preamble_embeddings=preamble_embeddings,
            preamble_lengths=preamble_lengths,
            batch_size=B,
            device=file_embeddings.device,
            dtype=file_embeddings.dtype,
        )

        h_final = self.encode_repository(file_embeddings, h_0, file_lengths)

        lora_params = None
        if self.lora_generator is not None:
            lora_params = self.lora_generator(h_final)

        return h_final, lora_params

    def config_dict(self) -> Dict:
        cfg = {
            "file_embed_dim": self.file_embed_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "init_type": self.init_type,
            "bptt_window": self.bptt_window,
        }
        if self.lora_generator is not None:
            cfg["lora_generator"] = self.lora_generator.config_dict()
        return cfg
