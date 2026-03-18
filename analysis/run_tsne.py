#!/usr/bin/env python3
"""
Paper-quality t-SNE visualization of generated LoRA weight vectors.
Colored by per-repo CR test Exact Match (%).
No dependency on full hypernetwork_sampled.py (avoids pyarrow requirement).

Run:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/run_tsne.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE

CHECKPOINT   = "/scratch/lhotsko/TRAINING_CHECKPOINTS/HYPERNET/full_repos/hypernet_best.pt"
SPLITS_DIR   = Path("/scratch/lhotsko/REPO_DATASET")
RESULTS_FILE = "/scratch/lhotsko/BASELINES/hypernet_no_oracle_cr_test.json"
OUTPUT_DIR   = Path("/home/lhotsko/RepoPeftData/analysis/figures")
SPLIT        = "cr_test"


# ── Minimal Hypernetwork reconstruction ────────────────────────────────────
class MinimalHypernet(nn.Module):
    """Reconstructed from checkpoint without importing hypernetwork_sampled."""
    def __init__(self, input_dim, hidden_dim, module_types, head_weights):
        super().__init__()
        self.types = module_types
        # Trunk: Linear -> GELU -> Linear -> GELU
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Output heads
        self.heads_A = nn.ModuleDict()
        self.heads_B = nn.ModuleDict()
        for t in module_types:
            wa = head_weights[f"heads_A.{t}.weight"]
            ba = head_weights[f"heads_A.{t}.bias"]
            self.heads_A[t] = nn.Linear(wa.shape[1], wa.shape[0])
            self.heads_A[t].weight = nn.Parameter(wa)
            self.heads_A[t].bias   = nn.Parameter(ba)

            wb = head_weights[f"heads_B.{t}.weight"]
            bb = head_weights[f"heads_B.{t}.bias"]
            self.heads_B[t] = nn.Linear(wb.shape[1], wb.shape[0])
            self.heads_B[t].weight = nn.Parameter(wb)
            self.heads_B[t].bias   = nn.Parameter(bb)

    def forward(self, x):
        h = self.trunk(x)
        return {t: (self.heads_A[t](h), self.heads_B[t](h)) for t in self.types}


# ── 1. Per-repo EM ──────────────────────────────────────────────────────────
results = json.load(open(RESULTS_FILE))
repo_stats = defaultdict(lambda: {"em": 0, "n": 0})
for e in results["entries"]:
    repo_stats[e["repo"]]["n"] += 1
    repo_stats[e["repo"]]["em"] += int(e["exact_match"])
repo_em = {r: 100 * s["em"] / s["n"] for r, s in repo_stats.items()}

# ── 2. Repo embeddings ──────────────────────────────────────────────────────
split_data = json.loads((SPLITS_DIR / f"{SPLIT}.json").read_text())
repos = split_data["repositories"]
repo_names = [r for r in sorted(repos.keys()) if r in repo_em and "embedding" in repos[r]]
print(f"Repos with embedding + EM: {len(repo_names)}")

# ── 3. Load checkpoint and build minimal model ──────────────────────────────
print("Loading checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
hconfig  = ckpt["hypernet_config"]
sd       = ckpt["hypernet_state_dict"]
mod_types = hconfig["types"]  # e.g. ['down_proj', 'gate_proj', ...]

hypernet = MinimalHypernet(
    input_dim=hconfig["input_dim"],
    hidden_dim=hconfig["hidden_dim"],
    module_types=mod_types,
    head_weights=sd,
)
# Load trunk weights manually
hypernet.trunk[0].weight = nn.Parameter(sd["trunk.0.weight"])
hypernet.trunk[0].bias   = nn.Parameter(sd["trunk.0.bias"])
hypernet.trunk[2].weight = nn.Parameter(sd["trunk.2.weight"])
hypernet.trunk[2].bias   = nn.Parameter(sd["trunk.2.bias"])
hypernet.eval()
print(f"Hypernetwork ready (rank={hconfig['rank']}, hidden={hconfig['hidden_dim']})")

# ── 4. Generate LoRA vectors ────────────────────────────────────────────────
lora_vecs = []
em_values = []
n_files_list = []
with torch.no_grad():
    for rn in repo_names:
        emb = repos[rn]["embedding"]
        ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        ctx = F.normalize(ctx, p=2, dim=-1)
        out = hypernet(ctx)
        parts = []
        for t in mod_types:
            a_mat, b_mat = out[t]
            parts.append(a_mat.float().flatten())
            parts.append(b_mat.float().flatten())
        lora_vecs.append(torch.cat(parts).numpy())
        em_values.append(repo_em[rn])
        n_files_list.append(len(repos[rn].get("qna_pairs", [])))

X    = np.stack(lora_vecs)
em_arr   = np.array(em_values)
size_arr = np.array(n_files_list)
print(f"LoRA vector dim: {X.shape[1]}, repos: {X.shape[0]}")

# Center: remove mean adapter so t-SNE shows per-repo *variation*
X_centered = X - X.mean(axis=0, keepdims=True)

# ── 5. PCA pre-reduction then t-SNE ────────────────────────────────────────
from sklearn.decomposition import PCA
n_pca = min(50, len(repo_names) - 1)
print(f"PCA reduction: {X_centered.shape[1]} -> {n_pca} dims (centered)...")
pca = PCA(n_components=n_pca, random_state=42)
X_pca = pca.fit_transform(X_centered)
print(f"Explained variance ({n_pca} PCs): {pca.explained_variance_ratio_.sum():.3f}")

print("Running t-SNE on PCA-reduced vectors...")
perp = min(15, len(repo_names) - 1)
tsne = TSNE(n_components=2, perplexity=perp, max_iter=2000, random_state=42,
            init="pca", learning_rate="auto")
X2 = tsne.fit_transform(X_pca)
print("t-SNE done.")

# ── 6. Paper-quality plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.8))

sc = ax.scatter(
    X2[:, 0], X2[:, 1],
    c=em_arr, cmap="RdYlGn",
    vmin=0, vmax=100,
    s=55, alpha=0.88,
    edgecolors="white", linewidths=0.4, zorder=3,
)

for i, rn in enumerate(repo_names):
    short = rn.split("/")[-1][:16]
    ax.annotate(
        short, (X2[i, 0], X2[i, 1]),
        fontsize=5.2, alpha=0.72,
        xytext=(3, 3), textcoords="offset points",
    )

cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label("CR Test Exact Match (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    "t-SNE of Generated LoRA Adapters (mean-centered, $n$=52 repos)\n"
    "Color = per-repo Exact Match (%)",
    fontsize=10,
)
ax.set_xlabel("t-SNE Component 1", fontsize=9)
ax.set_ylabel("t-SNE Component 2", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_pdf = OUTPUT_DIR / "lora_tsne.pdf"
out_png = OUTPUT_DIR / "lora_tsne.png"
fig.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ── 7. Similarity stats (on centered vectors) ───────────────────────────────
X_n = X_centered / (np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-8)
sim = X_n @ X_n.T
off = sim[~np.eye(len(repo_names), dtype=bool)]
print(f"\nPairwise LoRA cosine similarity (off-diagonal, n={len(repo_names)} repos):")
print(f"  Mean: {off.mean():.4f}")
print(f"  Std:  {off.std():.4f}")
print(f"  Min:  {off.min():.4f}")
print(f"  Max:  {off.max():.4f}")
