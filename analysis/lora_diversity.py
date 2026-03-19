#!/usr/bin/env python3
"""
LoRA diversity visualization: shows that Code2LoRA generates distinct adapters per repo.

Produces three figures:
  1. lora_similarity_heatmap.pdf  — pairwise cosine similarity (52×52 matrix)
  2. lora_layernorm_heatmap.pdf   — per-repo × per-module L2 norm heat map
  3. lora_sim_histogram.pdf       — distribution of inter-repo cosine similarities

No pyarrow dependency — uses MinimalHypernet from run_tsne.py.

Run:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/lora_diversity.py
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
from matplotlib.gridspec import GridSpec

CHECKPOINT   = "/scratch/lhotsko/TRAINING_CHECKPOINTS/HYPERNET/full_repos/hypernet_best.pt"
SPLITS_DIR   = Path("/scratch/lhotsko/REPO_DATASET")
RESULTS_FILE = "/scratch/lhotsko/BASELINES/hypernet_no_oracle_cr_test.json"
OUTPUT_DIR   = Path("/home/lhotsko/RepoPeftData/analysis/figures")
SPLIT        = "cr_test"


# ── Minimal Hypernetwork (same as run_tsne.py) ──────────────────────────────
class MinimalHypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, module_types, head_weights):
        super().__init__()
        self.types = module_types
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
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


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading results and embeddings...")
results = json.load(open(RESULTS_FILE))
repo_stats = defaultdict(lambda: {"em": 0, "n": 0})
for e in results["entries"]:
    repo_stats[e["repo"]]["n"] += 1
    repo_stats[e["repo"]]["em"] += int(e["exact_match"])
repo_em = {r: 100 * s["em"] / s["n"] for r, s in repo_stats.items()}

split_data = json.loads((SPLITS_DIR / f"{SPLIT}.json").read_text())
repos = split_data["repositories"]
repo_names = sorted([r for r in repos if r in repo_em and "embedding" in repos[r]])
print(f"  {len(repo_names)} repos with embeddings + EM scores")

# ── 2. Load checkpoint ────────────────────────────────────────────────────────
print("Loading hypernetwork checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
hconfig  = ckpt["hypernet_config"]
sd       = ckpt["hypernet_state_dict"]
mod_types = hconfig["types"]

hypernet = MinimalHypernet(hconfig["input_dim"], hconfig["hidden_dim"], mod_types, sd)
hypernet.trunk[0].weight = nn.Parameter(sd["trunk.0.weight"])
hypernet.trunk[0].bias   = nn.Parameter(sd["trunk.0.bias"])
hypernet.trunk[2].weight = nn.Parameter(sd["trunk.2.weight"])
hypernet.trunk[2].bias   = nn.Parameter(sd["trunk.2.bias"])
hypernet.eval()
rank = hconfig["rank"]
print(f"  rank={rank}, hidden={hconfig['hidden_dim']}, modules={mod_types}")

# ── 3. Generate LoRA vectors + per-module norms ───────────────────────────────
print("Generating LoRA vectors...")
lora_vecs   = []   # full flattened vectors (for similarity)
module_norms = []  # [n_repos × n_modules] matrix of L2 norms

# Module-type labels ordered for the heatmap (A×B product norm = ‖A‖·‖B‖)
module_labels = mod_types  # e.g. ['down_proj', 'gate_proj', 'q_proj', ...]

with torch.no_grad():
    for rn in repo_names:
        emb = repos[rn]["embedding"]
        ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        ctx = F.normalize(ctx, p=2, dim=-1)
        out = hypernet(ctx)

        parts = []
        norms_row = []
        for t in mod_types:
            A, B = out[t]          # A: [1, rank, in_dim], B: [1, out_dim, rank]
            parts.append(A.float().flatten())
            parts.append(B.float().flatten())
            # LoRA delta = B @ A  →  ‖ΔW‖_F ≈ ‖A‖_F · ‖B‖_F (rank-1 approximation insight)
            norms_row.append((A.float().norm() * B.float().norm()).item())

        lora_vecs.append(torch.cat(parts).numpy())
        module_norms.append(norms_row)

X = np.stack(lora_vecs)                    # [n_repos, vec_dim]
N_mat = np.array(module_norms)             # [n_repos, n_modules]
em_arr = np.array([repo_em[r] for r in repo_names])
print(f"  LoRA vector dim: {X.shape[1]}")

# Sort repos by EM for cleaner heatmaps
sort_idx = np.argsort(em_arr)
repo_names_sorted = [repo_names[i] for i in sort_idx]
short_names = [rn.split("/")[-1][:18] for rn in repo_names_sorted]
X_sorted   = X[sort_idx]
N_sorted   = N_mat[sort_idx]
em_sorted  = em_arr[sort_idx]

# Mean-center for similarity (removes shared "base" component)
X_c = X_sorted - X_sorted.mean(axis=0, keepdims=True)
X_n = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-8)
sim = X_n @ X_n.T   # [n_repos, n_repos]

off_diag = sim[~np.eye(len(repo_names), dtype=bool)]
print(f"\nOff-diagonal cosine similarity (mean-centered):")
print(f"  mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
      f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Pairwise cosine similarity heatmap ──────────────────────────────
print("\nPlotting cosine similarity heatmap...")
fig, ax = plt.subplots(figsize=(8.5, 7.0))

im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Cosine Similarity", fontsize=10)
cbar.ax.tick_params(labelsize=8)

n = len(repo_names)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short_names, rotation=90, fontsize=4.5)
ax.set_yticklabels(short_names, fontsize=4.5)

# Add EM colorbar on the right side (via scatter trick)
em_norm = plt.Normalize(em_sorted.min(), em_sorted.max())
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=em_norm)
sm.set_array([])
cbar2 = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.12)
cbar2.set_label("CR EM % (row order)", fontsize=8)
cbar2.ax.tick_params(labelsize=7)

# Tick colors by EM
cmap_em = plt.cm.RdYlGn
for i, (lbl, em_v) in enumerate(zip(ax.get_yticklabels(), em_sorted)):
    lbl.set_color(cmap_em(em_norm(em_v)))
for i, (lbl, em_v) in enumerate(zip(ax.get_xticklabels(), em_sorted)):
    lbl.set_color(cmap_em(em_norm(em_v)))

ax.set_title(
    f"Pairwise LoRA Cosine Similarity (mean-centered, $n$={n} repos)\n"
    f"Repos sorted by CR EM ↑ — off-diagonal mean={off_diag.mean():.3f}",
    fontsize=10,
)
fig.tight_layout()
p = OUTPUT_DIR / "lora_similarity_heatmap.pdf"
fig.savefig(str(p), dpi=300, bbox_inches="tight")
fig.savefig(str(p.with_suffix(".png")), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Figure 2: Per-module L2 norm heatmap ──────────────────────────────────────
print("Plotting per-module norm heatmap...")

# Normalize each column (module) to [0,1] so all modules are comparable
N_norm = (N_sorted - N_sorted.min(axis=0)) / (N_sorted.max(axis=0) - N_sorted.min(axis=0) + 1e-8)

fig, ax = plt.subplots(figsize=(max(6, len(mod_types) * 0.65), 7.5))
im = ax.imshow(N_norm, cmap="viridis", aspect="auto", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Normalised ‖ΔW‖_F per module", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xticks(range(len(mod_types)))
ax.set_xticklabels(mod_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(short_names, fontsize=4.5)

# Color y-labels by EM
for lbl, em_v in zip(ax.get_yticklabels(), em_sorted):
    lbl.set_color(cmap_em(em_norm(em_v)))

ax.set_xlabel("LoRA Module Type", fontsize=10)
ax.set_ylabel("Repository (sorted by CR EM ↑)", fontsize=10)
ax.set_title(
    "Per-Repo, Per-Module LoRA Weight Magnitude\n"
    "Each column normalised; rows sorted by CR EM ↑",
    fontsize=10,
)
fig.tight_layout()
p = OUTPUT_DIR / "lora_layernorm_heatmap.pdf"
fig.savefig(str(p), dpi=300, bbox_inches="tight")
fig.savefig(str(p.with_suffix(".png")), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ── Figure 3: Inter-repo similarity histogram ─────────────────────────────────
print("Plotting similarity histogram...")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(off_diag, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4, alpha=0.85)
ax.axvline(off_diag.mean(), color="#C44E52", lw=1.8, ls="--",
           label=f"Mean = {off_diag.mean():.3f}")
ax.axvline(0, color="black", lw=0.8, ls=":", alpha=0.5, label="Cosine = 0")
ax.set_xlabel("Pairwise Cosine Similarity (mean-centered LoRA vectors)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(
    f"Distribution of Inter-Repo LoRA Similarities ($n$={n} repos, "
    f"{n*(n-1)//2} pairs)",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
p = OUTPUT_DIR / "lora_sim_histogram.pdf"
fig.savefig(str(p), dpi=300, bbox_inches="tight")
fig.savefig(str(p.with_suffix(".png")), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

print("\nAll figures written to", OUTPUT_DIR)
