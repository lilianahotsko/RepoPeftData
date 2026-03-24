#!/usr/bin/env python3
"""
Compare Code2LoRA-generated LoRA weights vs FFT+DRC learned weight deltas.

Generates a comparative heatmap showing per-module weight norms:
  - Code2LoRA: B@A product norms per module type for representative repos
  - FFT+DRC: (W_finetuned - W_pretrained) norms per module type

Demonstrates Code2LoRA produces more structured, repo-specific adaptations
while FFT+DRC applies a uniform, blunt weight update across all repos.

Run:
    source /scratch/lhotsko/venvs/qwen-cu126-py312/bin/activate
    cd /home/lhotsko/RepoPeftData
    python analysis/lora_comparison.py
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

CHECKPOINT   = "/scratch/lhotsko/TRAINING_CHECKPOINTS/HYPERNET/full_repos/hypernet_best.pt"
FFT_DRC_DIR  = "/scratch/lhotsko/TRAINING_CHECKPOINTS/FFT_DRC4K/final"
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B"
SPLITS_DIR   = Path("/scratch/lhotsko/REPO_DATASET")
RESULTS_FILE = "/scratch/lhotsko/BASELINES/hypernet_no_oracle_cr_test.json"
OUTPUT_DIR   = Path("/home/lhotsko/RepoPeftData/analysis/figures")
PAPER_DIR    = Path("/home/lhotsko/RepoPeftData/RepoPeft_Paper/figures")
SPLIT        = "cr_test"

# Module types that match LoRA target modules
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
MODULE_LABELS  = ["Q", "K", "V", "O", "Up", "Gate", "Down"]
NUM_LAYERS     = 28


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


def load_fft_drc_norms():
    """Load FFT+DRC checkpoint and compute per-module weight delta norms."""
    from transformers import AutoModelForCausalLM

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    base_sd = base.state_dict()

    print("Loading FFT+DRC checkpoint...")
    fft_path = Path(FFT_DRC_DIR)
    from safetensors.torch import load_file
    fft_sd = {}
    for sf in sorted(fft_path.glob("model*.safetensors")):
        fft_sd.update(load_file(str(sf)))

    # Compute per-module-type norms aggregated across layers
    norms = {}
    for mod_type in TARGET_MODULES:
        layer_norms = []
        for layer_idx in range(NUM_LAYERS):
            for proj_part in [mod_type]:
                # Build key pattern: model.layers.{layer}.self_attn.{mod_type}.weight
                # or model.layers.{layer}.mlp.{mod_type}.weight
                for prefix in [f"model.layers.{layer_idx}.self_attn.{proj_part}.weight",
                               f"model.layers.{layer_idx}.mlp.{proj_part}.weight"]:
                    if prefix in fft_sd and prefix in base_sd:
                        delta = (fft_sd[prefix].float() - base_sd[prefix].float())
                        layer_norms.append(delta.norm().item())
        norms[mod_type] = np.mean(layer_norms) if layer_norms else 0.0

    del base, base_sd, fft_sd
    return norms


def load_code2lora_norms(n_repos=10):
    """Generate Code2LoRA adapters for representative repos and compute per-module norms."""
    print("Loading hypernetwork checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    hconfig = ckpt["hypernet_config"]
    sd = ckpt["hypernet_state_dict"]
    mod_types = hconfig["types"]

    hypernet = MinimalHypernet(hconfig["input_dim"], hconfig["hidden_dim"], mod_types, sd)
    hypernet.trunk[0].weight = nn.Parameter(sd["trunk.0.weight"])
    hypernet.trunk[0].bias   = nn.Parameter(sd["trunk.0.bias"])
    hypernet.trunk[2].weight = nn.Parameter(sd["trunk.2.weight"])
    hypernet.trunk[2].bias   = nn.Parameter(sd["trunk.2.bias"])
    hypernet.eval()

    # Load repo embeddings + EM scores
    results = json.load(open(RESULTS_FILE))
    repo_stats = defaultdict(lambda: {"em": 0, "n": 0})
    for e in results["entries"]:
        repo_stats[e["repo"]]["n"] += 1
        repo_stats[e["repo"]]["em"] += int(e["exact_match"])
    repo_em = {r: 100 * s["em"] / s["n"] for r, s in repo_stats.items()}

    split_data = json.loads((SPLITS_DIR / f"{SPLIT}.json").read_text())
    repos = split_data["repositories"]
    repo_names = sorted([r for r in repos if r in repo_em and "embedding" in repos[r]])

    # Pick n_repos spread across EM range
    sorted_by_em = sorted(repo_names, key=lambda r: repo_em[r])
    indices = np.linspace(0, len(sorted_by_em) - 1, n_repos, dtype=int)
    selected = [sorted_by_em[i] for i in indices]

    print(f"  Generating LoRA for {len(selected)} repos...")
    all_norms = []  # [n_repos × n_module_types]
    names = []

    with torch.no_grad():
        for rn in selected:
            emb = repos[rn]["embedding"]
            ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            ctx = F.normalize(ctx, p=2, dim=-1)
            out = hypernet(ctx)

            norms_row = []
            for t in mod_types:
                A, B = out[t]
                norms_row.append((A.float().norm() * B.float().norm()).item())
            all_norms.append(norms_row)
            short_name = rn.split("/")[-1][:15] if "/" in rn else rn[:15]
            names.append(f"{short_name} ({repo_em[rn]:.0f}%)")

    return np.array(all_norms), names, mod_types


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load Code2LoRA norms
    c2l_norms, repo_labels, mod_types = load_code2lora_norms(n_repos=12)

    # Load FFT+DRC norms
    fft_norms = load_fft_drc_norms()

    # Build FFT+DRC row (repeated for visual comparison)
    fft_row = np.array([fft_norms.get(t, 0.0) for t in mod_types])

    # Normalize both to comparable scale (relative to their own max)
    c2l_max = c2l_norms.max()
    fft_max = fft_row.max()

    c2l_norm = c2l_norms / c2l_max if c2l_max > 0 else c2l_norms
    fft_norm = fft_row / fft_max if fft_max > 0 else fft_row

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [len(repo_labels), 2]})

    # Top: Code2LoRA per-repo heatmap
    im1 = ax1.imshow(c2l_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_yticks(range(len(repo_labels)))
    ax1.set_yticklabels(repo_labels, fontsize=8)
    ax1.set_xticks(range(len(mod_types)))
    ax1.set_xticklabels([MODULE_LABELS[TARGET_MODULES.index(t)] if t in TARGET_MODULES else t
                         for t in mod_types], fontsize=10)
    ax1.set_title("Code2LoRA: Per-Repo LoRA Weight Norms (normalized)", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="Relative norm", shrink=0.8)

    # Bottom: FFT+DRC uniform row (repeated for visibility)
    fft_matrix = np.tile(fft_norm, (2, 1))
    im2 = ax2.imshow(fft_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["FFT+DRC"], fontsize=10)
    ax2.set_xticks(range(len(mod_types)))
    ax2.set_xticklabels([MODULE_LABELS[TARGET_MODULES.index(t)] if t in TARGET_MODULES else t
                         for t in mod_types], fontsize=10)
    ax2.set_title("FFT+DRC: Uniform Weight Delta Norms (normalized)", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="Relative norm", shrink=0.8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "lora_comparison_heatmap.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

    if PAPER_DIR.exists():
        import shutil
        shutil.copy2(out_path, PAPER_DIR / "lora_comparison_heatmap.pdf")
        print(f"Copied to {PAPER_DIR / 'lora_comparison_heatmap.pdf'}")

    plt.close()

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Code2LoRA norm variance across repos: {c2l_norms.var(axis=0).mean():.4f}")
    print(f"Code2LoRA norm std across repos:      {c2l_norms.std(axis=0).mean():.4f}")
    print(f"FFT+DRC norms (absolute):             {fft_row}")
    print(f"Code2LoRA norms (mean across repos):  {c2l_norms.mean(axis=0)}")


if __name__ == "__main__":
    main()
