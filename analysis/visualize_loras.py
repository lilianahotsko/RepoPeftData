#!/usr/bin/env python3
"""
Visualize generated LoRA adapters across repos.

Generates:
1. t-SNE / UMAP of flattened LoRA vectors colored by repo
2. Pairwise cosine similarity heatmap
3. Per-module-type analysis

Usage:
    python analysis/visualize_loras.py --checkpoint $SCRATCH/.../hypernet_best.pt --splits-dir $SCRATCH/REPO_DATASET
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description="Visualize generated LoRA adapters")
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--output-dir", type=str, default="analysis/figures")
    ap.add_argument("--limit-repos", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from hypernetwork.hypernetwork_sampled import Hypernetwork, get_module_specs
    from transformers import AutoModelForCausalLM

    # Load split
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    split_path = splits_dir / f"{args.split}.json"
    data = json.loads(split_path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})

    repo_names = sorted(repos.keys())[:args.limit_repos]
    repo_embeddings = {}
    for rn in repo_names:
        emb = repos[rn].get("embedding")
        if emb:
            repo_embeddings[rn] = emb

    print(f"Loaded embeddings for {len(repo_embeddings)} repos")

    # Load hypernetwork
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.is_dir():
        for name in ["hypernet_best.pt", "hypernet_state.pt"]:
            if (checkpoint_path / name).exists():
                checkpoint_path = checkpoint_path / name
                break

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hconfig = ckpt["hypernet_config"]

    # We need module specs from a model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": "cpu"},
    )
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    specs = get_module_specs(model, target_modules)
    del model

    hypernet = Hypernetwork(
        input_dim=hconfig["input_dim"],
        module_specs=ckpt["module_specs"],
        hidden_dim=hconfig.get("hidden_dim", 512),
        rank=hconfig["rank"],
    )
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.eval()

    # Generate LoRA vectors for each repo
    lora_vectors = {}
    for rn, emb in repo_embeddings.items():
        ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        ctx = F.normalize(ctx, p=2, dim=-1)
        with torch.no_grad():
            h_out = hypernet(ctx)

        # Flatten all A and B matrices into a single vector
        parts = []
        for t in hypernet.types:
            parts.append(h_out["A"][t].flatten())
            parts.append(h_out["B"][t].flatten())
        lora_vectors[rn] = torch.cat(parts).numpy()

    print(f"Generated LoRA vectors: dim={len(list(lora_vectors.values())[0])}")

    # t-SNE visualization
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(lora_vectors.keys())
        X = np.stack([lora_vectors[n] for n in names])

        tsne = TSNE(n_components=2, perplexity=min(30, len(names) - 1), random_state=42)
        X_2d = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], s=30, alpha=0.7)
        for i, name in enumerate(names):
            short_name = name.split("/")[-1][:15]
            ax.annotate(short_name, (X_2d[i, 0], X_2d[i, 1]), fontsize=5, alpha=0.6)

        ax.set_title("t-SNE of Generated LoRA Vectors")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(str(output_dir / "lora_tsne.pdf"), dpi=300)
        fig.savefig(str(output_dir / "lora_tsne.png"), dpi=150)
        print(f"Saved t-SNE plot to {output_dir / 'lora_tsne.pdf'}")
        plt.close()

    except ImportError:
        print("sklearn or matplotlib not available, skipping t-SNE plot")

    # Pairwise cosine similarity
    names = list(lora_vectors.keys())
    X = np.stack([lora_vectors[n] for n in names])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim_matrix = X_norm @ X_norm.T

    sim_stats = {
        "mean": float(sim_matrix.mean()),
        "std": float(sim_matrix.std()),
        "min": float(np.fill_diagonal(sim_matrix.copy(), 0) or sim_matrix[~np.eye(len(names), dtype=bool)].min()),
        "max": float(sim_matrix[~np.eye(len(names), dtype=bool)].max()),
        "n_repos": len(names),
    }
    print(f"\nPairwise LoRA cosine similarity:")
    print(f"  Mean: {sim_stats['mean']:.4f}")
    print(f"  Std:  {sim_stats['std']:.4f}")
    print(f"  Min:  {sim_stats['min']:.4f}")
    print(f"  Max:  {sim_stats['max']:.4f}")

    (output_dir / "lora_similarity_stats.json").write_text(
        json.dumps(sim_stats, indent=2), encoding="utf-8")

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        short_names = [n.split("/")[-1][:12] for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=90, fontsize=4)
        ax.set_yticklabels(short_names, fontsize=4)
        fig.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("Pairwise LoRA Cosine Similarity")
        fig.tight_layout()
        fig.savefig(str(output_dir / "lora_similarity_heatmap.pdf"), dpi=300)
        fig.savefig(str(output_dir / "lora_similarity_heatmap.png"), dpi=150)
        print(f"Saved heatmap to {output_dir / 'lora_similarity_heatmap.pdf'}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping heatmap")

    # Per-module-type statistics
    module_stats = {}
    for rn, emb in list(repo_embeddings.items())[:10]:
        ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        ctx = F.normalize(ctx, p=2, dim=-1)
        with torch.no_grad():
            h_out = hypernet(ctx)

        for t in hypernet.types:
            if t not in module_stats:
                module_stats[t] = {"A_norms": [], "B_norms": [], "A_means": [], "B_means": []}
            module_stats[t]["A_norms"].append(h_out["A"][t].norm().item())
            module_stats[t]["B_norms"].append(h_out["B"][t].norm().item())
            module_stats[t]["A_means"].append(h_out["A"][t].abs().mean().item())
            module_stats[t]["B_means"].append(h_out["B"][t].abs().mean().item())

    print("\nPer-module-type LoRA statistics (averaged over 10 repos):")
    for t in sorted(module_stats.keys()):
        s = module_stats[t]
        print(f"  {t}: A_norm={np.mean(s['A_norms']):.6f}  B_norm={np.mean(s['B_norms']):.6f}  "
              f"A_mean={np.mean(s['A_means']):.6f}  B_mean={np.mean(s['B_means']):.6f}")

    (output_dir / "lora_module_stats.json").write_text(
        json.dumps({t: {k: float(np.mean(v)) for k, v in s.items()} for t, s in module_stats.items()},
                   indent=2), encoding="utf-8")

    print(f"\nAll analysis outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
