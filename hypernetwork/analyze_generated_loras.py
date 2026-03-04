#!/usr/bin/env python3
"""
For every repo embedding in ir_test.json, generate LoRA weights via the hypernetwork
and compare how different they are across repos.

Metrics: pairwise cosine similarity, L2 distance, per-module-type and aggregate.
Output: summary stats, pairwise distance matrix, optional JSON/plot.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure we import from parent (RepoPeftData)
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import hypernetwork_sampled as _hn
Hypernetwork = _hn.Hypernetwork


def load_repo_embeddings(splits_dir: Path, split_name: str) -> list[tuple[str, list[float]]]:
    """Load (repo_name, embedding) for each repo in split. One embedding per repo."""
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    result = []
    for repo_name in sorted(repos.keys()):
        r = repos[repo_name]
        emb = r.get("embedding")
        if emb is None:
            continue
        result.append((repo_name, emb))
    return result


def resolve_checkpoint(checkpoint_arg: str) -> Path:
    """Resolve checkpoint path (dir -> hypernet_best.pt or hypernet_state.pt)."""
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.is_dir():
        for name in ("hypernet_best.pt", "hypernet_state.pt"):
            p = checkpoint_path / name
            if p.exists():
                return p
        ckpt_dirs = sorted(
            (p for p in checkpoint_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
            key=lambda p: int(p.name.split("-")[1]) if len(p.name.split("-")) > 1 and p.name.split("-")[1].isdigit() else -1,
        )
        if ckpt_dirs:
            last = ckpt_dirs[-1] / "hypernet_state.pt"
            if last.exists():
                return last
        raise FileNotFoundError(f"No checkpoint in {checkpoint_path}")
    return checkpoint_path


def flatten_lora(lora: dict) -> torch.Tensor:
    """Flatten A and B dicts into a single vector for comparison."""
    parts = []
    for t in sorted(lora["A"].keys()):
        a = lora["A"][t][0].float().flatten()  # [0] for batch_index
        b = lora["B"][t][0].float().flatten()
        parts.extend([a, b])
    return torch.cat(parts)


def cosine_similarity_matrix(vectors: list[torch.Tensor]) -> np.ndarray:
    """Pairwise cosine similarity matrix. vectors[i] shape (D,)."""
    stacked = torch.stack([F.normalize(v.float(), p=2, dim=0) for v in vectors])
    sim = (stacked @ stacked.T).cpu().numpy()
    return sim


def l2_distance_matrix(vectors: list[torch.Tensor]) -> np.ndarray:
    """Pairwise L2 distance matrix."""
    n = len(vectors)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (vectors[i] - vectors[j]).norm().item()
            dist[i, j] = dist[j, i] = d
    return dist


def main():
    ap = argparse.ArgumentParser(description="Generate LoRAs for ir_test repos and compare them")
    default_dataset = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to hypernet checkpoint (dir or .pt)")
    ap.add_argument("--splits-dir", type=str, default=default_dataset,
                    help="Dir with ir_test.json")
    ap.add_argument("--split", type=str, default="ir_test_structured")
    ap.add_argument("--limit-repos", type=int, default=None,
                    help="Use only first N repos (for quick testing)")
    ap.add_argument("--output", type=str, default=None,
                    help="Save results JSON here")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repo_embeddings = load_repo_embeddings(splits_dir, args.split)
    if args.limit_repos:
        repo_embeddings = repo_embeddings[: args.limit_repos]

    if not repo_embeddings:
        print(f"No repos with embeddings in {splits_dir / args.split}.json")
        sys.exit(1)

    repo_names = [r[0] for r in repo_embeddings]
    n_repos = len(repo_names)
    print(f"Loaded {n_repos} repos from {args.split}.json")

    # Load hypernetwork
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading hypernetwork from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hypernet_config = ckpt["hypernet_config"]
    module_specs = ckpt["module_specs"]
    hidden_dim = hypernet_config.get("hidden_dim", 512)

    hypernet = Hypernetwork(
        input_dim=hypernet_config["input_dim"],
        module_specs=module_specs,
        hidden_dim=hidden_dim,
        rank=hypernet_config["rank"],
    )
    hypernet.load_state_dict(ckpt["hypernet_state_dict"])
    hypernet.to(args.device)
    hypernet.eval()

    # Generate LoRA for each repo
    print("Generating LoRAs...")
    loras = []
    with torch.no_grad():
        for repo_name, emb in repo_embeddings:
            ctx = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(args.device)
            ctx = F.normalize(ctx, p=2, dim=-1)
            h_out = hypernet(ctx)
            loras.append((repo_name, h_out))

    # Flatten for comparison
    flat_vectors = [flatten_lora(lo) for _, lo in loras]

    # Pairwise metrics
    cos_sim = cosine_similarity_matrix(flat_vectors)
    l2_dist = l2_distance_matrix(flat_vectors)

    # Summary stats (excluding diagonal)
    triu_idx = np.triu_indices(n_repos, k=1)
    cos_vals = cos_sim[triu_idx]
    l2_vals = l2_dist[triu_idx]

    print("\n" + "=" * 60)
    print("LoRA similarity across repos")
    print("=" * 60)
    print(f"  Repos: {n_repos}")
    print(f"  Cosine similarity (pairwise):")
    print(f"    min={cos_vals.min():.4f}  max={cos_vals.max():.4f}  mean={cos_vals.mean():.4f}  std={cos_vals.std():.4f}")
    print(f"  L2 distance (pairwise):")
    print(f"    min={l2_vals.min():.4f}  max={l2_vals.max():.4f}  mean={l2_vals.mean():.4f}  std={l2_vals.std():.4f}")
    print("=" * 60)

    # Per-module-type comparison (optional detail)
    module_types = sorted(loras[0][1]["A"].keys())
    print("\nPer-module-type cosine similarity (mean pairwise):")
    for t in module_types:
        vecs = []
        for _, lo in loras:
            a = lo["A"][t][0].float().flatten()
            b = lo["B"][t][0].float().flatten()
            vecs.append(F.normalize(torch.cat([a, b]), p=2, dim=0))
        sim_m = cosine_similarity_matrix(vecs)
        vals = sim_m[triu_idx]
        print(f"  {t}: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f}")

    results = {
        "n_repos": n_repos,
        "repo_names": repo_names,
        "cosine_similarity": {
            "min": float(cos_vals.min()),
            "max": float(cos_vals.max()),
            "mean": float(cos_vals.mean()),
            "std": float(cos_vals.std()),
        },
        "l2_distance": {
            "min": float(l2_vals.min()),
            "max": float(l2_vals.max()),
            "mean": float(l2_vals.mean()),
            "std": float(l2_vals.std()),
        },
        "cosine_matrix": cos_sim.tolist(),
        "l2_matrix": l2_dist.tolist(),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
