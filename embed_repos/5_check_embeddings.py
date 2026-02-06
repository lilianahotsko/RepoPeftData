#!/usr/bin/env python3

# just for visualizations

import argparse
import json
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt


def load_embeddings(emb_dir: Path):
    names = []
    vecs = []
    bad = 0
    for p in sorted(emb_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue

        if not obj.get("ok", True):
            # some of your files might store ok=False
            continue

        v = obj.get("repo_embedding", None)
        if v is None:
            bad += 1
            continue

        v = np.asarray(v, dtype=np.float32)
        if v.ndim != 1:
            bad += 1
            continue
        if not np.all(np.isfinite(v)):
            bad += 1
            continue

        names.append(p.stem)
        vecs.append(v)

    if not vecs:
        raise SystemExit(f"No valid embeddings found in {emb_dir}")

    X = np.stack(vecs, axis=0)
    return names, X, bad


def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def pca_2d(X):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD for PCA
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2]  # [2, D]
    Z = Xc @ W.T
    return Z


def approx_duplicate_count(X, decimals=4):
    # rough: if many vectors identical after rounding => collapse/bug
    R = np.round(X, decimals=decimals)
    uniq = np.unique(R, axis=0).shape[0]
    return X.shape[0] - uniq, uniq


def sample_pairwise_sims(Xn, n_pairs=20000, seed=1234):
    rng = np.random.default_rng(seed)
    N = Xn.shape[0]
    if N < 2:
        return np.array([], dtype=np.float32)
    a = rng.integers(0, N, size=n_pairs)
    b = rng.integers(0, N, size=n_pairs)
    mask = a != b
    a = a[mask]
    b = b[mask]
    sims = np.sum(Xn[a] * Xn[b], axis=1)
    return sims.astype(np.float32)


def topk_neighbors(Xn, names, i, k=10):
    sims = Xn @ Xn[i]
    idx = np.argsort(-sims)[:k+1]  # includes itself
    out = []
    for j in idx:
        if j == i:
            continue
        out.append((names[j], float(sims[j])))
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=str, default="/home/lhotsko/scratch/repo_embeddings_v2")
    ap.add_argument("--pairs", type=int, default=20000, help="sampled pairs for similarity distribution")
    ap.add_argument("--neighbors", type=int, default=5, help="how many random repos to print neighbors for")
    ap.add_argument("--topk", type=int, default=8, help="neighbors per query")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plots", action="store_true", help="save PCA + similarity hist plots")
    ap.add_argument("--output-dir", type=str, default="./plots_v2", help="directory to save plots (default: ./plots)")
    args = ap.parse_args()


    emb_dir = Path(args.emb_dir).expanduser().resolve()
    names, X, bad = load_embeddings(emb_dir)

    print(f"Loaded repos: {len(names)}  dim={X.shape[1]}  bad_files={bad}")

    norms = np.linalg.norm(X, axis=1)
    print("Norms: mean/std/min/max =", float(norms.mean()), float(norms.std()), float(norms.min()), float(norms.max()))

    # If you L2-normalized at save time, norms should be ~1
    # But your embedding is concat(mean,max) then L2 normalize => also ~1
    if norms.mean() < 0.9 or norms.mean() > 1.1:
        print("WARNING: norms not ~1.0; did you skip normalization or store non-normalized vectors?")

    dup, uniq = approx_duplicate_count(X, decimals=4)
    print(f"Approx duplicates (rounded@1e-4): {dup}  unique: {uniq}")
    if dup > 0:
        print("NOTE: some near-duplicates exist. A few is fine; lots may indicate collapse or many near-identical repos.")

    Xn = l2_normalize(X)

    # Similarity distribution: if it’s too high everywhere => collapse; too low => very diverse/noisy
    sims = sample_pairwise_sims(Xn, n_pairs=args.pairs, seed=args.seed)
    if sims.size:
        print("Pairwise cosine (sampled): mean/std/min/max =",
              float(sims.mean()), float(sims.std()), float(sims.min()), float(sims.max()))
        # Healthy rough expectation: mean ~0.1–0.35, std not tiny
        if sims.std() < 0.03:
            print("WARNING: very low variance in similarities; embeddings may be collapsing or dominated by common tokens.")

    # PCA outliers
    Z = pca_2d(Xn)
    center = Z.mean(axis=0)
    dist = np.linalg.norm(Z - center, axis=1)
    out_idx = np.argsort(dist)[-10:][::-1]
    print("\nTop PCA outliers (often weird/huge repos, vendored code, generated code):")
    for j in out_idx:
        print(f"  dist={dist[j]:.3f}  {names[j]}")

    # Neighbor sanity checks
    rng = random.Random(args.seed)
    print("\nNearest-neighbor spot checks:")
    for i in rng.sample(range(len(names)), k=min(args.neighbors, len(names))):
        print("\nQuery:", names[i])
        for n, s in topk_neighbors(Xn, names, i, k=args.topk):
            print(f"  {s:.3f}  {n}")

    if args.plots:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot PCA
        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.6)
        plt.title("Repo embeddings PCA(2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        pca_path = output_dir / "pca_plot.png"
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved PCA plot to {pca_path}")

        # Plot similarity histogram
        if sims.size:
            plt.figure()
            plt.hist(sims, bins=50)
            plt.title("Sampled pairwise cosine similarities")
            plt.xlabel("cosine")
            plt.ylabel("count")
            plt.tight_layout()
            hist_path = output_dir / "similarity_hist.png"
            plt.savefig(hist_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved similarity histogram to {hist_path}")


if __name__ == "__main__":
    main()
