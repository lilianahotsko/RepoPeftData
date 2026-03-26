#!/usr/bin/env python3
"""
Extract pre-computed code embeddings (Qwen3-Embedding, 2048-dim) from
RepoPeft split JSONs and save as a torch dict for Text2LoRA code-conditioned
training.

Output format: {repo_slug: tensor([1, 2048])}
  - One embedding per repo (mean-pooled over code chunks)
  - Saved to $SCRATCH/TEXT2LORA_DATA/code_embeddings.pt

Usage:
    python baselines/text2lora/extract_code_embeddings.py \
        --splits-dir $SCRATCH/REPO_DATASET \
        --output     $SCRATCH/TEXT2LORA_DATA/code_embeddings.pt
"""

import argparse
import json
from pathlib import Path

import torch


def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--output",     required=True, type=Path)
    ap.add_argument("--splits",     nargs="+",
                    default=["train", "cr_val", "cr_test", "ir_val", "ir_test"])
    args = ap.parse_args()

    emb_dict: dict[str, torch.Tensor] = {}

    for split in args.splits:
        split_file = args.splits_dir / f"{split}.json"
        if not split_file.exists():
            print(f"  SKIP {split}.json (not found)")
            continue
        data = json.loads(split_file.read_text())
        repos = data.get("repositories", {})
        n_found = 0
        for repo_name, repo_data in repos.items():
            emb = repo_data.get("embedding")
            if emb is None or len(emb) == 0:
                continue
            repo_slug_name = slug(repo_name)
            # [1, emb_dim] — matches Text2LoRA's expected shape
            emb_dict[repo_slug_name] = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            n_found += 1
        print(f"  {split}: {n_found}/{len(repos)} repos with embeddings (dim={len(emb)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, args.output)
    print(f"\nSaved {len(emb_dict)} embeddings to {args.output}")
    # Verify dim
    sample = next(iter(emb_dict.values()))
    print(f"  Shape: {sample.shape} (expected [1, 2048])")


if __name__ == "__main__":
    main()
