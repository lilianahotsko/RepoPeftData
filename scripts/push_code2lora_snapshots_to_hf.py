#!/usr/bin/env python3
"""Push the Code2LoRA-snapshots (static hypernetwork) dataset to HF Hub.

Layout::

    <repo-id>/
        README.md
        commits/
            train.parquet      400 anchor commits (latest in_repo=='train' per train repo)
            ir_val.parquet     train repos x in_repo_split=='val' commits
            ir_test.parquet    train repos x in_repo_split=='test' commits
            cr_val.parquet     held-out cr_val repos, all commits
            cr_test.parquet    held-out cr_test repos, all commits
        qna/
            train.parquet      ~44 k v2-extractor QnAs at the anchor commit
            ir_val.parquet     canonical (unfiltered) QnAs at val commits
            ir_test.parquet    canonical (unfiltered) QnAs at test commits
            cr_val.parquet     canonical QnAs for cr_val repos
            cr_test.parquet    canonical QnAs for cr_test repos
        SNAPSHOTS_README.json  provenance + per-file sha256 + row counts

Each commits row carries a 2048-d ``repo_state_embedding`` (frozen Qwen3,
concat(mean_files, max_files), L2 normalized) that uniquely identifies a
(repo, commit) snapshot. The static Code2LoRA trainer joins commits and
qna on ``(repo_id, commit_sha)``.

Usage::

    python scripts/push_code2lora_snapshots_to_hf.py \\
        --repo-id   nanigock/repopeft-code2lora-snapshots \\
        --build-dir /scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


README = """---
license: mit
task_categories:
  - text-generation
  - feature-extraction
language:
  - code
pretty_name: Code2LoRA Snapshots Dataset (static hypernetwork)
size_categories:
  - 100K<n<1M
configs:
  - config_name: commits
    data_files:
      - split: train
        path: commits/train.parquet
      - split: ir_val
        path: commits/ir_val.parquet
      - split: ir_test
        path: commits/ir_test.parquet
      - split: cr_val
        path: commits/cr_val.parquet
      - split: cr_test
        path: commits/cr_test.parquet
  - config_name: qna
    data_files:
      - split: train
        path: qna/train.parquet
      - split: ir_val
        path: qna/ir_val.parquet
      - split: ir_test
        path: qna/ir_test.parquet
      - split: cr_val
        path: qna/cr_val.parquet
      - split: cr_test
        path: qna/cr_test.parquet
---

# Code2LoRA snapshots dataset

This dataset is the **static-hypernetwork** companion to
[`nanigock/repopeft-gru-commits-v2`](https://huggingface.co/datasets/nanigock/repopeft-gru-commits-v2).
Every example is a single ``(repo, commit)`` snapshot annotated with:

* a 2048-d ``repo_state_embedding`` (frozen
  `Qwen/Qwen3-Embedding-0.6B`, concat of mean and max pooled file vectors,
  L2-normalized; details below);
* a list of canonical QnA pairs (test-file assertions) live at that
  commit.

It is designed for the **direct Code2LoRA** baseline: a single feed-forward
hypernetwork maps `repo_state_embedding -> LoRA delta`, with no GRU rollout
and no diff handling. Per-commit evaluation gives a decay curve directly
comparable to the GRU.

## Splits

| split | rows (commits) | role |
|---|---|---|
| train      | 400      | one anchor commit per train repo (last in_repo_split=='train'). |
| ir_val     | ~3 k     | train repos, in_repo_split=='val' commits.   |
| ir_test    | ~6 k     | train repos, in_repo_split=='test' commits.  |
| cr_val     | ~9 k     | held-out cr_val repos, all kept commits.     |
| cr_test    | ~7 k     | held-out cr_test repos, all kept commits.    |

The train QnAs are **re-extracted at the anchor commit** with the v2
extractor: `extract_from_file -> select_balanced_pairs`
(`max_per_repo=200`, `max_per_function=5`, `max_per_file=20`). This
guarantees that every QnA is actually present in the repo at the snapshot
point (no stale assertions). The eval QnAs are the canonical
RepoPeftBench QnAs, identical to those used to score Code2LoRA-GRU<sub>commit</sub>.

## repo_state_embedding details

| hyperparameter | value |
|---|---|
| Model                  | `Qwen/Qwen3-Embedding-0.6B`     |
| File chunking          | 2048 tokens, 256 overlap        |
| Min window tokens      | 8                               |
| Per-chunk pooling      | attention-mean of `last_hidden_state` |
| Per-file pooling       | mean over chunk vectors (1024-d)|
| Per-repo pooling       | `concat(mean_files, max_files)` (2048-d) |
| Repo vector norm       | L2 normalized                   |
| File filter            | tracked `.py` blobs, size <= 2 MB |
| Identical files dedup  | by `git ls-tree` blob SHA       |

## Loading

```python
from datasets import load_dataset
commits = load_dataset("nanigock/repopeft-code2lora-snapshots", "commits")
qna     = load_dataset("nanigock/repopeft-code2lora-snapshots", "qna")
```

Join the two on `(repo_id, commit_sha)` to form
`(repo_state_embedding, prefix, target)` triples for static training.

## Citation

```bibtex
@misc{repopeft_code2lora_snapshots_2026,
  title  = {Code2LoRA snapshots: a static-hypernetwork dataset for repository-aware LoRA generation},
  year   = {2026},
  author = {RepoPeftData authors},
}
```
"""


def _materialize_dir(src: Path, dst: Path,
                     follow_symlinks: bool = True) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.is_symlink() and follow_symlinks:
            real = entry.resolve()
            if real.is_file():
                shutil.copy2(real, target)
            elif real.is_dir():
                _materialize_dir(real, target, follow_symlinks=True)
        elif entry.is_dir():
            _materialize_dir(entry, target, follow_symlinks=True)
        elif entry.is_file():
            shutil.copy2(entry, target)


def _stage(build_dir: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    print(f"\n[commits] copying {build_dir / 'commits'} ...", flush=True)
    _materialize_dir(build_dir / "commits", staging / "commits")
    print(f"\n[qna] copying {build_dir / 'qna'} ...", flush=True)
    _materialize_dir(build_dir / "qna", staging / "qna")
    for fname in ("SNAPSHOTS_README.json",):
        src = build_dir / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
    (staging / "README.md").write_text(README, encoding="utf-8")


def _push(staging: Path, repo_id: str, token: str | None,
          private: bool, commit_message: str) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"\nCreating dataset repo: {repo_id} (private={private})", flush=True)
    create_repo(repo_id=repo_id, repo_type="dataset", token=token,
                exist_ok=True, private=private)
    print(f"\nUploading {staging} -> {repo_id}", flush=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print(f"\nDone. https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--build-dir",
                    default="/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf")
    ap.add_argument("--staging-dir", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message",
                    default="Initial upload: Code2LoRA snapshots dataset.")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    if not (build_dir / "commits").is_dir() or not (build_dir / "qna").is_dir():
        sys.exit(f"build_dir is missing commits/ or qna/: {build_dir}")

    if args.staging_dir:
        staging = Path(args.staging_dir)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _stage(build_dir, staging)
        if not args.no_upload:
            _push(staging, args.repo_id, args.token,
                  args.private, args.commit_message)
        print(f"\nStaging tree preserved at {staging}")
    else:
        with tempfile.TemporaryDirectory(prefix="hf_push_snap_") as tmp:
            staging = Path(tmp) / "tree"
            _stage(build_dir, staging)
            if not args.no_upload:
                _push(staging, args.repo_id, args.token,
                      args.private, args.commit_message)


if __name__ == "__main__":
    main()
