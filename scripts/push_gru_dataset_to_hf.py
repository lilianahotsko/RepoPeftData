#!/usr/bin/env python3
"""Push the Code2LoRA-GRU train/val/test parquet splits to the HuggingFace Hub.

Default layout pushed (paper-grade setup; the GRU is trained on the smart-cap
QnA filter)::

    <repo-id>/
        README.md
        commits/
            train.parquet      # filtered production-code diffs per kept commit
            cr_val.parquet
            cr_test.parquet
        qna/
            train.parquet      # smart-cap filtered QnA pairs (paper-grade)
            cr_val.parquet     # canonical cross-repo val QnAs
            cr_test.parquet    # canonical cross-repo test QnAs
        splits/
            *.json             # repo-level split definitions
        SMART_CAP_README.json  # provenance of the smart-cap snapshot

All symlinks in the source directory are resolved into concrete files before
upload, so the published dataset is self-contained.

Usage::

    huggingface-cli login   # one time
    python scripts/push_gru_dataset_to_hf.py \
        --repo-id nanigock/repopeft-gru-commits \
        --smartcap-dir /scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap \
        --base-dir     /scratch/lhotsko/REPO_DATASET/commit_parquet_hf

The script avoids re-uploading files whose remote SHA already matches the
local one (HF's hashing-based dedup), so re-running it is cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


# A self-contained README rendered into the HF dataset card.
README = """---
license: mit
task_categories:
  - text-generation
  - feature-extraction
language:
  - code
pretty_name: Code2LoRA-GRU Commit-Level Dataset (smart-cap)
size_categories:
  - 100K<n<1M
configs:
  - config_name: commits
    data_files:
      - split: train
        path: commits/train.parquet
      - split: cr_val
        path: commits/cr_val.parquet
      - split: cr_test
        path: commits/cr_test.parquet
  - config_name: qna
    data_files:
      - split: train
        path: qna/train.parquet
      - split: cr_val
        path: qna/cr_val.parquet
      - split: cr_test
        path: qna/cr_test.parquet
---

# Code2LoRA-GRU commit-level dataset (smart-cap)

Per-repository commit sequences with test-file assertion QnA pairs, used to
train the **Code2LoRA-GRU<sub>commit</sub>** hypernetwork. The model
ingests the filtered production-code diff at each commit, evolves a GRU
hidden state across the commit history, and at each step generates a LoRA
that adapts a base LLM (`Qwen/Qwen2.5-Coder-1.5B`) to the repository's
current state. Targets are RHSs of newly-introduced assertions at each
commit.

## Splits

* **Cross-repo split** (`cross_repo_split` ∈ {train, cr_val, cr_test}):
  Each repository is assigned to exactly one of these sets. Held-out
  repositories are never seen during training.
* **In-repo split** (`in_repo_split` ∈ {train, val, test}): For each repo
  the kept commits are chopped 80 / 10 / 10 chronologically. Only the
  `train` slice contributes to the LoRA loss during training; the GRU is
  still allowed to ingest every diff in order.

## Smart-cap filtering

The `qna/train.parquet` shipped here is the **smart-cap snapshot** of the
raw QnA table. It applies three additive filters on the cross-repo
training set:
  1. Drop trivial targets (single-token bools / Nones / empty strings).
  2. Per-(`test_file`, `test_function`) round-robin: at most
     `max_per_function = 4` assertions per function.
  3. Per-commit hard cap: at most `max_per_commit = 8` distinct (file,
     function, target) triples per commit.
The `qna/cr_val.parquet` and `qna/cr_test.parquet` files are unfiltered
(verbatim copies of the canonical RepoPeftBench QnAs) so that cross-repo
evaluation numbers stay comparable across model versions.

## Schemas

### commits

| column | type | description |
|---|---|---|
| repo_id | string | `<owner>/<repo>` |
| cross_repo_split | string | train / cr_val / cr_test |
| commit_index | int32 | 0-based index within the kept sequence |
| commit_sha | string | git SHA of this kept commit |
| commit_timestamp | string | ISO 8601 |
| in_repo_split | string | train / val / test (chronological 80/10/10) |
| production_code_diff | large_string | filtered unified diff vs prev kept commit (test hunks removed) |
| n_new_assertions | int32 | number of assertion events introduced |
| n_added_assertions | int32 | events newly added at this commit |
| n_modified_assertions | int32 | events modified at this commit |

### qna

| column | type | description |
|---|---|---|
| repo_id | string | `<owner>/<repo>` |
| commit_sha | string | commit at which this assertion event lives |
| commit_index | int32 | matches the commits row |
| in_repo_split | string | train / val / test |
| cross_repo_split | string | train / cr_val / cr_test |
| test_file | string | path inside the repo, posix style |
| test_function | string | enclosing pytest / unittest function name |
| prefix | large_string | structured context: imports + class + helpers + test body up to the assertion |
| target | large_string | the assertion RHS to predict |
| assertion_type | string | pytest / unittest / numpy / etc. |
| assertion_event | string | `added` or `modified` |
| difficulty | string | easy / medium / hard (heuristic) |

## Usage

```python
from datasets import load_dataset

commits = load_dataset("nanigock/repopeft-gru-commits", "commits")
qna     = load_dataset("nanigock/repopeft-gru-commits", "qna")

print(commits)
# DatasetDict({train, cr_val, cr_test})
print(qna["train"][0]["prefix"][:200])
```

## Training code

The model and training script live at
[github.com/lilianahotsko/RepoPeftData](https://github.com/lilianahotsko/RepoPeftData).
See `scripts/slurm/train_code2lora_gru_commits.sh` for the SLURM launcher
and `hypernetwork/train_code2lora_gru_commits.py` for the trainer.

## Citation

```bibtex
@misc{repopeft_gru_commits_2026,
  title  = {Code2LoRA-GRU: A commit-sequential hypernetwork for repository-aware LoRA adaptation},
  year   = {2026},
  author = {RepoPeftData authors},
}
```
"""


def _materialize_dir(src: Path, dst: Path, follow_symlinks: bool = True) -> None:
    """Copy src into dst, resolving symlinks to their concrete content."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.is_symlink() and follow_symlinks:
            real = entry.resolve()
            if real.is_file():
                shutil.copy2(real, target)
                print(f"  resolved symlink: {entry.name} -> {real}")
            elif real.is_dir():
                _materialize_dir(real, target, follow_symlinks=True)
        elif entry.is_dir():
            _materialize_dir(entry, target, follow_symlinks=True)
        elif entry.is_file():
            shutil.copy2(entry, target)
        else:
            print(f"  skip (unknown type): {entry}")


def _stage_for_upload(smartcap_dir: Path,
                      base_dir: Path,
                      staging: Path,
                      include_base_qna: bool = True) -> None:
    """Build a flattened directory tree at ``staging`` that the HF Hub uploader
    can consume verbatim. The smart-cap directory has symlinks pointing back to
    the base parquet; we resolve them so the published dataset is standalone."""
    print(f"Staging at {staging}", flush=True)
    staging.mkdir(parents=True, exist_ok=True)

    # 1. commits/ comes from the base parquet (smart-cap only changes qna).
    src_commits = (smartcap_dir / "commits")
    if src_commits.is_symlink():
        src_commits = src_commits.resolve()
    if not src_commits.is_dir():
        src_commits = base_dir / "commits"
    print(f"\n[commits] copying from {src_commits} ...", flush=True)
    _materialize_dir(src_commits, staging / "commits")

    # 2. qna/ — smart-cap train.parquet + symlinked cr_val.parquet / cr_test.parquet
    print(f"\n[qna] resolving from {smartcap_dir / 'qna'} ...", flush=True)
    _materialize_dir(smartcap_dir / "qna", staging / "qna")

    # 3. splits/ if present
    src_splits = smartcap_dir / "splits"
    if src_splits.is_symlink():
        src_splits = src_splits.resolve()
    if src_splits.is_dir():
        print(f"\n[splits] copying from {src_splits} ...", flush=True)
        _materialize_dir(src_splits, staging / "splits")

    # 4. Smart-cap provenance JSON
    smart_readme = smartcap_dir / "SMART_CAP_README.json"
    if smart_readme.exists():
        shutil.copy2(smart_readme, staging / "SMART_CAP_README.json")

    # 5. Dataset card
    (staging / "README.md").write_text(README, encoding="utf-8")


def _push_to_hub(staging: Path, repo_id: str, token: str | None,
                 private: bool, commit_message: str) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"\nCreating dataset repo: {repo_id} (private={private})", flush=True)
    create_repo(repo_id=repo_id, repo_type="dataset", token=token,
                exist_ok=True, private=private)

    print(f"\nUploading {staging} -> {repo_id}\n  (this can take a while; "
          f"HF dedups by SHA on retries)", flush=True)
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
    ap.add_argument("--repo-id", required=True,
                    help="HF dataset repo, e.g. 'nanigock/repopeft-gru-commits'.")
    ap.add_argument("--smartcap-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf_smartcap",
                    help="Source smart-cap parquet directory.")
    ap.add_argument("--base-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf",
                    help="Source base parquet directory (used for commits/).")
    ap.add_argument("--staging-dir", default=None,
                    help="Where to materialize the flattened tree. Defaults "
                         "to a tempdir; pass a path on a big disk if you "
                         "want to keep the staging tree around.")
    ap.add_argument("--token", default=None,
                    help="HF token (default: env HF_TOKEN or huggingface-cli login).")
    ap.add_argument("--private", action="store_true",
                    help="Create as a private dataset.")
    ap.add_argument("--commit-message", default="Initial upload: smart-cap GRU dataset.")
    ap.add_argument("--no-upload", action="store_true",
                    help="Stage only; skip the actual HF push (for dry runs).")
    args = ap.parse_args()

    smartcap_dir = Path(args.smartcap_dir)
    base_dir = Path(args.base_dir)
    if not smartcap_dir.is_dir():
        sys.exit(f"smartcap dir not found: {smartcap_dir}")
    if not base_dir.is_dir():
        sys.exit(f"base dir not found: {base_dir}")

    if args.staging_dir:
        staging = Path(args.staging_dir)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _stage_for_upload(smartcap_dir, base_dir, staging)
        if not args.no_upload:
            _push_to_hub(staging, args.repo_id, args.token,
                         args.private, args.commit_message)
        print(f"\nStaging tree preserved at {staging}")
    else:
        with tempfile.TemporaryDirectory(prefix="hf_push_") as tmp:
            staging = Path(tmp) / "tree"
            _stage_for_upload(smartcap_dir, base_dir, staging)
            if not args.no_upload:
                _push_to_hub(staging, args.repo_id, args.token,
                             args.private, args.commit_message)


if __name__ == "__main__":
    main()
