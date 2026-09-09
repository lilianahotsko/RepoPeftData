#!/usr/bin/env python3
"""Publish the paper-exact dataset artifacts to the code2lora HF org.

Consumes the tree produced by ``scripts/build_paper_exact_datasets.py``
(default ``$SCRATCH/REPO_DATASET/paper_exact_hf``) and pushes:

1. ``code2lora/code2lora-data-static``  (NEW)
       qna/{train,cr_val,cr_test,ir_val,ir_test,ood_test}.parquet
       repos/{train,cr_val,cr_test}.parquet
   The static track of Table tab:dataset_actual, verbatim:
   409/51/52 repos, 39,612 / 6,213 / 6,414 / 4,833 / 5,222 QnAs,
   plus the 92-repo / 9,942-QnA static OOD slice.

2. ``code2lora/code2lora-data-snapshots``  (UPDATE)
       + qna8/{cr_val,cr_test,ir_val,ir_test}.parquet
   The evolution-track eval suites after the paper's eval-time cap
   (first <=8 QnAs per (repo, commit)): 58,944 / 44,732 / 38,783 / 42,061.

3. ``code2lora/code2lora-data-ood``  (UPDATE)
       + qna8.parquet   (14,813 rows)

4. ``code2lora/code2lora-data-smartcap``  (UPDATE)
       + qna/train_paper.parquet   (215,129 rows, in_repo_split=='train')

Usage::

    python scripts/push_paper_exact_to_hf.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")
SRC = Path(SCRATCH) / "REPO_DATASET" / "paper_exact_hf"

STATIC_README = """---
license: mit
pretty_name: Code2LoRA Static-Track Dataset (paper-exact)
tags: [code, assertion-completion, repository-level]
configs:
- config_name: qna
  data_files:
  - split: train
    path: qna/train.parquet
  - split: cr_val
    path: qna/cr_val.parquet
  - split: cr_test
    path: qna/cr_test.parquet
  - split: ir_val
    path: qna/ir_val.parquet
  - split: ir_test
    path: qna/ir_test.parquet
  - split: ood_test
    path: qna/ood_test.parquet
- config_name: repos
  data_files:
  - split: train
    path: repos/train.parquet
  - split: cr_val
    path: repos/cr_val.parquet
  - split: cr_test
    path: repos/cr_test.parquet
---

# code2lora-data-static — static-track RepoPeftBench (paper-exact)

The **Static track** of RepoPeftBench exactly as consumed by the paper's
main-results table: one anchor snapshot per repository, quality-filtered
assertion-completion QnAs, split into train / cross-repo (CR) /
in-repo (IR) suites. These are the splits behind Code2LoRA-Static and all
static-track baselines (pretrained, FFT, sLoRA, per-repo LoRA, Text2LoRA,
RAG, DRC).

## Row counts (identical to the paper's dataset table)

| split | repos | QnAs |
|---|---:|---:|
| `train`    | 409 | 39,612 |
| `cr_val`   |  51 |  6,213 |
| `cr_test`  |  52 |  6,414 |
| `ir_val`   | 409 |  4,833 |
| `ir_test`  | 409 |  5,222 |
| `ood_test` |  92 |  9,942 |

Total in-distribution: 62,294 QnAs over 512 repositories. `ir_val` /
`ir_test` reuse the 409 training repositories with held-out test
functions (random 8:1:1 protocol). `ood_test` is the static prefix used
by the prompt-based OOD rows.

## Configs

* **`qna`** — one row per assertion pair:
  `repo_id, prefix, target, assertion_type, difficulty, test_file,
  test_function, test_class, lineno, commit_idx`.
* **`repos`** — one row per repository with the frozen 2048-d
  Qwen3-Embedding repo-state vector (`repo_id, embedding`) consumed by the
  static hypernetwork (train / cr_val / cr_test; IR suites reuse the
  train-repo embeddings).

## Loading

```python
from datasets import load_dataset
qna   = load_dataset("code2lora/code2lora-data-static", "qna")
repos = load_dataset("code2lora/code2lora-data-static", "repos")
```

## Companion datasets

* `code2lora/code2lora-data-snapshots` evolution-track anchor snapshots + eval suites
* `code2lora/code2lora-data-commits`   per-commit metadata + diff embeddings (GRU)
* `code2lora/code2lora-data-smartcap`  smart-capped GRU training QnAs
* `code2lora/code2lora-data-ood`       post-cutoff OOD commit-derived split
"""

SNAPSHOTS_QNA8_SECTION = """
## Paper-exact capped eval suites (`qna8/`)

Every evolution-track number in the paper is computed after an
**eval-time cap: the first <=8 QnAs per (repo, commit) group in file row
order** (`--qnas-per-commit-limit 8` in the evaluators). The `qna/` files
above are the full uncapped supersets; `qna8/` ships the capped versions
verbatim, so the paper's dataset table can be reproduced by `wc -l`:

| file | rows | paper split |
|---|---:|---|
| `qna8/cr_val.parquet`  | 58,944 | Evolution CR Val  |
| `qna8/cr_test.parquet` | 44,732 | Evolution CR Test |
| `qna8/ir_val.parquet`  | 38,783 | Evolution IR Val  |
| `qna8/ir_test.parquet` | 42,061 | Evolution IR Test |

(The OOD counterpart, 14,813 rows, lives in
`code2lora/code2lora-data-ood` as `qna8.parquet`.)
"""

OOD_QNA8_SECTION = """
## Paper-exact capped eval file (`qna8.parquet`)

The paper scores OOD with an eval-time cap of the **first <=8 QnAs per
(repo, commit) group** in file row order. `qna.parquet` is the full
uncapped superset (181,500 rows); `qna8.parquet` is the capped version
actually consumed: **14,813 rows over 1,950 commits / 92 repos**, matching
the OOD Test row of the paper's dataset table.
"""

SMARTCAP_TRAIN_SECTION = """
## Paper-exact training file (`qna/train_paper.parquet`)

`qna/cr_train.parquet` mixes the capped training rows with uncapped
in-repo val/test passthrough rows. `qna/train_paper.parquet` is the
pre-filtered `in_repo_split == "train"` subset: **215,129 rows over
45,516 commits / 400 repos** — exactly the Code2LoRA-Evo training set
reported in the paper's dataset table.
"""


def _check(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((SRC / "PAPER_EXACT_README.json").read_text())
    print("Manifest row counts:")
    for k, v in manifest["files"].items():
        print(f"  {k}: {v['rows']}")

    api = HfApi()
    print(f"\nAuthenticated as: {api.whoami()['name']}")

    # ---- 1. New dataset: code2lora-data-static ----
    rid = "code2lora/code2lora-data-static"
    print(f"\n=== {rid} (create + upload) ===")
    if not args.dry_run:
        create_repo(rid, repo_type="dataset", exist_ok=True, private=False)
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            (staging / "README.md").write_text(STATIC_README)
            api.upload_folder(folder_path=str(staging), repo_id=rid,
                              repo_type="dataset",
                              commit_message="Add dataset card")
        api.upload_folder(
            folder_path=str(_check(SRC / "static")),
            repo_id=rid, repo_type="dataset",
            commit_message="Paper-exact static-track splits "
                           "(62,294 in-dist + 9,942 OOD QnAs)")
        api.upload_file(
            path_or_fileobj=str(SRC / "PAPER_EXACT_README.json"),
            path_in_repo="PAPER_EXACT_README.json",
            repo_id=rid, repo_type="dataset",
            commit_message="Provenance manifest")
    print(f"  -> https://huggingface.co/datasets/{rid}")

    # ---- 2. Update snapshots with qna8/ ----
    rid = "code2lora/code2lora-data-snapshots"
    print(f"\n=== {rid} (add qna8/) ===")
    if not args.dry_run:
        for suite in ("cr_val", "cr_test", "ir_val", "ir_test"):
            api.upload_file(
                path_or_fileobj=str(_check(SRC / "qna8" / f"{suite}.parquet")),
                path_in_repo=f"qna8/{suite}.parquet",
                repo_id=rid, repo_type="dataset",
                commit_message=f"Add paper-exact 8-capped eval suite "
                               f"qna8/{suite}.parquet")
        readme = api.hf_hub_download(rid, "README.md", repo_type="dataset")
        text = Path(readme).read_text()
        if "qna8/" not in text:
            text += SNAPSHOTS_QNA8_SECTION
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "README.md"
                p.write_text(text)
                api.upload_file(path_or_fileobj=str(p),
                                path_in_repo="README.md",
                                repo_id=rid, repo_type="dataset",
                                commit_message="Document qna8/ capped suites")
    print(f"  -> https://huggingface.co/datasets/{rid}")

    # ---- 3. Update ood with qna8.parquet ----
    rid = "code2lora/code2lora-data-ood"
    print(f"\n=== {rid} (add qna8.parquet) ===")
    if not args.dry_run:
        api.upload_file(
            path_or_fileobj=str(_check(SRC / "qna8" / "ood_test.parquet")),
            path_in_repo="qna8.parquet",
            repo_id=rid, repo_type="dataset",
            commit_message="Add paper-exact 8-capped OOD eval file "
                           "(14,813 rows)")
        readme = api.hf_hub_download(rid, "README.md", repo_type="dataset")
        text = Path(readme).read_text()
        if "qna8.parquet" not in text:
            text += OOD_QNA8_SECTION
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "README.md"
                p.write_text(text)
                api.upload_file(path_or_fileobj=str(p),
                                path_in_repo="README.md",
                                repo_id=rid, repo_type="dataset",
                                commit_message="Document qna8.parquet")
    print(f"  -> https://huggingface.co/datasets/{rid}")

    # ---- 4. Update smartcap with train_paper.parquet ----
    rid = "code2lora/code2lora-data-smartcap"
    print(f"\n=== {rid} (add qna/train_paper.parquet) ===")
    if not args.dry_run:
        api.upload_file(
            path_or_fileobj=str(_check(SRC / "smartcap" / "train_paper.parquet")),
            path_in_repo="qna/train_paper.parquet",
            repo_id=rid, repo_type="dataset",
            commit_message="Add paper-exact C2L-Evo training subset "
                           "(215,129 rows, in_repo_split=='train')")
        readme = api.hf_hub_download(rid, "README.md", repo_type="dataset")
        text = Path(readme).read_text()
        if "train_paper.parquet" not in text:
            text += SMARTCAP_TRAIN_SECTION
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "README.md"
                p.write_text(text)
                api.upload_file(path_or_fileobj=str(p),
                                path_in_repo="README.md",
                                repo_id=rid, repo_type="dataset",
                                commit_message="Document train_paper.parquet")
    print(f"  -> https://huggingface.co/datasets/{rid}")

    print("\nDone.")


if __name__ == "__main__":
    main()
