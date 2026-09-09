#!/usr/bin/env python3
"""Push the Code2LoRA question-answering (``qa`` task) dataset to HF Hub.

The QA dataset supplies LLM-generated (question, answer) pairs *about each
commit's production-code change*. It reuses the GRU v2 commit splits 1:1 -- each
row keeps the commit's ``in_repo_split`` (train/val/test => ir splits) and
``cross_repo_split`` labels -- so it joins onto
``code2lora/...gru-commits-v2`` (or the local ``commit_parquet_hf_v2``) on
``(repo_id, commit_sha)`` and trains alongside ``assert_rhs`` / ``code_gen``.

Layout::

    <repo-id>/
        README.md
        qna/
            train.parquet     train repos (in_repo_split in {train,val,test})
            cr_val.parquet     held-out cr_val repos
            cr_test.parquet    held-out cr_test repos

Usage::

    python scripts/push_qa_dataset_to_hf.py \\
        --repo-id   code2lora/code2lora-data-qa \\
        --build-dir /scratch/lhotsko/REPO_DATASET/qna_qa_hf \\
        --private
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
  - question-answering
language:
  - code
  - en
pretty_name: Code2LoRA Question-Answering Dataset (qa task)
size_categories:
  - 100K<n<1M
configs:
  - config_name: qna
    data_files:
      - split: train
        path: qna/train.parquet
      - split: cr_val
        path: qna/cr_val.parquet
      - split: cr_test
        path: qna/cr_test.parquet
---

# Code2LoRA question-answering dataset (`qa` task)

LLM-generated question/answer pairs grounded in each training repository's
commit history. This is the `qa`-task companion to the Code2LoRA-GRU commit
dataset: every row is keyed by `(repo_id, commit_sha)` and carries the commit's
`in_repo_split` and `cross_repo_split` labels, so it lines up 1:1 with the GRU
v2 commit walk and trains alongside the `assert_rhs` task.

**164,173 unique pairs** over 9,188 commits from 500 training repositories,
in two complementary question families (see the `qa_kind` column):

* **`diff`** — 73,829 pairs prompted with the commit's production-code diff
  (API signature changes, new APIs, behavioral deltas, refactoring rationale).
* **`repo_state`** — 90,344 pairs prompted with the repository's file contents
  at that commit (module purpose, function/class behavior, signatures, return
  values, constants, control flow).

## How it was built

Questions were generated with `gpt-5.4-mini` (OpenAI Batch API) under a fixed
per-family taxonomy prompt. To keep the set balanced and high-quality:

* **Per-repo cap (20), split-stratified** — each repo contributes at most 20
  commits, sampled evenly across its chronological timeline, with a reserved
  minimum from the `val`/`test` in-repo splits so ir-eval is never starved.
* **Adaptive question count (3-20, median 10)** — the count per family per
  commit adapts to the diff / repository size; tiny diffs get a few
  highest-value questions, large changes get the full set.
* **4k-token input cap** — diffs / file contents are token-trimmed so each
  request stays within 4k input tokens.

Total generation cost: **$60.15**.

## Splits

| split | role |
|---|---|
| `train`   | training repos. Filter `in_repo_split` for in-repo eval: `val` -> ir_val, `test` -> ir_test. |
| `cr_val`  | held-out cross-repo validation repos. |
| `cr_test` | held-out cross-repo test repos. |

`ir_val` / `ir_test` are not separate files — derive them from `train` via the
`in_repo_split` column (exactly as the GRU v2 trainer does). The in-repository
evaluation splits are a subset of `train`, so the 164,173 total is not inflated
by counting them twice.

| split (disjoint) | diff | repo_state | total |
|---|---|---|---|
| train (in-repo train) | 40,357 | 48,849 | 89,206 |
| ir_val (in `train`)   |  8,369 | 10,689 | 19,058 |
| ir_test (in `train`)  |  9,333 | 11,698 | 21,031 |
| cr_val                |  7,882 |  9,561 | 17,443 |
| cr_test               |  7,888 |  9,547 | 17,435 |
| **all**               | **73,829** | **90,344** | **164,173** |

## Schema (per row)

| column | type | notes |
|---|---|---|
| `repo_id` | string | `owner/name` |
| `cross_repo_split` | string | `train` / `cr_val` / `cr_test` |
| `commit_index` | int32 | chronological index among kept commits |
| `commit_sha` | string | join key with the commits parquet |
| `in_repo_split` | string | `train` / `val` / `test` |
| `task` | string | always `qa` |
| `question_index` | int32 | position within the commit's question set |
| `category` | string | taxonomy category (per family) |
| `qa_kind` | string | `diff` or `repo_state` |
| `prefix` | large_string | the question (model prompt) |
| `target` | large_string | the answer (training target) |

## Loading

```python
from datasets import load_dataset
qa = load_dataset("code2lora/code2lora-data-qa", "qna")
# in-repo eval splits live inside `train`:
ir_test = qa["train"].filter(lambda r: r["in_repo_split"] == "test")
# pick a question family:
diff_qa = qa["train"].filter(lambda r: r["qa_kind"] == "diff")
```

Join with the GRU v2 commits on `(repo_id, commit_sha)` to attach
`diff_embedding` / `repo_state_embedding` for hypernetwork training.

## Citation

```bibtex
@misc{code2lora_qa_2026,
  title  = {Code2LoRA QA: a commit-grounded question-answering dataset for repository-aware LoRA generation},
  year   = {2026},
  author = {RepoPeftData authors},
}
```
"""


def _stage(build_dir: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    src_qna = build_dir / "qna"
    if not src_qna.is_dir():
        sys.exit(f"build_dir is missing qna/: {build_dir}")
    dst_qna = staging / "qna"
    dst_qna.mkdir(parents=True, exist_ok=True)
    for pq in sorted(src_qna.glob("*.parquet")):
        # ir_val / ir_test are subsets of train.parquet (in_repo_split
        # val/test); skip them so the released total is not double-counted.
        if pq.name.startswith("ir_"):
            print(f"[qna] skipping {pq.name} (derivable from train.parquet)", flush=True)
            continue
        print(f"[qna] staging {pq.name} ...", flush=True)
        shutil.copy2(pq, dst_qna / pq.name)
    (staging / "README.md").write_text(README, encoding="utf-8")


def _push(staging: Path, repo_id: str, token, private: bool,
          commit_message: str) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"\nCreating dataset repo: {repo_id} (private={private})", flush=True)
    create_repo(repo_id=repo_id, repo_type="dataset", token=token,
                exist_ok=True, private=private)
    print(f"Uploading {staging} -> {repo_id}", flush=True)
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
                    default="/scratch/lhotsko/REPO_DATASET/qna_qa_hf_v2")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message",
                    default="Full QA dataset: diff + repository-state (164,173 pairs).")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="hf_push_qa_") as tmp:
        staging = Path(tmp) / "tree"
        _stage(Path(args.build_dir), staging)
        if not args.no_upload:
            _push(staging, args.repo_id, args.token, args.private,
                  args.commit_message)


if __name__ == "__main__":
    main()
