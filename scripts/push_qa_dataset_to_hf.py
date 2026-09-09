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
  - 10K<n<100K
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

LLM-generated question/answer pairs **about each commit's production-code
change**. This is the `qa`-task companion to the Code2LoRA-GRU commit dataset:
every row is keyed by `(repo_id, commit_sha)` and carries the commit's
`in_repo_split` and `cross_repo_split` labels, so it lines up 1:1 with the GRU
v2 commit walk and trains alongside the `assert_rhs` / `code_gen` tasks.

## How it was built

For every kept commit (non-empty `production_code_diff`) the diff was sent to
`gpt-5.4-mini` (OpenAI Batch API) with a fixed taxonomy prompt. To keep the set
balanced and high-quality:

* **Per-repo cap (20), split-stratified** — each repo contributes at most 20
  commits, sampled evenly across its chronological timeline, with a reserved
  minimum from the `val`/`test` in-repo splits so ir-eval is never starved.
* **Diff-size-adaptive question count (3-10)** — tiny diffs get ~3 of the
  highest-value questions; large refactors get the full 10. Categories are
  chosen by priority: `intent, behavior_change, scope, new_api,
  signature_change, usage, refactor, dependency, edge_case, risk`.
* **4k-token input cap** — diffs are token-trimmed so each request stays within
  4k input tokens.

## Splits

| split | role |
|---|---|
| `train`   | training repos. Filter `in_repo_split` for in-repo eval: `val` -> ir_val, `test` -> ir_test. |
| `cr_val`  | held-out cross-repo validation repos. |
| `cr_test` | held-out cross-repo test repos. |

`ir_val` / `ir_test` are not separate files — derive them from `train` via the
`in_repo_split` column (exactly as the GRU v2 trainer does).

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
| `category` | string | one of the 10 taxonomy categories |
| `prefix` | large_string | the question (model prompt) |
| `target` | large_string | the answer (training target) |

## Loading

```python
from datasets import load_dataset
qa = load_dataset("code2lora/code2lora-data-qa", "qna")
# in-repo eval splits live inside `train`:
ir_test = qa["train"].filter(lambda r: r["in_repo_split"] == "test")
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
                    default="/scratch/lhotsko/REPO_DATASET/qna_qa_hf")
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message",
                    default="Initial upload: Code2LoRA QA dataset (~74k pairs).")
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
