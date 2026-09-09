#!/usr/bin/env python3
"""Create four NEW self-contained, paper-named datasets on the code2lora org.

These contain ONLY the post-cap / paper-used data, one dataset per training
scenario. They do not touch any existing dataset.

Scenario -> HF repo id:
  1) Static track (single snapshot)      -> code2lora/code2lora-static
  2) GRU / Code2LoRA-Evo (all commits)   -> code2lora/code2lora-evo
  3) Code2LoRA-Static anchor (commit ds) -> code2lora/code2lora-static-anchor
  4) OOD holdout (static + evo)           -> code2lora/repopeftbench-ood

Every parquet is verified against the paper's Table tab:dataset_actual row
counts before upload. Files are uploaded directly from their source paths
(no on-disk duplication of the large embedding parquets).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")
DS = Path(SCRATCH) / "REPO_DATASET"
PE = DS / "paper_exact_hf"                       # built by build_paper_exact_datasets.py
GRU_COMMITS = DS / "commit_parquet_hf_v2/commits"    # diff + repo_state per commit
ANCHOR_COMMITS = DS / "code2lora_snapshots_hf/commits"  # repo_state per commit
ANCHOR_QNA = DS / "code2lora_snapshots_hf/qna"          # anchor train QnAs

# path_in_repo -> (source_path, expected_rows)
DATASETS: dict[str, dict] = {
    "code2lora/code2lora-static": {
        "files": {
            "qna/train.parquet": (PE / "static/qna/train.parquet", 39612),
            "qna/cr_val.parquet": (PE / "static/qna/cr_val.parquet", 6213),
            "qna/cr_test.parquet": (PE / "static/qna/cr_test.parquet", 6414),
            "qna/ir_val.parquet": (PE / "static/qna/ir_val.parquet", 4833),
            "qna/ir_test.parquet": (PE / "static/qna/ir_test.parquet", 5222),
            "repos/train.parquet": (PE / "static/repos/train.parquet", 409),
            "repos/cr_val.parquet": (PE / "static/repos/cr_val.parquet", 51),
            "repos/cr_test.parquet": (PE / "static/repos/cr_test.parquet", 52),
        },
        "readme": "STATIC",
    },
    "code2lora/code2lora-evo": {
        "files": {
            "commits/train.parquet": (GRU_COMMITS / "train.parquet", 58617),
            "commits/cr_val.parquet": (GRU_COMMITS / "cr_val.parquet", 8614),
            "commits/cr_test.parquet": (GRU_COMMITS / "cr_test.parquet", 6618),
            "qna/train.parquet": (PE / "smartcap/train_paper.parquet", 215129),
            "qna/cr_val.parquet": (PE / "qna8/cr_val.parquet", 58944),
            "qna/cr_test.parquet": (PE / "qna8/cr_test.parquet", 44732),
            "qna/ir_val.parquet": (PE / "qna8/ir_val.parquet", 38783),
            "qna/ir_test.parquet": (PE / "qna8/ir_test.parquet", 42061),
        },
        "readme": "EVO",
    },
    "code2lora/code2lora-static-anchor": {
        "files": {
            "commits/train.parquet": (ANCHOR_COMMITS / "train.parquet", 400),
            "commits/ir_val.parquet": (ANCHOR_COMMITS / "ir_val.parquet", 5710),
            "commits/ir_test.parquet": (ANCHOR_COMMITS / "ir_test.parquet", 6179),
            "commits/cr_val.parquet": (ANCHOR_COMMITS / "cr_val.parquet", 8614),
            "commits/cr_test.parquet": (ANCHOR_COMMITS / "cr_test.parquet", 6618),
            "qna/train.parquet": (ANCHOR_QNA / "train.parquet", 44149),
            "qna/cr_val.parquet": (PE / "qna8/cr_val.parquet", 58944),
            "qna/cr_test.parquet": (PE / "qna8/cr_test.parquet", 44732),
            "qna/ir_val.parquet": (PE / "qna8/ir_val.parquet", 38783),
            "qna/ir_test.parquet": (PE / "qna8/ir_test.parquet", 42061),
        },
        "readme": "ANCHOR",
    },
    "code2lora/repopeftbench-ood": {
        "files": {
            "commits/ood_test.parquet": (GRU_COMMITS / "ood_test.parquet", 1950),
            "qna_static/ood_test.parquet": (PE / "static/qna/ood_test.parquet", 9942),
            "qna_evo/ood_test.parquet": (PE / "qna8/ood_test.parquet", 14813),
        },
        "readme": "OOD",
    },
}

README_STATIC = """---
license: mit
pretty_name: Code2LoRA-Static (static track, single snapshot)
tags: [code, assertion-completion, repository-level]
configs:
- config_name: qna
  data_files:
  - {split: train, path: qna/train.parquet}
  - {split: cr_val, path: qna/cr_val.parquet}
  - {split: cr_test, path: qna/cr_test.parquet}
  - {split: ir_val, path: qna/ir_val.parquet}
  - {split: ir_test, path: qna/ir_test.parquet}
- config_name: repos
  data_files:
  - {split: train, path: repos/train.parquet}
  - {split: cr_val, path: repos/cr_val.parquet}
  - {split: cr_test, path: repos/cr_test.parquet}
---

# Code2LoRA-Static — static track (single snapshot)

The **Static track** of RepoPeftBench exactly as consumed by the paper's
main-results table (`Code2LoRA-Static`, single anchor snapshot per
repository). Contains only the paper-used QnAs (post quality-filter). For
the out-of-distribution slice see `code2lora/repopeftbench-ood`.

| config | split | rows |
|---|---|---:|
| qna | train | 39,612 |
| qna | cr_val | 6,213 |
| qna | cr_test | 6,414 |
| qna | ir_val | 4,833 |
| qna | ir_test | 5,222 |
| repos | train / cr_val / cr_test | 409 / 51 / 52 |

* `qna` — one row per assertion pair (`repo_id, prefix, target,
  assertion_type, difficulty, test_file, test_function, test_class,
  lineno, commit_idx`).
* `repos` — frozen 2048-d Qwen3-Embedding repo-state vector per repo
  (`repo_id, embedding`); IR suites reuse the train-repo embeddings.

```python
from datasets import load_dataset
qna   = load_dataset("code2lora/code2lora-static", "qna")
repos = load_dataset("code2lora/code2lora-static", "repos")
```

Companion datasets: `code2lora/code2lora-evo`,
`code2lora/code2lora-static-anchor`, `code2lora/repopeftbench-ood`.
"""

README_EVO = """---
license: mit
pretty_name: Code2LoRA-Evo (evolution track, GRU)
tags: [code, assertion-completion, repository-level]
configs:
- config_name: commits
  data_files:
  - {split: train, path: commits/train.parquet}
  - {split: cr_val, path: commits/cr_val.parquet}
  - {split: cr_test, path: commits/cr_test.parquet}
- config_name: qna
  data_files:
  - {split: train, path: qna/train.parquet}
  - {split: cr_val, path: qna/cr_val.parquet}
  - {split: cr_test, path: qna/cr_test.parquet}
  - {split: ir_val, path: qna/ir_val.parquet}
  - {split: ir_test, path: qna/ir_test.parquet}
---

# Code2LoRA-Evo — evolution track (GRU)

Everything the **Code2LoRA-Evo** GRU hypernetwork consumes, post-cap and
self-contained. The GRU walks per-commit `diff_embedding` /
`repo_state_embedding` and is trained with the smart-capped
(`<=8`/commit, `<=4`/file) training QnAs; evaluation uses the paper's
eval-time cap (first `<=8` QnAs per (repo, commit)).

| config | split | rows | role |
|---|---|---:|---|
| commits | train | 58,617 | per-commit diff + repo-state embeddings |
| commits | cr_val / cr_test | 8,614 / 6,618 | held-out repo commit walks |
| qna | train | 215,129 | smart-capped training loss (`Code2LoRA-Evo`) |
| qna | cr_val | 58,944 | capped eval (`<=8`/commit) |
| qna | cr_test | 44,732 | capped eval |
| qna | ir_val | 38,783 | capped eval |
| qna | ir_test | 42,061 | capped eval |

Join `commits` and `qna` on `(repo_id, commit_sha)`.

```python
from datasets import load_dataset
commits = load_dataset("code2lora/code2lora-evo", "commits")
qna     = load_dataset("code2lora/code2lora-evo", "qna")
```

Companion datasets: `code2lora/code2lora-static`,
`code2lora/code2lora-static-anchor`, `code2lora/repopeftbench-ood`.
"""

README_ANCHOR = """---
license: mit
pretty_name: Code2LoRA-Static anchor (evolution track)
tags: [code, assertion-completion, repository-level]
configs:
- config_name: commits
  data_files:
  - {split: train, path: commits/train.parquet}
  - {split: ir_val, path: commits/ir_val.parquet}
  - {split: ir_test, path: commits/ir_test.parquet}
  - {split: cr_val, path: commits/cr_val.parquet}
  - {split: cr_test, path: commits/cr_test.parquet}
- config_name: qna
  data_files:
  - {split: train, path: qna/train.parquet}
  - {split: cr_val, path: qna/cr_val.parquet}
  - {split: cr_test, path: qna/cr_test.parquet}
  - {split: ir_val, path: qna/ir_val.parquet}
  - {split: ir_test, path: qna/ir_test.parquet}
---

# Code2LoRA-Static (anchor) — evolution track

The direct-projection **Code2LoRA-Static** variant reported in the paper's
evolution-track / per-commit results: one anchor snapshot per repository
with a 2048-d `repo_state_embedding`, no GRU rollout. It shares the
evolution track's capped eval suites with `code2lora/code2lora-evo`.

| config | split | rows | role |
|---|---|---:|---|
| commits | train | 400 | anchor commit repo-state embeddings |
| commits | ir_val / ir_test | 5,710 / 6,179 | per-commit repo-state (eval) |
| commits | cr_val / cr_test | 8,614 / 6,618 | per-commit repo-state (eval) |
| qna | train | 44,149 | anchor training QnAs (`Code2LoRA-Static`) |
| qna | cr_val / cr_test | 58,944 / 44,732 | capped eval (`<=8`/commit) |
| qna | ir_val / ir_test | 38,783 / 42,061 | capped eval |

```python
from datasets import load_dataset
commits = load_dataset("code2lora/code2lora-static-anchor", "commits")
qna     = load_dataset("code2lora/code2lora-static-anchor", "qna")
```

Companion datasets: `code2lora/code2lora-static`,
`code2lora/code2lora-evo`, `code2lora/repopeftbench-ood`.
"""

README_OOD = """---
license: mit
pretty_name: RepoPeftBench OOD holdout (static + evolution)
tags: [code, assertion-completion, repository-level, ood]
configs:
- config_name: commits
  data_files:
  - {split: ood_test, path: commits/ood_test.parquet}
- config_name: qna_static
  data_files:
  - {split: ood_test, path: qna_static/ood_test.parquet}
- config_name: qna_evo
  data_files:
  - {split: ood_test, path: qna_evo/ood_test.parquet}
---

# RepoPeftBench — OOD holdout (static + evolution)

Post-cutoff out-of-distribution repositories, holding both evaluation
views used in the paper's OOD table, post-cap and self-contained.

| config | split | rows | role |
|---|---|---:|---|
| commits | ood_test | 1,950 | per-commit diff + repo-state embeddings (GRU / anchor) |
| qna_static | ood_test | 9,942 | static-track prompts (prompt-based OOD rows) |
| qna_evo | ood_test | 14,813 | commit-derived capped eval (`<=8`/commit) |

```python
from datasets import load_dataset
commits    = load_dataset("code2lora/repopeftbench-ood", "commits")
qna_static = load_dataset("code2lora/repopeftbench-ood", "qna_static")
qna_evo    = load_dataset("code2lora/repopeftbench-ood", "qna_evo")
```

Companion datasets: `code2lora/code2lora-static`,
`code2lora/code2lora-evo`, `code2lora/code2lora-static-anchor`.
"""

READMES = {"STATIC": README_STATIC, "EVO": README_EVO,
           "ANCHOR": README_ANCHOR, "OOD": README_OOD}


def verify_all() -> None:
    print("Verifying source row counts...")
    ok = True
    for rid, spec in DATASETS.items():
        for path_in_repo, (src, exp) in spec["files"].items():
            if not src.exists():
                print(f"  MISSING {rid}:{path_in_repo} <- {src}")
                ok = False
                continue
            n = pq.ParquetFile(str(src)).metadata.num_rows
            status = "OK" if n == exp else "MISMATCH"
            if n != exp:
                ok = False
            print(f"  [{status}] {rid}:{path_in_repo} rows={n} exp={exp}")
    if not ok:
        raise SystemExit("Verification failed; aborting.")
    print("All source files verified.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="Only push this repo id (for retries).")
    args = ap.parse_args()

    verify_all()
    if args.dry_run:
        print("Dry run: no repos created or uploaded.")
        return

    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}\n")

    for rid, spec in DATASETS.items():
        if args.only and rid != args.only:
            continue
        print(f"=== {rid} ===")
        create_repo(rid, repo_type="dataset", exist_ok=True, private=False)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "README.md"
            p.write_text(READMES[spec["readme"]])
            api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md",
                            repo_id=rid, repo_type="dataset",
                            commit_message="Add dataset card")
        for path_in_repo, (src, _exp) in spec["files"].items():
            print(f"  uploading {path_in_repo} ...", flush=True)
            api.upload_file(path_or_fileobj=str(src),
                            path_in_repo=path_in_repo,
                            repo_id=rid, repo_type="dataset",
                            commit_message=f"Add {path_in_repo}")
        print(f"  -> https://huggingface.co/datasets/{rid}\n")

    print("Done.")


if __name__ == "__main__":
    main()
