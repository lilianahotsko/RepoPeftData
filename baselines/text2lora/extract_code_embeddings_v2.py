#!/usr/bin/env python3
"""Extract per-repo code embeddings (Qwen3-Embedding, 2048-dim) from the
**v2 commit-derived dataset** for Text2LoRA Code-SFT training.

Source: ``$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/*.parquet``,
which already stores ``repo_state_embedding`` (fp16, [2048]) per
(repo, commit). Text2LoRA conditions on **one** vector per repo, so we
pick a canonical commit per repo (see policy below) and write a single
``.pt`` keyed by ``slug(repo_id) = repo_id.replace('/', '__')``.

Canonical-commit policy
-----------------------
* If the repo appears in ``commits/train.parquet`` (the 400 V2-train
  anchor commits) we use **that** row's embedding -- that is the same
  representation the trained hypernet learnt to map.
* Otherwise (held-out CR-val / CR-test repos) we use the embedding from
  the **max commit_index** row in the suite that contains the repo,
  in this priority order: ``cr_val`` -> ``cr_test`` -> ``ir_val`` ->
  ``ir_test``. This mirrors how a v1 "HEAD" embedding would have been
  chosen.

Outputs
-------
* ``--output`` (e.g. ``$SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt``):
  ``{repo_slug: torch.Tensor[1, 2048] (fp32)}``. Shape matches the v1
  ``extract_code_embeddings.py`` artifact, so the existing SFT trainer
  and evaluator code can ingest it unchanged.
* ``<output>.train_repos.txt``: one ``repo_id`` per line for the repos
  that appear in ``commits/train.parquet`` -- this is the exact V2 train
  set (used downstream as the SFT trainer's de-facto repo list).

Usage
-----
    python baselines/text2lora/extract_code_embeddings_v2.py \
        --commits-dir $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits \
        --output      $SCRATCH/TEXT2LORA_DATA/code_embeddings_v2.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def _materialize_embedding_column(table) -> np.ndarray:
    """Convert the ``repo_state_embedding`` fixed-size-list column to a
    contiguous ``np.ndarray[N, 2048] (float32)``.

    The parquet stores fp16 fixed-size lists; PyArrow surfaces them as
    Python lists of floats. We go through ``.to_numpy()`` (when
    available) for speed and fall back to a Python-list cast otherwise.
    """
    col = table.column("repo_state_embedding")
    # Try the fast path: fixed_size_list[half] -> view as ndarray.
    try:
        values = col.combine_chunks().values
        # values is a FloatArray (half precision); to_numpy(zero_copy_only=False)
        # returns fp16 ndarray.
        flat = values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(col[0].as_py())
        return flat.reshape(len(col), dim)
    except Exception:
        pass
    # Slow path
    dim = len(col[0].as_py())
    out = np.empty((len(col), dim), dtype=np.float32)
    for i, v in enumerate(col.to_pylist()):
        out[i] = v
    return out


def _load_canonical_emb_per_repo(
    parquet_path: Path,
    pick: str = "max_commit_index",
) -> Dict[str, np.ndarray]:
    """Read one commit parquet, return ``{repo_id: emb (fp32, [dim])}``
    for the canonical row per repo (default: row with the highest
    ``commit_index``).
    """
    tbl = pq.read_table(
        str(parquet_path),
        columns=["repo_id", "commit_index", "repo_state_embedding"],
    )
    repos = tbl["repo_id"].to_pylist()
    idxs = tbl["commit_index"].to_pylist()
    embs = _materialize_embedding_column(tbl)

    best_idx: Dict[str, int] = {}
    best_ci: Dict[str, int] = {}
    for i, (r, ci) in enumerate(zip(repos, idxs)):
        if not r:
            continue
        ci_int = int(ci) if ci is not None else -1
        if r not in best_ci or ci_int > best_ci[r]:
            best_ci[r] = ci_int
            best_idx[r] = i
    out: Dict[str, np.ndarray] = {}
    for r, i in best_idx.items():
        out[r] = embs[i].astype(np.float32, copy=False)
    return out


def main() -> None:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--commits-dir",
        default=f"{scratch}/REPO_DATASET/code2lora_snapshots_hf/commits",
        help="Directory with v2 snapshot commit parquets.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to write the .pt file with "
             "{repo_slug: tensor[1, 2048]}.",
    )
    ap.add_argument(
        "--include-suites", nargs="+",
        default=["train", "cr_val", "cr_test", "ir_val", "ir_test"],
        help="Suites to scan, in priority order. Train wins ties; later "
             "suites only contribute repos that the earlier suites do "
             "not already cover.",
    )
    args = ap.parse_args()

    commits_dir = Path(args.commits_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[args] commits_dir = {commits_dir}")
    print(f"[args] output     = {output}")
    print(f"[args] suites     = {args.include_suites}", flush=True)

    if not commits_dir.exists():
        raise SystemExit(f"commits dir not found: {commits_dir}")

    emb_dict: Dict[str, torch.Tensor] = {}
    train_repos: List[str] = []
    per_suite_repos: Dict[str, int] = {}
    for suite in args.include_suites:
        path = commits_dir / f"{suite}.parquet"
        if not path.exists():
            print(f"  [skip] missing {path}", flush=True)
            continue
        per_repo = _load_canonical_emb_per_repo(path)
        n_new = 0
        for r, emb in per_repo.items():
            # Key by the slugged form (``foo__bar``) to match the v1
            # ``code_embeddings.pt`` convention -- both
            # ``baselines/text2lora/train_code_sft.py`` and
            # ``evaluation/run_baselines_v2.py`` look up by slug.
            key = slug(r)
            if key in emb_dict:
                continue
            emb_dict[key] = torch.from_numpy(emb).float().unsqueeze(0)  # [1, dim]
            n_new += 1
            if suite == "train":
                train_repos.append(r)
        per_suite_repos[suite] = n_new
        print(f"  {suite:<8s}: {len(per_repo):>5d} repos in parquet, "
              f"{n_new:>5d} new -> running total {len(emb_dict):>5d}",
              flush=True)

    if not emb_dict:
        raise SystemExit("No embeddings extracted -- check --commits-dir.")

    # Atomic write.
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(emb_dict, tmp)
    os.replace(tmp, output)
    sample = next(iter(emb_dict.values()))
    print(f"\nSaved {len(emb_dict)} per-repo embeddings -> {output}")
    print(f"  Per-repo tensor shape: {tuple(sample.shape)} "
          f"(dtype={sample.dtype})")
    print(f"  Suite contribution: {per_suite_repos}")

    # Sibling train_repos.txt -- the v2 train repo list.
    train_repos_path = output.with_name(output.name + ".train_repos.txt")
    train_repos_path.write_text("\n".join(sorted(train_repos)) + "\n")
    print(f"  Wrote train-repo list ({len(train_repos)} repos) -> "
          f"{train_repos_path}")


if __name__ == "__main__":
    main()
