#!/usr/bin/env python3
"""Build per-``(repo_id, commit_sha)`` dependency-resolved context (DRC)
caches for the v2 commit-derived QnA dataset.

For every commit touched by a QnA we materialize the repository tree at
that commit via ``git archive | tar -x`` into a tmpdir, run the v2
function-scoped DRC pipeline once per QnA, and write a single JSON keyed
by ``(test_file, lineno, col_offset)``. The tmpdir is removed
immediately after the commit's QnAs are processed.

This is the per-commit analogue of
``baselines/oracle_context/build_context_v2.py``, which only walks the
HEAD working tree at a single repo snapshot.

Resumable, shardable, and safe to rerun -- each per-commit JSON is
written atomically via ``tmp + rename``.

Usage::

    python evaluation/build_drc_cache_per_commit.py \
        --qna-dir /scratch/lhotsko/REPO_DATASET/commit_parquet_hf/qna \
        --suites cr_val cr_test ir_val ir_test \
        --repos-root /scratch/lhotsko/REPO_DATASET/repositories \
        --out-dir /scratch/lhotsko/ORACLE_CONTEXT_CACHE_COMMITS \
        --num-shards 4 --shard-i 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the existing v2 DRC primitives. They were written against an
# arbitrary ``repo_dir`` argument so we can point them at our tmpdir.
from baselines.oracle_context.build_context_v2 import (  # noqa: E402
    _build_repo_file_index,
    _find_source_roots,
    build_context_for_pair,
)


# ---------------------------------------------------------------------------
# Suite enumeration -- mirrors evaluation/build_rag_cache_per_commit.py.
# ---------------------------------------------------------------------------

_SUITE_FALLBACK_IN_SPLIT = {"ir_test": "test", "ir_val": "val"}


def _list_suite_repos(qna_dir: Path, suites: List[str]) -> List[str]:
    """Return the sorted union of ``repo_id`` across all requested suites.

    Reads only the ``repo_id`` column (plus ``in_repo_split`` when falling
    back to ``train.parquet``), so memory stays small even across all four
    suites.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    repos: Set[str] = set()
    for s in suites:
        direct = qna_dir / f"{s}.parquet"
        if direct.exists():
            tbl = pq.read_table(str(direct), columns=["repo_id"])
        elif s in _SUITE_FALLBACK_IN_SPLIT:
            train = qna_dir / "train.parquet"
            if not train.exists():
                print(f"[enumerate] WARNING: missing {direct} (and "
                      f"{train} fallback)", flush=True)
                continue
            tbl = pq.read_table(str(train), columns=["repo_id", "in_repo_split"])
            tbl = tbl.filter(pc.equal(tbl["in_repo_split"],
                                      _SUITE_FALLBACK_IN_SPLIT[s]))
        else:
            print(f"[enumerate] WARNING: missing {direct}", flush=True)
            continue
        for r in tbl["repo_id"].to_pylist():
            if r:
                repos.add(r)
        print(f"[enumerate] suite={s:8s} +{tbl.num_rows:,} qnas; "
              f"running unique-repo total: {len(repos):,}", flush=True)
    return sorted(repos)


_QNA_COLS = [
    "repo_id", "commit_sha", "commit_index", "in_repo_split", "test_file",
    "test_function", "prefix", "lineno", "col_offset", "assertion_event_id",
]


def _columns_to_read(parquet_path: Path) -> List[str]:
    """Return the subset of ``_QNA_COLS`` actually present in the parquet --
    keeps the loader portable across in-distribution (has assertion_event_id)
    and OOD (no assertion_event_id) qna banks."""
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(str(parquet_path)).schema.names)
    return [c for c in _QNA_COLS if c in schema_names]


def _load_repo_qnas(
    qna_dir: Path, suites: List[str], repo_id: str,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Stream the QnAs for a single repo across all requested suites,
    keyed by ``(repo_id, commit_sha)``. Memory bounded by one repo's
    worth of QnAs (a few thousand rows at most)."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in suites:
        direct = qna_dir / f"{s}.parquet"
        if direct.exists():
            cols = _columns_to_read(direct)
            tbl = pq.read_table(
                str(direct), columns=cols,
                filters=[("repo_id", "=", repo_id)],
            )
        elif s in _SUITE_FALLBACK_IN_SPLIT:
            train = qna_dir / "train.parquet"
            if not train.exists():
                continue
            cols = _columns_to_read(train)
            tbl = pq.read_table(
                str(train), columns=cols,
                filters=[
                    ("repo_id", "=", repo_id),
                    ("in_repo_split", "=", _SUITE_FALLBACK_IN_SPLIT[s]),
                ],
            )
        else:
            continue
        if tbl.num_rows == 0:
            continue
        present = set(tbl.column_names)
        shas = tbl["commit_sha"].to_pylist()
        tfs = tbl["test_file"].to_pylist()
        prefixes = tbl["prefix"].to_pylist()
        lines = tbl["lineno"].to_pylist()
        cols_v = tbl["col_offset"].to_pylist()
        eids = (tbl["assertion_event_id"].to_pylist()
                if "assertion_event_id" in present
                else [""] * tbl.num_rows)
        tfuncs = tbl["test_function"].to_pylist()
        for i in range(tbl.num_rows):
            sh = shas[i]
            if not sh:
                continue
            by_key[(repo_id, sh)].append({
                "test_file": tfs[i] or "",
                "test_function": tfuncs[i] or "",
                "prefix": prefixes[i] or "",
                "lineno": int(lines[i]) if lines[i] is not None else 0,
                "col_offset": int(cols_v[i]) if cols_v[i] is not None else 0,
                "assertion_event_id": eids[i] or "",
                "suite": s,
            })
    return by_key


# ---------------------------------------------------------------------------
# Git archive into a tmp dir.
# ---------------------------------------------------------------------------

def _git_archive_to_tmpdir(repo_dir: Path, sha: str, dst: Path) -> bool:
    """Materialize the tree at ``<sha>`` into ``dst``. Returns True on success."""
    dst.mkdir(parents=True, exist_ok=True)
    # Two-process pipe: git archive | tar -x
    try:
        ga = subprocess.Popen(
            ["git", "-C", str(repo_dir), "archive", "--format=tar", sha],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        # Use -C to extract into dst.
        tar = subprocess.Popen(
            ["tar", "-x", "-C", str(dst)],
            stdin=ga.stdout, stderr=subprocess.DEVNULL,
        )
        ga.stdout.close()
        tar.wait(timeout=300)
        ga.wait(timeout=300)
        if ga.returncode != 0 or tar.returncode != 0:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# Per-commit DRC pass
# ---------------------------------------------------------------------------

def _process_one_commit(
    *,
    repo_dir: Path,
    repo_id: str,
    sha: str,
    qnas: List[Dict[str, Any]],
    tmp_root: Path,
) -> Optional[Dict[str, Any]]:
    """Materialize ``<repo>@<sha>`` and run DRC v2 on each QnA."""
    with tempfile.TemporaryDirectory(dir=str(tmp_root), prefix="drc_") as td:
        td_path = Path(td)
        ok = _git_archive_to_tmpdir(repo_dir, sha, td_path)
        if not ok:
            return None
        try:
            file_index = _build_repo_file_index(td_path)
            source_roots = _find_source_roots(td_path)
        except Exception as e:
            return {"_error": f"file_index: {type(e).__name__}: {e}"}

        test_file_cache: Dict[str, str] = {}
        contexts: Dict[str, Dict[str, Any]] = {}
        n_with_ctx = 0
        n_with_func_scope = 0
        total_chars = 0
        for q in qnas:
            # Read the test file from the tree at this commit.
            tf_rel = q["test_file"]
            if tf_rel and tf_rel not in test_file_cache:
                test_path = td_path / tf_rel
                if test_path.exists() and test_path.is_file():
                    try:
                        test_file_cache[tf_rel] = test_path.read_text(
                            encoding="utf-8", errors="ignore")
                    except Exception:
                        test_file_cache[tf_rel] = ""
                else:
                    test_file_cache[tf_rel] = ""
            metadata = {"file": tf_rel, "lineno": q["lineno"]}
            try:
                ctx = build_context_for_pair(
                    q["prefix"], metadata, td_path, file_index,
                    test_file_cache, source_roots=source_roots,
                )
            except Exception as e:
                ctx = {
                    "resolved_imports": [],
                    "used_names": [],
                    "enclosing_function": None,
                    "extracted_code": "",
                    "n_imports_parsed": 0,
                    "n_files_resolved": 0,
                    "n_chars_extracted": 0,
                    "_error": f"{type(e).__name__}: {e}",
                }
            if ctx.get("n_chars_extracted", 0) > 0:
                n_with_ctx += 1
                total_chars += int(ctx["n_chars_extracted"])
            if ctx.get("enclosing_function"):
                n_with_func_scope += 1
            key = q.get("assertion_event_id") or f"{tf_rel}::{q['lineno']}::{q['col_offset']}"
            contexts[key] = ctx

        return {
            "repo_id": repo_id,
            "commit_sha": sha,
            "n_qnas": len(qnas),
            "n_qnas_with_context": n_with_ctx,
            "n_qnas_with_function_scope": n_with_func_scope,
            "total_chars_extracted": total_chars,
            "contexts": contexts,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--qna-dir",
        default=f"{scratch}/REPO_DATASET/code2lora_snapshots_hf/qna",
        help="Directory with v2 commit-derived qna parquets (standalone "
             "<suite>.parquet preferred; train.parquet+in_repo_split is "
             "used as a fallback for ir_*).",
    )
    ap.add_argument(
        "--suites", nargs="+",
        default=["cr_val", "cr_test", "ir_val", "ir_test"],
        choices=["train", "cr_val", "cr_test", "ir_val", "ir_test", "ood_test"],
    )
    ap.add_argument(
        "--repos-root",
        default=f"{scratch}/REPO_DATASET/repositories",
    )
    ap.add_argument(
        "--out-dir",
        default=f"{scratch}/ORACLE_CONTEXT_CACHE_COMMITS",
    )
    ap.add_argument(
        "--tmp-root",
        default=os.environ.get("SLURM_TMPDIR", "/tmp"),
        help="Where to materialize per-commit working trees. SLURM_TMPDIR "
             "is preferred on cluster nodes.",
    )
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only process the first N (repo, sha) keys.")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if the cache file already exists.")
    args = ap.parse_args()

    qna_dir = Path(args.qna_dir).expanduser().resolve()
    repos_root = Path(args.repos_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tmp_root = Path(args.tmp_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[args] qna_dir   = {qna_dir}")
    print(f"[args] repos_root = {repos_root}")
    print(f"[args] out_dir   = {out_dir}")
    print(f"[args] tmp_root  = {tmp_root}")
    print(f"[args] suites    = {args.suites}")
    print(f"[args] shard     = {args.shard_i + 1}/{args.num_shards}", flush=True)

    repos = _list_suite_repos(qna_dir, args.suites)
    print(f"\n[enumerate] total unique repos across suites: {len(repos):,}",
          flush=True)
    if args.num_shards > 1:
        repos = [r for i, r in enumerate(repos) if i % args.num_shards == args.shard_i]
        print(f"[enumerate] after shard {args.shard_i + 1}/{args.num_shards}: "
              f"{len(repos):,} repos", flush=True)
    if args.limit:
        repos = repos[: args.limit]
        print(f"[enumerate] --limit -> {len(repos)} repos", flush=True)

    t0 = time.time()
    n_repos_done = 0
    n_commits_done = 0
    n_commits_failed = 0
    n_commits_skipped_cached = 0
    total_qnas_done = 0
    total_qnas_with_ctx = 0
    for repo_id in repos:
        if "/" not in repo_id:
            continue
        repo_dir = repos_root / repo_id
        if not repo_dir.exists():
            print(f"  [skip] {repo_id}: clone missing under {repos_root}",
                  flush=True)
            continue

        repo_qnas = _load_repo_qnas(qna_dir, args.suites, repo_id)
        if not repo_qnas:
            n_repos_done += 1
            continue
        commit_keys = sorted(repo_qnas.keys(), key=lambda k: k[1])
        for r, sh in commit_keys:
            safe = r.replace("/", "__")
            out_path = out_dir / f"{safe}__{sh}.json"
            if out_path.exists() and not args.force:
                n_commits_skipped_cached += 1
                continue
            try:
                payload = _process_one_commit(
                    repo_dir=repo_dir, repo_id=r, sha=sh,
                    qnas=repo_qnas[(r, sh)], tmp_root=tmp_root,
                )
            except Exception as e:
                print(f"  [error] {r} @ {sh[:8]}: {type(e).__name__}: {e}",
                      flush=True)
                payload = None
            n_commits_done += 1
            if payload is None:
                n_commits_failed += 1
                # Sentinel so we don't retry on every rerun.
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp.write_text(json.dumps({
                    "repo_id": r, "commit_sha": sh, "contexts": {},
                    "_error": "git_archive_failed",
                }))
                os.replace(tmp, out_path)
                continue
            total_qnas_done += int(payload.get("n_qnas", 0))
            total_qnas_with_ctx += int(payload.get("n_qnas_with_context", 0))
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, out_path)
        n_repos_done += 1
        elapsed = (time.time() - t0) / 60.0
        rate = n_repos_done / max(elapsed, 1e-6)
        eta = (len(repos) - n_repos_done) / max(rate, 1e-6)
        coverage = (100.0 * total_qnas_with_ctx / max(total_qnas_done, 1))
        print(f"  [{n_repos_done}/{len(repos)} repos] last={repo_id} "
              f"commits_built={n_commits_done} cached={n_commits_skipped_cached} "
              f"failed={n_commits_failed} "
              f"coverage_so_far={coverage:.1f}% "
              f"elapsed={elapsed:.1f}m ETA={eta:.1f}m", flush=True)

    print(f"\nDone. Repos: {n_repos_done}, commits built: {n_commits_done}, "
          f"already-cached: {n_commits_skipped_cached}, failed: {n_commits_failed}. "
          f"Cache dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
