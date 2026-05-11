#!/usr/bin/env python3
"""Build ALL splits for the static-commit (Code2LoRA-direct) pipeline.

For every (repo, commit) listed in the static-commit manifest, this script:
  1. Reads each .py file as it exists at that commit via ``git show`` (no
     checkout, no on-disk perturbation).
  2. Detects test files via the same heuristics as ``2_separate_tests`` /
     ``create_qnas_v2``.
  3. Re-extracts QnA pairs with the v2 extractor and applies v2's balanced
     selection (max_per_repo / per_function / per_file).
  4. Looks up the corresponding snapshot embedding from
     ``snapshot_embeddings.json``.
  5. Annotates each entry with ``n_commits_after_anchor`` AND
     ``n_files_changed_since_anchor`` (one cheap ``git diff --shortstat`` per
     non-anchor commit).
  6. Writes JSON splits compatible with both:
       - ``hypernetwork/hypernetwork_sampled.py`` (consumes ``train.json``,
         ``cr_val.json``, ``cr_test.json`` from ``--splits-dir``)
       - the per-commit eval driver (consumes ``<suite>.json`` with
         ``<repo>@@<sha>`` keys).

Produced files in ``--out-dir``::

    train.json          400 train repos at their anchor commits      (role=train_snapshot)
    ir_val.json         400 train repos at every in_repo_split=val   commit
    ir_test.json        400 train repos at every in_repo_split=test  commit
    cr_val.json          49 cr_val  repos at every commit
    cr_test.json         51 cr_test repos at every commit
    ood_test.json        ~146 ood   repos at every commit            (if manifest has ood rows)
    SPLITS_README.json   provenance + stats

Each entry has the canonical shape ``hypernetwork_sampled`` consumes::

    {"embedding": [2048], "qna_pairs": [{prefix, target, ...}], "metadata": {...}}

Train.json keys = ``<repo_id>`` (one per repo).
All eval JSONs keys = ``<repo_id>@@<commit_sha>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse the v2 extractor + test-file heuristics verbatim.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from create_qnas_v2 import extract_from_file, select_balanced_pairs  # noqa: E402

_sep_tests = import_module("2_separate_tests")
contains_test_content = _sep_tests.contains_test_content
is_test_by_path = _sep_tests.is_test_by_path
SKIP_DIRS = _sep_tests.SKIP_DIRS


REPOS_ROOTS_DEFAULT = [
    "/scratch/lhotsko/REPO_DATASET/repositories",
    "/scratch/lhotsko/REPO_DATASET/repositories_ood",
    "/scratch/lhotsko/REPO_DATASET/repositories_ood_matched",
]


def _resolve_repo_path(repo_id: str, roots: List[Path]) -> Path | None:
    for root in roots:
        p = root / repo_id
        if p.is_dir() and (p / ".git").exists():
            return p
    return None


def _git(repo_path: Path, *args: str, binary: bool = False) -> bytes | str:
    res = subprocess.run(["git", *args], cwd=str(repo_path),
                         capture_output=True, check=False)
    if res.returncode != 0:
        return b"" if binary else ""
    return res.stdout if binary else res.stdout.decode("utf-8", errors="ignore")


def _list_py_files_at(repo_path: Path, sha: str) -> List[str]:
    out = _git(repo_path, "ls-tree", "-r", "--name-only", sha)
    files = []
    for ln in out.splitlines():
        if not ln.endswith(".py"):
            continue
        parts = Path(ln).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        files.append(ln)
    return files


def _read_file_at(repo_path: Path, sha: str, rel: str) -> str:
    raw = _git(repo_path, "show", f"{sha}:{rel}", binary=True)
    if not raw:
        return ""
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _detect_test_files_at(repo_path: Path, sha: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in _list_py_files_at(repo_path, sha):
        if is_test_by_path(rel):
            txt = _read_file_at(repo_path, sha, rel)
            if txt:
                out.append((rel, txt))
            continue
        txt = _read_file_at(repo_path, sha, rel)
        if not txt:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if contains_test_content(txt):
                out.append((rel, txt))
    return out


def _extract_qnas_at(
    repo_id: str, repo_path: Path, sha: str,
    max_per_repo: int, max_per_function: int, max_per_file: int,
) -> Tuple[List[dict], Dict[str, int]]:
    test_files = _detect_test_files_at(repo_path, sha)
    all_pairs: List[dict] = []
    for rel, source in test_files:
        try:
            pairs = extract_from_file(source, rel)
        except Exception:
            continue
        for p in pairs:
            md = p.setdefault("metadata", {})
            md["repo"] = repo_id
            md["test_file"] = rel
        all_pairs.extend(pairs)
    if not all_pairs:
        return [], {"n_test_files": len(test_files),
                    "n_extracted": 0, "n_selected": 0}
    selected = select_balanced_pairs(
        all_pairs, max_per_repo=max_per_repo,
        max_per_function=max_per_function, max_per_file=max_per_file,
    )
    return selected, {"n_test_files": len(test_files),
                      "n_extracted": len(all_pairs),
                      "n_selected": len(selected)}


def _files_changed(repo_path: Path, sha_a: str, sha_b: str) -> int:
    """Count .py files changed between sha_a and sha_b (anchor..commit)."""
    if sha_a == sha_b:
        return 0
    out = _git(repo_path, "diff", "--name-only", f"{sha_a}..{sha_b}")
    return sum(1 for ln in out.splitlines() if ln.endswith(".py"))


def _read_manifest(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["commit_index"] = int(row["commit_index"])
            row["anchor_index"] = int(row["anchor_index"])
            row["n_commits_after_anchor"] = int(row["n_commits_after_anchor"])
            rows.append(row)
    return rows


# Map manifest role -> output split file. (role -> split_name)
# A manifest row may contribute to MULTIPLE splits (e.g. cr_test_train
# also serves as a "decay" point alongside cr_test_test).
ROLE_TO_SPLIT = {
    "train_snapshot":  "train",
    # in-repo eval suites: commits of the 400 train repos at their val/test
    # cuts. Those aren't in the manifest under those role names yet; we
    # synthesize them from commits parquet below.
    "cr_test_train":   "cr_test_history",
    "cr_test_val":     "cr_test_history",
    "cr_test_test":    "cr_test_history",
    "ir_test":         "ir_test_history",
    "ood_test":        "ood_test_history",
}


def _build_ir_val_test_rows(
    parquet_dir: Path, train_anchors: Dict[str, Tuple[str, int]],
) -> List[dict]:
    """Pull every (repo, commit) with in_repo_split in {val, test} for the 400
    cross_repo=='train' repos from the commits parquet, so we can score the
    static model on the in-repo val/test commits even though they're not
    materialized in the manifest under a dedicated role."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    rows: List[dict] = []
    commits_path = parquet_dir / "commits" / "train.parquet"
    if not commits_path.exists():
        print(f"  [ir_val/ir_test] missing {commits_path}, skipping")
        return rows
    tbl = pq.read_table(commits_path, columns=[
        "repo_id", "commit_sha", "commit_index", "in_repo_split",
    ])
    # Drop train rows; keep val + test.
    mask = pc.or_(pc.equal(tbl.column("in_repo_split"), "val"),
                  pc.equal(tbl.column("in_repo_split"), "test"))
    tbl = tbl.filter(mask)
    df = tbl.to_pandas().drop_duplicates(["repo_id", "commit_sha"])
    for r in df.itertuples(index=False):
        anc = train_anchors.get(r.repo_id)
        if anc is None:
            continue
        anc_sha, anc_idx = anc
        rows.append({
            "repo_id": r.repo_id,
            "commit_sha": r.commit_sha,
            "commit_index": int(r.commit_index),
            "role": "ir_val" if r.in_repo_split == "val" else "ir_test",
            "anchor_sha": anc_sha,
            "anchor_index": anc_idx,
            "n_commits_after_anchor": int(r.commit_index) - anc_idx,
            "in_repo_split": r.in_repo_split,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/manifest.tsv")
    ap.add_argument("--snapshot-embeddings",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/snapshot_embeddings.json")
    ap.add_argument("--parquet-dir",
                    default="/scratch/lhotsko/REPO_DATASET/commit_parquet_hf",
                    help="Source of train repos' val/test commits parquet.")
    ap.add_argument("--repos-roots", nargs="+", default=REPOS_ROOTS_DEFAULT)
    ap.add_argument("--out-dir",
                    default="/scratch/lhotsko/REPO_DATASET/static_commit/splits_at_anchor")
    ap.add_argument("--max-per-repo", type=int, default=200)
    ap.add_argument("--max-per-function", type=int, default=5)
    ap.add_argument("--max-per-file", type=int, default=20)
    ap.add_argument("--skip-files-changed", action="store_true",
                    help="Don't compute n_files_changed_since_anchor (faster).")
    ap.add_argument("--limit-repos", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--only-suites", nargs="+", default=None,
                    help="Restrict to a subset of {train, ir_val, ir_test, "
                         "cr_val, cr_test, ood_test}.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in args.repos_roots]

    print(f"Reading manifest {args.manifest} ...")
    manifest_rows = _read_manifest(Path(args.manifest))
    train_anchors: Dict[str, Tuple[str, int]] = {}
    for row in manifest_rows:
        if row["role"] == "train_snapshot":
            train_anchors[row["repo_id"]] = (
                row["commit_sha"], row["commit_index"])
    print(f"  rows={len(manifest_rows):,}  train_anchors={len(train_anchors)}")

    # Map every manifest row into one of our output suites.
    rows_by_suite: Dict[str, List[dict]] = defaultdict(list)
    seen_pair: set = set()  # (repo, sha) — dedup across roles
    for row in manifest_rows:
        role = row["role"]
        if role == "train_snapshot":
            suite = "train"
        elif role.startswith("cr_test_"):
            suite = "cr_test"
        elif role == "ir_test":
            # role=ir_test in manifest = commits of cross_repo='train' repos
            # at their in_repo_split=='test' time. We name the OUTPUT split
            # "ir_test" too.
            suite = "ir_test"
        elif role == "ood_test":
            suite = "ood_test"
        else:
            continue
        key = (row["repo_id"], row["commit_sha"], suite)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        rows_by_suite[suite].append(row)

    # Synthesize ir_val (val commits of train repos) from the commits parquet.
    print("Pulling ir_val rows from commits parquet ...")
    ir_val_rows = [r for r in _build_ir_val_test_rows(
        Path(args.parquet_dir), train_anchors) if r["role"] == "ir_val"]
    rows_by_suite["ir_val"].extend(ir_val_rows)
    print(f"  ir_val rows pulled from parquet: {len(ir_val_rows)}")

    # Need a CR-val split for early-stop. Build it from manifest cr_val rows
    # if present; otherwise from parquet.
    if not rows_by_suite.get("cr_val"):
        import pyarrow.parquet as pq
        p = Path(args.parquet_dir) / "commits" / "cr_val.parquet"
        if p.exists():
            tbl = pq.read_table(p, columns=["repo_id", "commit_sha", "commit_index"])
            df = tbl.to_pandas().drop_duplicates(["repo_id", "commit_sha"])
            for r in df.itertuples(index=False):
                rows_by_suite["cr_val"].append({
                    "repo_id": r.repo_id, "commit_sha": r.commit_sha,
                    "commit_index": int(r.commit_index), "role": "cr_val",
                    "anchor_sha": "", "anchor_index": -1,
                    "n_commits_after_anchor": 0,
                    "in_repo_split": "",
                })

    if args.only_suites:
        rows_by_suite = {k: v for k, v in rows_by_suite.items()
                         if k in args.only_suites}

    for s, rs in rows_by_suite.items():
        print(f"  suite={s}: {len(rs):,} rows")

    print("Loading snapshot embeddings ...")
    snap_emb = json.loads(Path(args.snapshot_embeddings).read_text(encoding="utf-8"))
    print(f"  {len(snap_emb):,} embeddings")

    # Process every suite.
    summary: Dict[str, dict] = {}
    for suite, rows in rows_by_suite.items():
        if args.shard_total > 1:
            rows = [r for i, r in enumerate(rows)
                    if i % args.shard_total == args.shard_index]
        if args.limit_rows > 0:
            rows = rows[: args.limit_rows]

        repositories: Dict[str, dict] = {}
        agg = Counter()
        unique_repos = sorted({r["repo_id"] for r in rows})
        if args.limit_repos > 0:
            unique_repos = unique_repos[: args.limit_repos]
            rows = [r for r in rows if r["repo_id"] in set(unique_repos)]

        print(f"\n[{suite}] processing {len(rows):,} rows "
              f"across {len(unique_repos)} repos "
              f"(shard {args.shard_index+1}/{args.shard_total}) ...", flush=True)

        # Cache resolved repo paths.
        repo_path_cache: Dict[str, Path | None] = {}

        for i, r in enumerate(rows, 1):
            repo_id = r["repo_id"]
            sha = r["commit_sha"]
            ci = r["commit_index"]

            repo_path = repo_path_cache.get(repo_id)
            if repo_path is None:
                repo_path = _resolve_repo_path(repo_id, roots)
                repo_path_cache[repo_id] = repo_path
            if repo_path is None:
                agg["no_clone"] += 1
                continue

            emb = snap_emb.get(f"{repo_id}@{sha}")
            if emb is None:
                agg["no_emb"] += 1
                continue

            pairs, stats = _extract_qnas_at(
                repo_id, repo_path, sha,
                max_per_repo=args.max_per_repo,
                max_per_function=args.max_per_function,
                max_per_file=args.max_per_file,
            )
            agg["test_files_total"] += stats["n_test_files"]
            agg["extracted_total"] += stats["n_extracted"]
            agg["selected_total"] += stats["n_selected"]
            if not pairs:
                agg["no_qna"] += 1
                continue

            # Distance metadata.
            anc_sha = r.get("anchor_sha") or ""
            anc_idx = r.get("anchor_index", -1)
            n_after = r.get("n_commits_after_anchor", 0)
            if anc_sha and not args.skip_files_changed and anc_sha != sha:
                n_files = _files_changed(repo_path, anc_sha, sha)
            else:
                n_files = 0

            key = repo_id if suite == "train" else f"{repo_id}@@{sha}"
            repositories[key] = {
                "embedding": emb,
                "qna_pairs": pairs,
                "metadata": {
                    "repo_id": repo_id,
                    "commit_sha": sha,
                    "commit_index": ci,
                    "anchor_sha": anc_sha,
                    "anchor_index": anc_idx,
                    "n_commits_after_anchor": n_after,
                    "n_files_changed_since_anchor": n_files,
                    "in_repo_split": r.get("in_repo_split", ""),
                    "role": r["role"],
                    "qna_source": "v2_extractor_at_commit",
                    "n_test_files": stats["n_test_files"],
                    "n_extracted_before_select": stats["n_extracted"],
                },
            }
            agg["kept"] += 1

            if i % 100 == 0 or i == len(rows):
                print(f"  [{suite}] [{i}/{len(rows)}] kept={agg['kept']} "
                      f"qna={agg['selected_total']:,}", flush=True)

        # Write.
        if args.shard_total == 1:
            fname = f"{suite}.json"
        else:
            fname = f"{suite}.shard{args.shard_index}.json"
        out_path = out_dir / fname
        out_path.write_text(json.dumps({"repositories": repositories}),
                            encoding="utf-8")
        summary[suite] = {
            "n_rows_in": len(rows),
            "n_entries_out": len(repositories),
            "stats": dict(agg),
            "path": str(out_path),
        }
        print(f"  -> {out_path} ({len(repositories):,} entries)")

    readme = {
        "manifest": args.manifest,
        "snapshot_embeddings": args.snapshot_embeddings,
        "parquet_dir": args.parquet_dir,
        "repos_roots": [str(p) for p in roots],
        "caps": {
            "max_per_repo": args.max_per_repo,
            "max_per_function": args.max_per_function,
            "max_per_file": args.max_per_file,
        },
        "summary": summary,
        "shard": [args.shard_index, args.shard_total],
    }
    rname = "SPLITS_README.json" if args.shard_total == 1 else f"SPLITS_README.shard{args.shard_index}.json"
    (out_dir / rname).write_text(json.dumps(readme, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for s, info in summary.items():
        print(f"  {s}: {info['n_entries_out']:,} entries / {info['n_rows_in']:,} rows  "
              f"qna_kept={info['stats'].get('selected_total', 0):,}")


if __name__ == "__main__":
    main()
