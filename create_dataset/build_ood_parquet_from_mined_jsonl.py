#!/usr/bin/env python3
"""
Clone repos listed in ``mine_ood_repos.py`` JSONL output and build commit-level
Parquet (``commits.parquet`` + ``qna_pairs.parquet``) using the same pipeline as
the training corpus: ``build_commit_parquet_db.py``.

Layout:

  * Clones go under ``--repos-root/<owner>/<repo>/``. Default:
    ``<work-dir>/repositories_ood`` so OOD trees stay separate from the main
    ``repositories/`` tree. Override with e.g.
    ``--repos-root \$SCRATCH/REPO_DATASET/repositories_ood``.
  * Writes ``--splits-dir/ood_test.json`` with stub ``repositories`` entries so
    ``cross_repo_split`` in Parquet is ``ood_test`` (see ``CROSS_REPO_SPLIT_FILES``
    in ``build_commit_parquet_db.py``).
  * Parquet output: ``--out-dir/commits.parquet`` and ``qna_pairs.parquet``.

**Git history:** the parquet builder walks the full first-parent history to find
commits that introduce new assertions. Use ``--git-depth 0`` (default) for a
full clone; shallow clones often yield empty or tiny Parquet.

Example::

    python create_dataset/build_ood_parquet_from_mined_jsonl.py \\
        --mined-jsonl  $SCRATCH/REPO_DATASET/ood_repos_100.jsonl \\
        --repos-root   $SCRATCH/REPO_DATASET/repositories_ood \\
        --work-dir     $SCRATCH/REPO_DATASET/ood_bundle \\
        --out-dir      $SCRATCH/REPO_DATASET/commit_parquet_ood \\
        --clone-jobs 8 --parquet-workers 8

Eval loaders can point at ``--out-dir`` and filter ``cross_repo_split == 'ood_test'``
(or pass ``ood_test`` in ``--cross_repo_eval_splits`` where supported).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _clone_url(row: Dict[str, Any]) -> str:
    u = (row.get("html_url") or "").strip()
    if u.endswith(".git"):
        return u
    if "github.com" in u and u:
        return u + ".git" if not u.endswith("/") else u[:-1] + ".git"
    fn = row.get("full_name", "").strip()
    if "/" in fn:
        return f"https://github.com/{fn}.git"
    raise ValueError(f"Cannot derive clone URL from row: {row!r}")


def _write_ood_split(splits_dir: Path, repo_ids: List[str]) -> Path:
    splits_dir.mkdir(parents=True, exist_ok=True)
    # Keys only; pair lists unused by the parquet builder (assignment map).
    data = {
        "repositories": {rid: {"pairs": []} for rid in repo_ids},
    }
    out = splits_dir / "ood_test.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _write_repo_list(path: Path, repo_ids: List[str]) -> None:
    path.write_text("\n".join(repo_ids) + "\n", encoding="utf-8")


def _clone_one(
    *,
    full_name: str,
    clone_url: str,
    dest: Path,
    git_depth: int,
    timeout: int,
) -> Tuple[str, str, str]:
    """Return (full_name, status, message). status ok|skip|error."""
    if (dest / ".git").is_dir():
        return full_name, "skip", "already cloned"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if git_depth > 0:
        cmd.extend(["--depth", str(git_depth)])
    cmd.extend([clone_url, str(dest)])
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return full_name, "error", f"timeout after {timeout}s"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        return full_name, "error", tail.strip()
    return full_name, "ok", ""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    builder = Path(__file__).resolve().parent / "build_commit_parquet_db.py"
    if not builder.is_file():
        print(f"Missing {builder}", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mined-jsonl", type=Path, required=True, help="Output from mine_ood_repos.py (JSON lines).")
    ap.add_argument(
        "--repos-root",
        type=Path,
        default=None,
        help="Clone destination root (<owner>/<repo>). Default: <work-dir>/repositories_ood",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Default scratch layout: splits + repo_list live here. Default: parent of --out-dir / ood_work",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Parquet output directory (commits.parquet + qna_pairs.parquet).",
    )
    ap.add_argument("--clone-jobs", type=int, default=4, help="Parallel git clones.")
    ap.add_argument("--parquet-workers", type=int, default=4, help="Workers for build_commit_parquet_db.py.")
    ap.add_argument(
        "--git-depth",
        type=int,
        default=0,
        help="git clone --depth N; 0 = full history (recommended for commit/QnA mining).",
    )
    ap.add_argument("--clone-timeout", type=int, default=3600, help="Per-repo clone timeout (seconds).")
    ap.add_argument("--diff-timeout", type=int, default=600, help="Forwarded to build_commit_parquet_db.py.")
    ap.add_argument("--skip-clone", action="store_true", help="Assume repos already exist under --repos-root.")
    ap.add_argument("--skip-parquet", action="store_true", help="Only clone + write split stubs.")
    ap.add_argument("--resume-parquet", action="store_true", help="Pass --resume to build_commit_parquet_db.py.")
    ap.add_argument("--no-concat", action="store_true", help="Forward --no-concat (shards only).")
    ap.add_argument("--limit-repos", type=int, default=None, help="Process only first N repos from the JSONL.")
    args = ap.parse_args()

    mined = args.mined_jsonl.expanduser().resolve()
    if not mined.is_file():
        print(f"Not found: {mined}", file=sys.stderr)
        return 2

    out_dir = args.out_dir.expanduser().resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else out_dir.parent / f"{out_dir.name}_work"
    )
    repos_root = (
        args.repos_root.expanduser().resolve()
        if args.repos_root
        else work_dir / "repositories_ood"
    )
    splits_dir = work_dir / "splits"
    repo_list_path = work_dir / "ood_repo_ids.txt"

    rows = _iter_jsonl(mined)
    if args.limit_repos is not None:
        rows = rows[: args.limit_repos]

    triples: List[Tuple[str, str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        fn = (row.get("full_name") or "").strip()
        if not fn or "/" not in fn:
            continue
        key = fn.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            url = _clone_url(row)
        except ValueError as e:
            print(f"skip row: {e}", file=sys.stderr)
            continue
        owner, _, name = fn.partition("/")
        triples.append((fn, url, repos_root / owner / name))

    if not triples:
        print("No valid repos in JSONL.", file=sys.stderr)
        return 2

    repo_ids = [t[0] for t in triples]
    _write_ood_split(splits_dir, repo_ids)
    _write_repo_list(repo_list_path, repo_ids)
    print(f"Wrote {splits_dir / 'ood_test.json'} ({len(repo_ids)} repos)")
    print(f"Wrote {repo_list_path}")

    if not args.skip_clone:
        print(f"Cloning {len(triples)} repos into {repos_root} (jobs={args.clone_jobs}, depth={args.git_depth or 'full'})...")
        err = 0
        with ThreadPoolExecutor(max_workers=max(1, args.clone_jobs)) as ex:
            futs = {
                ex.submit(
                    _clone_one,
                    full_name=fn,
                    clone_url=url,
                    dest=dest,
                    git_depth=args.git_depth,
                    timeout=args.clone_timeout,
                ): fn
                for fn, url, dest in triples
            }
            for fut in as_completed(futs):
                fn, status, msg = fut.result()
                if status == "ok":
                    print(f"  clone ok: {fn}")
                elif status == "skip":
                    print(f"  skip: {fn}")
                else:
                    err += 1
                    print(f"  ERROR {fn}: {msg[:400]}", file=sys.stderr)
        if err:
            print(f"{err} clone error(s); fix or remove bad rows before parquet.", file=sys.stderr)
    else:
        print("Skipping clone (--skip-clone).")

    if args.skip_parquet:
        print("Skipping parquet (--skip-parquet).")
        return 0

    cmd = [
        sys.executable,
        str(builder),
        "--splits-dir",
        str(splits_dir),
        "--repos-root",
        str(repos_root),
        "--out-dir",
        str(out_dir),
        "--repo-list",
        str(repo_list_path),
        "--workers",
        str(max(1, args.parquet_workers)),
        "--diff-timeout",
        str(args.diff_timeout),
    ]
    if args.resume_parquet:
        cmd.append("--resume")
    if args.no_concat:
        cmd.append("--no-concat")

    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = f"{root}:{pp}" if pp else str(root)
    # Ensure local imports in build_commit_parquet_db resolve (create_dataset as cwd).
    r = subprocess.run(cmd, cwd=str(builder.parent), env=env)
    return int(r.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
