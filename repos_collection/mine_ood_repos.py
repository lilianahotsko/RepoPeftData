#!/usr/bin/env python3
"""
Mine out-of-distribution GitHub repos for evaluation: same coarse filters as the
main dataset (Python, non-fork, star/size/license/recency bounds, pytest signal in
dependency files), but restrict *repository creation* to after a cutoff derived
from the training corpus (or set explicitly).

Also verifies:
  - no full_name overlap with baseline repos;
  - candidates are not GitHub forks whose parent/root chain touches a baseline repo.

Usage examples
----------------
  # 1) Infer latest baseline created_at from the API (cached), then mine 100 repos
  GITHUB_TOKEN=ghp_xxx python repos_collection/mine_ood_repos.py mine \\
      --baseline-json src/gru_train.json --baseline-json src/gru_cr_val.json \\
      --target 100 --out ~/scratch/ood_repos_100.jsonl

  # 2) Fixed temporal cutoff (repos first seen on GitHub strictly after this date)
  GITHUB_TOKEN=ghp_xxx python repos_collection/mine_ood_repos.py mine \\
      --created-after 2025-04-01 \\
      --baseline-json /path/to/train.json --baseline-json /path/to/cr_test.json \\
      --target 100 --out ~/scratch/ood_repos.jsonl

  # 3) Verify an existing JSONL
  GITHUB_TOKEN=ghp_xxx python repos_collection/mine_ood_repos.py verify \\
      --baseline-json src/gru_train.json \\
      --repos-jsonl ~/scratch/ood_repos_100.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import requests

GITHUB_API = "https://api.github.com"
API_VERSION = "2022-11-28"

# Paper / README-aligned defaults (override via CLI).
DEFAULT_STAR_SLICES = ("6..20", "21..50", "51..100", "101..300", "301..700", "701..2000")
DEFAULT_SIZE_RANGE = "100..204800"  # GitHub size is KB; ~100 KiB .. ~200 MiB
DEFAULT_PUSHED_AFTER = "2025-01-01"  # "recent activity" / README 2025+

PYTEST_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "Pipfile",
)

LICENSE_QUERIES = (
    "license:mit",
    "license:apache-2.0",
)


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def safe_get(
    token: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 4,
) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_headers(token), params=params, timeout=30)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                print(f"  [rate limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 + attempt)
                continue
            return r
        except requests.RequestException as e:
            print(f"  [error] {e}, retry {attempt + 1}", flush=True)
            time.sleep(2**attempt)
    return None


def parse_iso_dt(s: str) -> datetime:
    # GitHub: 2011-01-26T19:06:43Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def load_baseline_ids(paths: List[Path]) -> Set[str]:
    ids: Set[str] = set()
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Baseline file not found: {p}")
        if p.suffix.lower() == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        fn = item.get("full_name") or item.get("repo")
                        if fn:
                            ids.add(_norm_repo(str(fn)))
            continue
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and "repositories" in raw:
            for k in raw["repositories"].keys():
                ids.add(_norm_repo(k))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    fn = item.get("full_name") or item.get("repo")
                    if fn:
                        ids.add(_norm_repo(str(fn)))
        else:
            raise ValueError(f"Unsupported JSON shape in {p}")
    ids.discard("")
    return ids


def _norm_repo(name: str) -> str:
    return name.strip().lower()


def display_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return name


def fetch_repo(token: str, full_name: str) -> Optional[Dict[str, Any]]:
    owner, _, repo = full_name.partition("/")
    if not repo:
        return None
    r = safe_get(token, f"{GITHUB_API}/repos/{owner}/{repo}")
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def infer_created_cutoff_date(token: str, baseline_ids: Set[str], cache_path: Optional[Path]) -> date:
    """Return the calendar date D such that OOD repos should satisfy created > D (strict)."""
    cached: Dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}

    max_dt: Optional[datetime] = None
    for i, rid in enumerate(sorted(baseline_ids)):
        key = display_name(rid)
        created_raw: Optional[str] = None
        if key in cached and isinstance(cached[key], dict):
            created_raw = cached[key].get("created_at")
        if not created_raw:
            data = fetch_repo(token, key)
            if not data:
                print(f"  [warn] could not fetch baseline repo {key}", flush=True)
                time.sleep(0.2)
                continue
            created_raw = data.get("created_at")
            cached[key] = {"created_at": created_raw, "pushed_at": data.get("pushed_at")}
            time.sleep(0.15)
        if created_raw:
            dt = parse_iso_dt(created_raw)
            if max_dt is None or dt > max_dt:
                max_dt = dt
        if (i + 1) % 50 == 0:
            print(f"  [infer] processed {i + 1}/{len(baseline_ids)} baseline repos ...", flush=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))

    if max_dt is None:
        raise RuntimeError("Could not infer cutoff: no created_at values collected.")

    return max_dt.astimezone(timezone.utc).date()


def fork_chain_hits_baseline(repo_json: Dict[str, Any], baseline: Set[str]) -> bool:
    cur: Optional[Dict[str, Any]] = repo_json
    seen: Set[str] = set()
    while cur and cur.get("fork"):
        parent = cur.get("parent")
        if not isinstance(parent, dict):
            break
        pfn = parent.get("full_name")
        if not pfn:
            break
        n = _norm_repo(pfn)
        if n in seen:
            break
        seen.add(n)
        if n in baseline:
            return True
        cur = parent
    return False


def check_pytest_in_deps(token: str, full_name: str) -> bool:
    owner, _, repo = full_name.partition("/")
    if not repo:
        return False
    for filename in PYTEST_FILES:
        r = safe_get(token, f"{GITHUB_API}/repos/{owner}/{repo}/contents/{filename}")
        if r is None or r.status_code != 200:
            continue
        try:
            body = r.json()
            b64 = body.get("content", "")
            content = base64.b64decode(b64).decode("utf-8", errors="ignore")
            if "pytest" in content.lower():
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


def search_repo_page(
    token: str,
    q: str,
    *,
    page: int,
    per_page: int = 100,
) -> List[Dict[str, Any]]:
    r = safe_get(
        token,
        f"{GITHUB_API}/search/repositories",
        params={"q": q, "sort": "created", "order": "desc", "per_page": per_page, "page": page},
    )
    if r is None or r.status_code != 200:
        if r is not None and r.status_code == 422:
            print(f"  [422] query: {q}", flush=True)
        return []
    data = r.json()
    return list(data.get("items") or [])


def extract_row(item: Dict[str, Any]) -> Dict[str, Any]:
    lic = item.get("license") or {}
    return {
        "full_name": item["full_name"],
        "html_url": item.get("html_url", ""),
        "description": item.get("description") or "",
        "stars": item.get("stargazers_count", 0),
        "forks_count": item.get("forks_count", 0),
        "size_kb": item.get("size", 0),
        "language": item.get("language") or "",
        "license": lic.get("spdx_id") or "",
        "created_at": item.get("created_at"),
        "pushed_at": item.get("pushed_at"),
        "updated_at": item.get("updated_at"),
        "default_branch": item.get("default_branch") or "main",
        "archived": bool(item.get("archived")),
        "is_fork": bool(item.get("fork")),
        "topics": item.get("topics") or [],
    }


def mine_cmd(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("Set GITHUB_TOKEN", file=sys.stderr)
        return 2

    baseline_paths = [Path(p).expanduser().resolve() for p in args.baseline_json]
    baseline = load_baseline_ids(baseline_paths)
    print(f"Loaded {len(baseline)} baseline repo ids from {len(baseline_paths)} file(s).")

    if args.created_after:
        cutoff_s = args.created_after.strip()
        try:
            cutoff_date = date.fromisoformat(cutoff_s)
        except ValueError:
            print("--created-after must be YYYY-MM-DD", file=sys.stderr)
            return 2
        print(f"Using explicit created-after cutoff date: {cutoff_date} (search: created:>{cutoff_date})")
    else:
        cache = Path(args.infer_cache).expanduser() if args.infer_cache else None
        cutoff_date = infer_created_cutoff_date(token, baseline, cache)
        print(f"Inferred max baseline created_at date: {cutoff_date} (search: created:>{cutoff_date})")

    created_qual = f"created:>{cutoff_date.isoformat()}"
    pushed_qual = f"pushed:>={args.pushed_after}"
    archived_q = "archived:false"

    accepted: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.append_out:
        print(f"Refusing to overwrite {out_path} (use --append-out)", file=sys.stderr)
        return 2

    pre_existing_count = 0
    if args.append_out and out_path.exists():
        for row in _iter_jsonl(out_path):
            fn = row.get("full_name", "")
            if fn:
                seen.add(_norm_repo(fn))
                accepted.append(row)
        pre_existing_count = len(accepted)
        print(f"--append-out: loaded {pre_existing_count} existing rows from {out_path}")

    star_slices: Tuple[str, ...] = tuple(args.star_slice.split(",")) if args.star_slice else DEFAULT_STAR_SLICES

    for lic_q in LICENSE_QUERIES:
        if len(accepted) >= args.target:
            break
        for stars in star_slices:
            if len(accepted) >= args.target:
                break
            base = (
                f"language:python fork:false {archived_q} {lic_q} "
                f"stars:{stars} size:{args.size_range} {pushed_qual} {created_qual}"
            )
            print(f"\n[search] {base}", flush=True)
            for page in range(1, args.max_pages + 1):
                if len(accepted) >= args.target:
                    break
                items = search_repo_page(token, base, page=page)
                if not items:
                    break
                for item in items:
                    if len(accepted) >= args.target:
                        break
                    fn = item.get("full_name") or ""
                    nk = _norm_repo(fn)
                    if not nk or nk in seen:
                        continue
                    if nk in baseline:
                        continue
                    if item.get("fork") or item.get("archived"):
                        continue

                    detail = fetch_repo(token, fn)
                    time.sleep(0.2)
                    if not detail:
                        continue
                    if detail.get("fork") or detail.get("archived"):
                        continue
                    if fork_chain_hits_baseline(detail, baseline):
                        print(f"  skip (fork-of-baseline chain): {fn}", flush=True)
                        continue

                    created = detail.get("created_at")
                    if created:
                        cdt = parse_iso_dt(created).astimezone(timezone.utc).date()
                        if cdt <= cutoff_date:
                            continue

                    if not check_pytest_in_deps(token, fn):
                        continue

                    row = extract_row(detail)
                    row["uses_pytest"] = True
                    row["ood_cutoff_date"] = cutoff_date.isoformat()
                    row["search_query"] = base
                    row["mined_at"] = datetime.now(timezone.utc).isoformat()

                    accepted.append(row)
                    seen.add(nk)
                    print(f"  [{len(accepted)}/{args.target}] ok {fn}", flush=True)

                time.sleep(1.0)

    if args.append_out:
        new_rows = accepted[pre_existing_count:]
        open_mode = "a" if out_path.exists() and pre_existing_count > 0 else "w"
    else:
        new_rows = accepted
        open_mode = "w"

    with out_path.open(open_mode, encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(new_rows)} new line(s) to {out_path} (total unique rows in memory: {len(accepted)})")
    if len(accepted) < args.target:
        print(
            f"Warning: only {len(accepted)} repos collected (< {args.target}). "
            "Try more star slices, both licenses, a later --pushed-after, or a lower cutoff.",
            file=sys.stderr,
        )
        return 1
    return 0


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def verify_cmd(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("Set GITHUB_TOKEN", file=sys.stderr)
        return 2

    baseline = load_baseline_ids([Path(p).expanduser().resolve() for p in args.baseline_json])
    repos_path = Path(args.repos_jsonl).expanduser().resolve()
    bad = 0
    for row in _iter_jsonl(repos_path):
        fn = row.get("full_name", "")
        nk = _norm_repo(fn)
        if nk in baseline:
            print(f"FAIL overlap with baseline: {fn}")
            bad += 1
            continue
        data = fetch_repo(token, fn)
        time.sleep(0.2)
        if not data:
            print(f"FAIL could not fetch: {fn}")
            bad += 1
            continue
        if data.get("fork"):
            print(f"FAIL marked fork on GitHub: {fn}")
            bad += 1
            continue
        if fork_chain_hits_baseline(data, baseline):
            print(f"FAIL fork ancestry hits baseline: {fn}")
            bad += 1
            continue
        print(f"ok {fn}")
    if bad:
        print(f"\n{bad} problem(s).")
        return 1
    print("\nAll repos pass overlap + fork checks.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine / verify OOD GitHub repos.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="Search and write JSONL of OOD repos.")
    m.add_argument(
        "--baseline-json",
        action="append",
        default=[],
        help="Split JSON (repositories dict) or list of {full_name}. Repeatable.",
    )
    m.add_argument("--target", type=int, default=100)
    m.add_argument("--out", required=True, help="Output JSONL path.")
    m.add_argument(
        "--created-after",
        default=None,
        help="YYYY-MM-DD — repo created_at date must be strictly after this day "
        "(omit to infer from max(created_at) over baseline via API).",
    )
    m.add_argument(
        "--infer-cache",
        default=None,
        help="JSON path to cache baseline created_at from API (default: <out>.baseline_created_cache.json)",
    )
    m.add_argument("--pushed-after", default=DEFAULT_PUSHED_AFTER, help="pushed:>= filter (YYYY-MM-DD).")
    m.add_argument("--size-range", default=DEFAULT_SIZE_RANGE, help="GitHub size range in KB, e.g. 100..204800.")
    m.add_argument(
        "--star-slice",
        default=None,
        help="Comma-separated star ranges for search shards (default: built-in slices).",
    )
    m.add_argument("--max-pages", type=int, default=10, help="Max pages per query (100 repos/page, GitHub cap 1000).")
    m.add_argument(
        "--append-out",
        action="store_true",
        help="Append to output file instead of requiring a non-existent file.",
    )

    v = sub.add_parser("verify", help="Re-check JSONL against baseline + fork ancestry.")
    v.add_argument("--baseline-json", action="append", default=[], required=True)
    v.add_argument("--repos-jsonl", required=True)

    args = ap.parse_args()
    if args.cmd == "mine":
        if not args.baseline_json:
            print("Provide at least one --baseline-json", file=sys.stderr)
            return 2
        if not args.created_after and not args.infer_cache:
            out = Path(args.out).expanduser()
            args.infer_cache = str(out.with_suffix(".baseline_created_cache.json"))
        return mine_cmd(args)
    if args.cmd == "verify":
        return verify_cmd(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
