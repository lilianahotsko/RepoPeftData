#!/usr/bin/env python3
"""
Expanded GitHub Repository Miner for scaling experiments.
Targets 2000+ Python+pytest repos with wider search criteria.

Usage:
    GITHUB_TOKEN=ghp_xxx python repos_collection/mine_repos_expanded.py
    GITHUB_TOKEN=ghp_xxx python repos_collection/mine_repos_expanded.py --resume
"""

import os
import requests
import time
import json
import base64
import argparse
from datetime import datetime
from pathlib import Path

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OUTPUT_FILE = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
    "repos_expanded_mining.jsonl",
)
EXISTING_REPOS_FILE = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
    "repos_pytest.json",
)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

QUERIES = [
    # Wider star range, MIT
    ("language:python pushed:>=2024-01-01 license:mit size:3000..15000", [
        "50..100", "101..200", "201..299",
    ]),
    # Original range with Apache-2.0
    ("language:python pushed:>=2024-01-01 license:apache-2.0 size:3000..15000", [
        "50..100", "101..200", "201..500", "501..1000",
    ]),
    # Wider size range (larger repos), MIT
    ("language:python pushed:>=2024-01-01 license:mit size:15001..50000", [
        "100..300", "301..500", "501..1000",
    ]),
    # Older repos still active
    ("language:python pushed:>=2023-06-01 license:mit size:3000..15000", [
        "100..200", "201..299",
    ]),
]

PYTEST_FILES = [
    "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
    "pyproject.toml", "setup.cfg", "tox.ini", "Pipfile",
]

TARGET_NEW = 1500


def safe_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                print(f"  [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r
            return None
        except requests.RequestException as e:
            print(f"  [error] {e}, retry {attempt+1}")
            time.sleep(2 ** attempt)
    return None


def search_repos(query, star_range, page=1):
    q = f"{query} stars:{star_range}"
    r = safe_get(
        "https://api.github.com/search/repositories",
        params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
    )
    return r.json() if r else {}


def check_pytest(full_name):
    for filename in PYTEST_FILES:
        url = f"https://api.github.com/repos/{full_name}/contents/{filename}"
        r = safe_get(url)
        if r and r.status_code == 200:
            try:
                content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="ignore")
                if "pytest" in content.lower():
                    return True
            except Exception:
                pass
    return False


def extract_metadata(item):
    return {
        "full_name": item["full_name"],
        "url": item["html_url"],
        "description": item.get("description", ""),
        "stars": item["stargazers_count"],
        "forks": item["forks_count"],
        "size_kb": item["size"],
        "language": item.get("language", ""),
        "license": item.get("license", {}).get("spdx_id", ""),
        "created_at": item["created_at"],
        "pushed_at": item["pushed_at"],
        "default_branch": item["default_branch"],
        "is_fork": item["fork"],
        "archived": item["archived"],
        "topics": item.get("topics", []),
        "uses_pytest": None,
        "mined_at": datetime.utcnow().isoformat() + "Z",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Load existing repos to skip
    existing_names = set()
    for fpath in [EXISTING_REPOS_FILE, OUTPUT_FILE.replace(".jsonl", ".json")]:
        if os.path.exists(fpath):
            try:
                data = json.load(open(fpath))
                if isinstance(data, list):
                    for r in data:
                        existing_names.add(r.get("full_name", ""))
            except:
                pass

    # Load already-mined from output JSONL
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    existing_names.add(r.get("full_name", ""))
                except:
                    pass
    existing_names.discard("")
    print(f"Existing repos to skip: {len(existing_names)}")

    collected = 0
    with open(OUTPUT_FILE, "a") as out:
        for query, star_ranges in QUERIES:
            if collected >= TARGET_NEW:
                break
            for star_range in star_ranges:
                if collected >= TARGET_NEW:
                    break
                print(f"\n[Query: {query[:40]}... stars:{star_range}]")
                for page in range(1, 11):
                    if collected >= TARGET_NEW:
                        break
                    data = search_repos(query, star_range, page)
                    items = data.get("items", [])
                    if not items:
                        break

                    page_new = 0
                    for item in items:
                        fn = item["full_name"]
                        if fn in existing_names or item["fork"] or item["archived"]:
                            continue
                        existing_names.add(fn)
                        meta = extract_metadata(item)
                        uses_pytest = check_pytest(fn)
                        meta["uses_pytest"] = uses_pytest
                        out.write(json.dumps(meta) + "\n")
                        out.flush()
                        collected += 1
                        page_new += 1
                        status = "pytest" if uses_pytest else "no-pytest"
                        if collected % 10 == 0:
                            print(f"  [{collected}/{TARGET_NEW}] {fn} ({status})")
                        time.sleep(0.3)

                    print(f"  page {page}: +{page_new} new | total mined: {collected}")
                    time.sleep(1)

    # Count results
    pytest_count = 0
    total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    total += 1
                    if r.get("uses_pytest"):
                        pytest_count += 1
                except:
                    pass
    print(f"\n[Done] {total} total new repos, {pytest_count} with pytest")


if __name__ == "__main__":
    main()
