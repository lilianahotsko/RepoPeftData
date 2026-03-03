#!/usr/bin/env python3
"""Fetch repo size (in KB) from GitHub API and append it to each line in pytest_repos_5k.jsonl."""

import json
import os
import time
import requests
from pathlib import Path

# Load .env manually
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
INPUT = Path(__file__).resolve().parent / "pytest_repos_5k.jsonl"
OUTPUT = INPUT  # overwrite in place

def get_repo_size(full_name: str) -> int | None:
    """Return repo size in KB from GitHub API, or None on failure."""
    url = f"https://api.github.com/repos/{full_name}"
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("size")
        elif resp.status_code == 403:
            # Rate limited — wait for reset
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 5)
            print(f"  Rate limited, sleeping {wait}s...")
            time.sleep(wait)
        elif resp.status_code == 404:
            print(f"  404 Not Found: {full_name}")
            return None
        else:
            print(f"  HTTP {resp.status_code} for {full_name}, retry {attempt+1}")
            time.sleep(2 ** attempt)
    return None


def main():
    lines = [l for l in INPUT.read_text().splitlines() if l.strip() and l.strip().startswith("{")]
    print(f"Loaded {len(lines)} repos from {INPUT}")
    updated = []
    for i, line in enumerate(lines):
        obj = json.loads(line)
        full_name = obj["full_name"]

        if "repo_size_kb" in obj:
            updated.append(json.dumps(obj, ensure_ascii=False))
            continue

        size = get_repo_size(full_name)
        if size is not None:
            obj["repo_size_kb"] = size
        else:
            obj["repo_size_kb"] = -1  # mark as failed

        updated.append(json.dumps(obj, ensure_ascii=False))

        if (i + 1) % 50 == 0 or i == len(lines) - 1:
            print(f"  [{i+1}/{len(lines)}] {full_name} -> {size} KB")

    OUTPUT.write_text("\n".join(updated) + "\n")
    print(f"Done. Written {len(updated)} lines to {OUTPUT}")

    # Summary
    sizes = [json.loads(l).get("repo_size_kb", -1) for l in updated]
    valid = [s for s in sizes if s > 0]
    failed = sum(1 for s in sizes if s == -1)
    print(f"Valid: {len(valid)}  Failed: {failed}")
    if valid:
        print(f"Size (KB): min={min(valid)}  max={max(valid)}  "
              f"mean={sum(valid)/len(valid):.0f}  median={sorted(valid)[len(valid)//2]}")


if __name__ == "__main__":
    main()
