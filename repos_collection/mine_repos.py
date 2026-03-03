"""
GitHub Repository Miner
Collects Python repos with filters and checks for pytest usage.
"""

import os
import requests
import time
import json
import base64
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
TARGET_REPOS  = 2000
OUTPUT_FILE   = "/home/lhotsko/scratch/repos_pytest_stars_300_1000.json"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

BASE_QUERY = "language:python pushed:>=2024-01-01 license:mit size:3000..15000"

# Split star ranges to bypass the 1000-result cap per query
STAR_RANGES = [
    "300..400", "401..500", "501..700", "701..1000",
    # "1001..1500", "1501..2500", "2501..5000", "5001..10000", ">10000"
]

PYTEST_FILES = [
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "Pipfile",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - int(time.time()), 1)
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


def parse_star_range(star_range):
    """Parse '300..400' or '>10000' into (min_stars, max_stars)."""
    if star_range.startswith(">"):
        return (int(star_range[1:]), None)  # no upper bound
    parts = star_range.split("..")
    if len(parts) == 2:
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if parts[1] else None
        return (lo, hi)
    return (0, None)


def search_repos(star_range, page=1):
    q = f"{BASE_QUERY} stars:{star_range}"
    r = safe_get(
        "https://api.github.com/search/repositories",
        params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
    )
    return r.json() if r else {}


def check_pytest(full_name):
    """Returns True if pytest is found in any common dependency file."""
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
        "full_name":        item["full_name"],
        "url":              item["html_url"],
        "description":      item.get("description", ""),
        "stars":            item["stargazers_count"],
        "forks":            item["forks_count"],
        "watchers":         item["watchers_count"],
        "size_kb":          item["size"],
        "open_issues":      item["open_issues_count"],
        "language":         item.get("language", ""),
        "license":          item.get("license", {}).get("spdx_id", ""),
        "created_at":       item["created_at"],
        "pushed_at":        item["pushed_at"],
        "updated_at":       item["updated_at"],
        "default_branch":   item["default_branch"],
        "is_fork":          item["fork"],
        "archived":         item["archived"],
        "has_wiki":         item["has_wiki"],
        "topics":           item.get("topics", []),
        "uses_pytest":      None,   # filled in next step
        "mined_at":         datetime.utcnow().isoformat() + "Z",
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # all_repos   = {}   # full_name -> metadata (dedup by full_name)
    # collected   = 0

    # print(f"[*] Mining up to {TARGET_REPOS} repos ...\n")

    # # Clear output file for incremental writes
    # with open(OUTPUT_FILE, "w") as f:
    #     pass

    # # ── Step 1: collect candidates ────────────────────────────────────────────
    # for star_range in STAR_RANGES:
    #     if collected >= TARGET_REPOS:
    #         break
    #     min_stars, max_stars = parse_star_range(star_range)
    #     print(f"[star range {star_range}]")

    #     for page in range(1, 11):
    #         if collected >= TARGET_REPOS:
    #             break

    #         data  = search_repos(star_range, page)
    #         items = data.get("items", [])
    #         if not items:
    #             break

    #         added = 0
    #         for item in items:
    #             fn = item["full_name"]
    #             stars = item.get("stargazers_count", 0)
    #             # Client-side filter: GitHub's search index can be stale
    #             if stars < min_stars:
    #                 continue
    #             if max_stars is not None and stars > max_stars:
    #                 continue
    #             if fn not in all_repos:
    #                 all_repos[fn] = extract_metadata(item)
    #                 collected += 1
    #                 added += 1
    #                 with open(OUTPUT_FILE, "a") as f:
    #                     f.write(json.dumps(all_repos[fn]) + "\n")

    #         print(f"  page {page}: +{added} (filtered by stars) | total {collected}")
    #         time.sleep(1)   # stay under secondary rate limits

    # print(f"\n[*] Collected {len(all_repos)} unique repos. Checking pytest usage ...\n")

    # # ── Step 2: check pytest ──────────────────────────────────────────────────
    # repos_list = list(all_repos.values())
    repos_list = json.load(open(OUTPUT_FILE))
    for i, repo in enumerate(repos_list):
        uses = check_pytest(repo["full_name"])
        repo["uses_pytest"] = uses
        status = "✓ pytest" if uses else "✗ no pytest"
        print(f"  [{i+1}/{len(repos_list)}] {repo['full_name']} — {status}")
        time.sleep(0.3)

    # ── Step 3: save (overwrite with JSON array for final format) ────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(repos_list, f, indent=2)

    pytest_count = sum(1 for r in repos_list if r["uses_pytest"])
    print(f"\n[✓] Done. {len(repos_list)} repos saved to {OUTPUT_FILE}")
    print(f"    uses_pytest=True : {pytest_count}")
    print(f"    uses_pytest=False: {len(repos_list) - pytest_count}")

if __name__ == "__main__":
    main()
