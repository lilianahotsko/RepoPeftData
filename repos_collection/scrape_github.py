#!/usr/bin/env python3
import os
import time
import json
import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    raise SystemExit("Set GITHUB_TOKEN first")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "pytest-miner",
}

SEARCH_URL = f"{GITHUB_API}/search/code"


def gh_request(params, max_retries=5):
    retry_count = 0
    while True:
        try:
            r = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=60)

            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                print(f"[RATE LIMIT 403] Sleeping {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 429:
                # 429 Too Many Requests - check for Retry-After header or rate limit reset
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                else:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(5, reset - int(time.time()) + 2)
                print(f"[RATE LIMIT 429] Sleeping {wait}s (Retry-After: {retry_after}, Reset: {r.headers.get('X-RateLimit-Reset', 'N/A')})")
                time.sleep(wait)
                continue

            if r.status_code == 408:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[TIMEOUT] Max retries ({max_retries}) exceeded for query: {params.get('q', 'N/A')}")
                    raise
                wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                print(f"[TIMEOUT 408] Retry {retry_count}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
                continue

            if r.status_code in (500, 502, 503, 504):
                time.sleep(3)
                continue

            if r.status_code == 422:
                # Print error details for debugging
                try:
                    error_data = r.json()
                    print(f"[ERROR 422] Query: {params.get('q', 'N/A')}")
                    print(f"[ERROR 422] Response: {error_data}")
                except:
                    print(f"[ERROR 422] Query: {params.get('q', 'N/A')}")
                    print(f"[ERROR 422] Response text: {r.text[:500]}")
                raise

            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count > max_retries:
                print(f"[TIMEOUT] Max retries ({max_retries}) exceeded for query: {params.get('q', 'N/A')}")
                raise
            wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
            print(f"[TIMEOUT Exception] Retry {retry_count}/{max_retries} after {wait_time}s")
            time.sleep(wait_time)
            continue


# def pytest_queries(star_range):
#     base_signals = [
#         "filename:pytest.ini",
#         "filename:conftest.py",
#         '"import pytest"',
#         '"@pytest.fixture"',
#         '"pytest.mark"',
#     ]
#     return [
#         f"{sig} in:file language:Python fork:false stars:{star_range}"
#         for sig in base_signals
#     ]

def pytest_queries(star_range=None, pushed_range=None):
    base_signals = [
        "filename:pytest.ini",
        "filename:conftest.py",
        '"import pytest"',
        '"@pytest.fixture"',
        '"pytest.mark"',
    ]
    # Note: stars: and pushed: qualifiers are not supported in code search API
    # They're only available in repository search. We'll filter by stars after fetching.
    # Use NOT is:fork instead of fork:false
    queries = [
        f"{sig} language:Python NOT is:fork"
        for sig in base_signals
    ]
    return queries



def parse_star_range(star_range):
    """Parse star range like '6..20' into (min, max) tuple."""
    if ".." in star_range:
        parts = star_range.split("..")
        min_stars = int(parts[0]) if parts[0] else 0
        max_stars = int(parts[1]) if parts[1] else float('inf')
        return (min_stars, max_stars)
    elif star_range.startswith(">"):
        return (int(star_range[1:]), float('inf'))
    else:
        return (int(star_range), int(star_range))


def get_repo_details(full_name, max_retries=3):
    """Fetch full repository details including star count."""
    url = f"{GITHUB_API}/repos/{full_name}"
    retry_count = 0
    while True:
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                print(f"[RATE LIMIT 403] Sleeping {wait}s")
                time.sleep(wait)
                continue
            
            if r.status_code == 429:
                # 429 Too Many Requests
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                else:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(5, reset - int(time.time()) + 2)
                print(f"[RATE LIMIT 429] Sleeping {wait}s for repo details")
                time.sleep(wait)
                continue
            
            if r.status_code == 408:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[TIMEOUT] Max retries ({max_retries}) exceeded for repo: {full_name}")
                    return None
                wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
                time.sleep(wait_time)
                continue
            
            if r.status_code in (500, 502, 503, 504):
                time.sleep(3)
                continue
            
            if r.status_code == 404:
                # Repository not found or private
                return None
            
            if r.status_code == 200:
                return r.json()
            
            # For other errors, return None
            return None
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count > max_retries:
                print(f"[TIMEOUT] Max retries ({max_retries}) exceeded for repo: {full_name}")
                return None
            wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
            time.sleep(wait_time)
            continue


def mine_range(star_range, max_pages=10, pushed_range=None):
    repos = {}
    min_stars, max_stars = parse_star_range(star_range)
    seen_repos = set()  # Track repos we've already fetched details for
    
    for q in pytest_queries(star_range, pushed_range=pushed_range):
        for page in range(1, max_pages + 1):
            params = {"q": q, "per_page": 100, "page": page}
            data = gh_request(params)

            items = data.get("items", [])
            if not items:
                break

            for it in items:
                repo = it["repository"]
                full_name = repo["full_name"]
                
                # Skip if we've already processed this repo
                if full_name in repos:
                    continue
                
                # Fetch full repo details if we haven't seen it yet
                if full_name not in seen_repos:
                    seen_repos.add(full_name)
                    repo_details = get_repo_details(full_name)
                    if not repo_details:
                        continue
                    
                    stars = repo_details.get("stargazers_count", 0)
                    
                    # Filter by star range (since stars: qualifier isn't supported in code search)
                    if stars < min_stars or stars > max_stars:
                        continue
                    
                    repos[full_name] = {
                        "full_name": full_name,
                        "url": repo_details.get("html_url", repo.get("html_url", "")),
                        "stars": stars,
                        "language": repo_details.get("language", repo.get("language")),
                        "matched_query": q,
                    }
                    
                    time.sleep(0.1)  # Small delay to avoid rate limits

            time.sleep(0.2)

    return repos


def load_existing_repos(outfile):
    """Load existing repos from file to avoid duplicates."""
    repos = {}
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            for line in f:
                if line.strip():
                    repo = json.loads(line)
                    repos[repo["full_name"]] = repo
    return repos


def append_repos_to_file(repos, outfile):
    """Append new repos to file, skipping duplicates."""
    existing = load_existing_repos(outfile)
    new_count = 0
    
    with open(outfile, "a") as f:
        for repo in sorted(repos.values(), key=lambda x: x["stars"], reverse=True):
            if repo["full_name"] not in existing:
                f.write(json.dumps(repo) + "\n")
                existing[repo["full_name"]] = repo
                new_count += 1
    
    return new_count, len(existing)


def main(target=5000, outfile="repos_collection/pytest_repos_5k.jsonl"):
    star_slices = [
        # "6..20", "21..50", 
        # "51..100",
        "101..300", "301..700", "701..2000", "2001..100000"
    ]
    pushed_range = "2025-01-01..2025-12-31"
    
    # Load existing repos if file exists
    all_repos = load_existing_repos(outfile)
    print(f"Loaded {len(all_repos)} existing repos from {outfile}")

    for star_range in star_slices:
        print(f"\n🔎 Mining stars:{star_range}")
        repos = mine_range(star_range, pushed_range=pushed_range)

        # Append new repos to file immediately
        new_count, total_count = append_repos_to_file(repos, outfile)
        print(f"Added {new_count} new repos, total: {total_count}")

        if total_count >= target:
            break

    print(f"\n✅ Final count: {total_count} repos in {outfile}")


if __name__ == "__main__":
    main()
