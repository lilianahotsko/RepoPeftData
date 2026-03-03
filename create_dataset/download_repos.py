import os
import json
import subprocess

target_dir = "/home/lhotsko/scratch/REPO_DATASET/repositories"
repos_files = [
    "/home/lhotsko/scratch/repos_pytest_stars_300_1000.json",
    "/home/lhotsko/scratch/repos_pytest.json",
]

all_repos = []
for repo_file in repos_files:
    with open(repo_file) as f:
        repos = json.load(f)
    all_repos.extend(repos)

all_repos = [r for r in all_repos if r["uses_pytest"]]
print(f"Found {len(all_repos)} repos with pytest")
print(f"Joint size: {sum(r['size_kb'] for r in all_repos):,} KB")
print(f"Target dir: {target_dir}\n")

cloned = 0
skipped = 0
for i, repo in enumerate(all_repos, 1):
    owner, repo_name = repo["full_name"].split("/")
    repo_dir = os.path.join(target_dir, owner, repo_name)
    if os.path.exists(os.path.join(repo_dir, ".git")):
        skipped += 1
        if skipped <= 3 or skipped % 100 == 0:
            print(f"  [{i}/{len(all_repos)}] Skipped (already cloned): {repo['full_name']}")
        continue
    print(f"  [{i}/{len(all_repos)}] Cloning: {repo['full_name']}")
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    subprocess.run(["git", "clone", repo["url"], repo_dir])
    cloned += 1

print(f"\nDone. Cloned: {cloned}, Skipped: {skipped}")

