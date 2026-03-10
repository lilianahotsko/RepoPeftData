#!/bin/bash
#SBATCH --job-name=scale_pipe
#SBATCH --output=slurm_logs/scale_pipeline_%j.out
#SBATCH --error=slurm_logs/scale_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --account=def-yuntian

# Phase 3 Data Pipeline: Clone new repos, extract tests, create QnA, compute embeddings
# Prerequisite: run mine_repos_expanded.py first to populate repos_expanded_mining.jsonl

source scripts/slurm/common.sh
mkdir -p slurm_logs

MINED_FILE="$SCRATCH/repos_expanded_mining.jsonl"
REPOS_DIR="$SCRATCH/REPO_DATASET/repositories"

echo "===== Phase 3: Process new repos for scaling ====="
echo "Start: $(date)"

# Step 1: Clone new repos with pytest
echo "--- Step 1: Clone new repos ---"
python3 << 'PYEOF'
import json, subprocess, os, sys
from pathlib import Path

mined_file = os.environ["SCRATCH"] + "/repos_expanded_mining.jsonl"
repos_dir = Path(os.environ["SCRATCH"]) / "REPO_DATASET" / "repositories"

existing = set()
for author_dir in repos_dir.iterdir():
    if author_dir.is_dir():
        for rd in author_dir.iterdir():
            if rd.is_dir():
                existing.add(f"{author_dir.name}/{rd.name}")

cloned = 0
with open(mined_file) as f:
    for line in f:
        r = json.loads(line.strip())
        if not r.get("uses_pytest"):
            continue
        name = r["full_name"]
        if name in existing:
            continue
        author, repo = name.split("/")
        dest = repos_dir / author / repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{name}.git"
        branch = r.get("default_branch", "main")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, url, str(dest)],
                timeout=120, capture_output=True
            )
            cloned += 1
            print(f"  Cloned [{cloned}]: {name}")
        except Exception as e:
            print(f"  Failed: {name}: {e}")
print(f"Cloned {cloned} new repos")
PYEOF

# Step 2: Separate test files
echo "--- Step 2: Separate test files ---"
python create_dataset/2_separate_tests.py --repos-dir "$REPOS_DIR"

# Step 3: Create QnA pairs
echo "--- Step 3: Create QnA pairs ---"
python create_dataset/create_qnas_v2.py --repos-dir "$REPOS_DIR"

# Step 4: Compute embeddings
echo "--- Step 4: Compute embeddings ---"
python create_dataset/embed_repos.py --repos-dir "$REPOS_DIR"

# Step 5: Recreate expanded splits
echo "--- Step 5: Recreate expanded splits ---"
python create_dataset/expand_train_split.py --min-qnas 1

# Step 6: Quick stats
echo "--- Stats ---"
python3 -c "
import json
from pathlib import Path
p = Path('$SCRATCH/REPO_DATASET/expanded/train.json')
d = json.loads(p.read_text())
repos = d['repositories']
pairs = sum(len(r['qna_pairs']) for r in repos.values())
print(f'Expanded training: {len(repos)} repos, {pairs} pairs')
"

echo "Phase 3 pipeline complete: $(date)"
