
import os
from tqdm import tqdm

repo_root = '/home/lhotsko/scratch/repositories'
repo_names = open('filtered_repo_urls.txt').read().splitlines()
repo_names = [repo_name.strip() for repo_name in repo_names]


def clone_repository(repo_name, repo_root):
    url_parts = repo_name.rstrip('/').split('/')
    owner = url_parts[-2] 
    repo = url_parts[-1]  
    if repo.endswith('.git'):
        repo = repo[:-4]
    
    repo_path = os.path.join(repo_root, owner, repo)
    
    if os.path.exists(repo_path):
        return 'skipped', repo_path
    
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    
    clone_cmd = f"git clone {repo_name} {repo_path} > /dev/null 2>&1"
    result = os.system(clone_cmd)
    
    if result == 0:
        return 'success', repo_path
    else:
        return 'failed', repo_path

success_count = 0
skip_count = 0
fail_count = 0

with tqdm(repo_names, desc="Cloning repositories") as pbar:
    for repo_name in pbar:
        status, repo_path = clone_repository(repo_name, repo_root)
        if status == 'success':
            success_count += 1
        elif status == 'skipped':
            skip_count += 1
        else:
            fail_count += 1
        
        pbar.set_postfix({
            'success': success_count,
            'skipped': skip_count,
            'failed': fail_count
        })


