import os
import shutil
from pathlib import Path
from tqdm import tqdm

repos = open('filtered_repo_urls.txt').read().splitlines()
repos = [repo.replace('https://github.com/', '') for repo in repos]

test_files_root = '/home/lhotsko/scratch/test_files'
repos_without_tests_root = '/home/lhotsko/scratch/repos_without_tests'


def find_test_items(repo_path, test_items=None, current_path=""):

    if test_items is None:
        test_items = []
    
    try:
        for item in os.listdir(repo_path):
            if item.startswith('.') or item in ['__pycache__', 'node_modules', '.git']:
                continue
            
            item_path = os.path.join(repo_path, item)
            relative_path = os.path.join(current_path, item) if current_path else item
            
            if 'test' in item.lower():
                if os.path.isdir(item_path):
                    test_items.append((relative_path, True, item_path))
                elif os.path.isfile(item_path):
                    test_items.append((relative_path, False, item_path))
            else:
                if os.path.isdir(item_path):
                    find_test_items(item_path, test_items, relative_path)
    
    except (PermissionError, OSError):
        pass
    
    return test_items


def should_ignore_test_item(name, path):
    if 'test' in name.lower():
        return True
    return False


def copy_cleaned_repo(repo_path, repo_name, repos_without_tests_root):
    dest_root = os.path.join(repos_without_tests_root, repo_name)
    
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    
    os.makedirs(dest_root, exist_ok=True)
    
    def should_skip(item_name):
        if item_name.startswith('.') or item_name in ['__pycache__', 'node_modules']:
            return True
        if 'test' in item_name.lower():
            return True
        return False
    
    def copy_recursive(src, dst):
        try:
            for item in os.listdir(src):
                if should_skip(item):
                    continue
                
                src_path = os.path.join(src, item)
                dst_path = os.path.join(dst, item)
                
                if os.path.isdir(src_path):
                    os.makedirs(dst_path, exist_ok=True)
                    copy_recursive(src_path, dst_path)
                elif os.path.isfile(src_path):
                    if not item.endswith(('.pyc', '.pyo')):
                        shutil.copy2(src_path, dst_path)
        except (PermissionError, OSError):
            pass
    
    try:
        copy_recursive(repo_path, dest_root)
        return True
    except Exception:
        return False


def copy_test_files(repo_path, repo_name, test_files_root):
    test_items = find_test_items(repo_path)
    
    if not test_items:
        return 0
    
    # Sort items: directories first, then files, and by path length (to handle nested dirs)
    test_items.sort(key=lambda x: (not x[1], len(x[0])))
    
    dest_root = os.path.join(test_files_root, repo_name)
    os.makedirs(dest_root, exist_ok=True)
    
    copied_count = 0
    copied_paths = set()
    
    for relative_path, is_directory, source_path in test_items:
        dest_path = os.path.join(dest_root, relative_path)
        
        if any(dest_path.startswith(copied + os.sep) or dest_path == copied 
               for copied in copied_paths):
            continue
        
        try:
            if is_directory:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path, 
                              ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.pyo'))
                copied_paths.add(dest_path)

                for root, dirs, files in os.walk(dest_path):
                    copied_count += len([f for f in files 
                                       if not f.endswith(('.pyc', '.pyo')) and not f.startswith('.')])
            else:
                parent_dir = os.path.dirname(dest_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                copied_paths.add(dest_path)
        except (PermissionError, OSError, shutil.Error) as e:
            continue  
    
    return copied_count


def has_test_folder(path, max_depth=5, current_depth=0):
    if current_depth > max_depth:
        return False
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                if 'test' in item.lower():
                    return True 
                if has_test_folder(item_path, max_depth, current_depth + 1):
                    return True
            elif os.path.isfile(item_path) and 'test' in item.lower():
                return True
                
    except (PermissionError, OSError):
        pass
    return False

counter = 0
repos_without_tests = []
copied_repos = 0
total_files_copied = 0

for repo in tqdm(repos, desc="Processing repositories"):
    repo_path = os.path.join('/home/lhotsko/scratch/repositories', repo)
    
    if not os.path.exists(repo_path):
        continue
    
    has_test_at_root = False
    try:
        for item in os.listdir(repo_path):
            item_path = os.path.join(repo_path, item)
            if os.path.isdir(item_path) and not item.startswith('.') and 'test' in item.lower():
                has_test_at_root = True
                break
            elif os.path.isfile(item_path) and 'test' in item.lower():
                has_test_at_root = True
                break
    except (PermissionError, OSError):
        pass
    
    if not has_test_at_root:
        found = has_test_folder(repo_path)
        if not found:
            counter += 1
            repos_without_tests.append(repo)
        else:
            files_copied = copy_test_files(repo_path, repo, test_files_root)
            if files_copied > 0:
                copied_repos += 1
                total_files_copied += files_copied
                copy_cleaned_repo(repo_path, repo, repos_without_tests_root)
            else:
                counter += 1
                repos_without_tests.append(repo)
    else:
        files_copied = copy_test_files(repo_path, repo, test_files_root)
        if files_copied > 0:
            copied_repos += 1
            total_files_copied += files_copied
            copy_cleaned_repo(repo_path, repo, repos_without_tests_root)
        else:
            counter += 1
            repos_without_tests.append(repo)

print(f"\nSummary:")
print(f"Repositories without tests: {counter}")
print(f"Repositories with tests copied: {copied_repos}")
print(f"Total test files copied: {total_files_copied}")
if repos_without_tests:
    print(f"\nRepositories without tests:")
    print(repos_without_tests)