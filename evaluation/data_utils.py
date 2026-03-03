"""
Shared data loading utilities for RepoPeftBench evaluation.
"""

import json
import os
from pathlib import Path
from typing import Optional


def get_default_splits_dir() -> str:
    return os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )


def load_split(
    splits_dir: Path,
    split_name: str,
    limit_repos: Optional[int] = None,
    repo_filter: Optional[str] = None,
) -> list[dict]:
    """
    Load split JSON (e.g. cr_test.json).
    Returns list of {repo, prefix, target}.
    Filters out targets starting with comma.
    """
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if repo_filter is not None:
        repo_names = [r for r in repo_names if r == repo_filter]
    elif limit_repos is not None and limit_repos > 0:
        repo_names = repo_names[:limit_repos]
    items = []
    for repo in repo_names:
        r = repos[repo]
        pairs = r.get("qna_pairs", [])
        for p in pairs:
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue
            items.append({
                "repo": repo,
                "prefix": prefix,
                "target": target,
                "assertion_type": p.get("assertion_type", ""),
                "metadata": p.get("metadata", {}),
            })
    return items


def load_split_with_embeddings(
    splits_dir: Path,
    split_name: str,
    limit_repos: Optional[int] = None,
    repo_filter: Optional[str] = None,
) -> list[dict]:
    """
    Load split JSON with repo embeddings.
    Returns list of {repo, prefix, target, embedding}.
    Filters out targets starting with comma.
    """
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if repo_filter is not None:
        repo_names = [r for r in repo_names if r == repo_filter]
    elif limit_repos is not None and limit_repos > 0:
        repo_names = repo_names[:limit_repos]
    items = []
    for repo in repo_names:
        r = repos[repo]
        pairs = r.get("qna_pairs", [])
        emb = r.get("embedding")
        if emb is None:
            continue
        file_embeddings = r.get("file_embeddings")
        for p in pairs:
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue
            item = {
                "repo": repo,
                "prefix": prefix,
                "target": target,
                "embedding": emb,
                "assertion_type": p.get("assertion_type", ""),
                "metadata": p.get("metadata", {}),
            }
            if file_embeddings is not None:
                item["file_embeddings"] = file_embeddings
            items.append(item)
    return items


def get_bos_id(tok):
    """Get BOS token id for generation."""
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return tok.pad_token_id


def prepare_input_ids(prefix: str, tokenizer, bos_id: int, max_input_tokens: int) -> list[int]:
    """Tokenize prefix with left truncation."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = [bos_id] + prefix_ids
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
    return input_ids
