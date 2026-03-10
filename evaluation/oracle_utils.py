"""
Shared oracle context utilities for training and evaluation scripts.

Load pre-built oracle contexts (from baselines/oracle_context/build_context.py)
and augment QnA prefixes with resolved source code.
"""

import json
import os
from pathlib import Path
from typing import Optional


def get_default_oracle_cache_dir(version: str = "v1") -> str:
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    if version == "v2":
        return os.path.join(scratch, "ORACLE_CONTEXT_CACHE_V2")
    return os.path.join(scratch, "ORACLE_CONTEXT_CACHE")


def load_oracle_cache(cache_dir: Path, repo_name: str) -> dict:
    """Load pre-built oracle contexts for a repo. Returns {key: context_dict}."""
    safe_name = repo_name.replace("/", "__")
    cache_path = cache_dir / f"{safe_name}.json"
    if not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return data.get("contexts", {})


def lookup_oracle_context(contexts: dict, metadata: dict) -> str:
    """Look up extracted code for a specific QnA pair by file::lineno key."""
    key = f"{metadata.get('file', '')}::{metadata.get('lineno', 0)}"
    ctx = contexts.get(key)
    if ctx and ctx.get("extracted_code"):
        return ctx["extracted_code"]
    return ""


def augment_prefix_with_oracle(
    prefix: str,
    oracle_code: str,
    max_oracle_chars: Optional[int] = None,
) -> str:
    """
    Prepend oracle context to prefix with a separator.

    If *max_oracle_chars* is set, the oracle code is truncated from the
    end (keeping the first definitions which are usually most relevant).
    """
    if not oracle_code or not oracle_code.strip():
        return prefix

    if max_oracle_chars is not None and len(oracle_code) > max_oracle_chars:
        oracle_code = oracle_code[:max_oracle_chars]

    return oracle_code + "\n\n\n" + prefix


def load_oracle_for_split(
    cache_dir: Path,
    split_path: Path,
) -> dict[str, dict]:
    """
    Load oracle contexts for all repos in a split JSON.
    Returns {repo_name: {key: context_dict}}.
    """
    if not split_path.exists():
        return {}
    data = json.loads(split_path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    result = {}
    for repo_name in repos:
        result[repo_name] = load_oracle_cache(cache_dir, repo_name)
    return result
