#!/usr/bin/env python3
"""
Generate the *question answering* (``qa``) QnA dataset for Code2LoRA.

For every kept commit in the GRU v2 commit parquets, ask an OpenAI model to
produce ``--qnas-per-commit`` (default 10) question/answer pairs *about that
commit's production-code change* (its ``production_code_diff``). The output
mirrors the GRU split layout 1:1 -- each generated row keeps the commit's
``in_repo_split`` (train/val/test => ir splits) and ``cross_repo_split``
(train/cr_val/cr_test) labels -- so the resulting ``qna/{split}.parquet`` files
plug straight into the existing GRU pipeline alongside ``assert_rhs`` /
``code_gen``.

Cost control: requests are submitted through the OpenAI **Batch API**
(``/v1/chat/completions``, 24h window), which is ~50% cheaper than synchronous
calls. The default model is ``gpt-5.4-mini``.

Because batches are asynchronous, the workflow has three phases (run as
subcommands), plus a ``run`` convenience that chains them:

    # 1. build request shards, upload, create batches, write a manifest
    python create_dataset/generate_qna_batch.py submit \
        --commits-dir $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits \
        --work-dir    $SCRATCH/REPO_DATASET/qna_qa_work \
        --splits train cr_val cr_test

    # 2. poll until every batch reaches a terminal state
    python create_dataset/generate_qna_batch.py poll \
        --work-dir $SCRATCH/REPO_DATASET/qna_qa_work --wait

    # 3. download outputs, parse, write qna/{split}.parquet
    python create_dataset/generate_qna_batch.py fetch \
        --work-dir $SCRATCH/REPO_DATASET/qna_qa_work \
        --out-dir  $SCRATCH/REPO_DATASET/qna_qa_hf/qna

    # ...or all three with polling in between
    python create_dataset/generate_qna_batch.py run \
        --commits-dir $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits \
        --work-dir    $SCRATCH/REPO_DATASET/qna_qa_work \
        --out-dir     $SCRATCH/REPO_DATASET/qna_qa_hf/qna

The OpenAI key is read from ``OPENAI_API_KEY`` (loaded from the repo-root
``.env`` if present). Nothing is ever hard-coded.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_QNAS_PER_COMMIT = 10
TASK_ID = "qa"

# OpenAI Batch hard limits (kept conservative for headroom).
MAX_REQUESTS_PER_BATCH = 40_000      # API max is 50k
MAX_BATCH_FILE_BYTES = 180 * 1024 * 1024  # API max is 200 MB

# Stable taxonomy. GPT is asked to emit one QnA per category, in this order,
# so every commit yields a balanced set. ``question_index`` == position here.
CATEGORIES = [
    "intent",            # 0 main purpose of the change
    "new_api",           # 1 new function/method/class introduced
    "signature_change",  # 2 changed signature / type hints
    "behavior_change",   # 3 how runtime behaviour differs
    "refactor",          # 4 why a construct was restructured/renamed
    "dependency",        # 5 import / dependency change
    "scope",             # 6 which files/modules changed and how
    "usage",             # 7 how a user invokes the new/changed feature
    "edge_case",         # 8 tolerances / None-handling / defaults
    "risk",              # 9 side-effects / backward-compat concerns
]

# Selection priority: when a commit gets fewer than the full 10 questions
# (small diffs), keep the most broadly-answerable, high-value categories first.
CATEGORY_PRIORITY = [
    "intent",
    "behavior_change",
    "scope",
    "new_api",
    "signature_change",
    "usage",
    "refactor",
    "dependency",
    "edge_case",
    "risk",
]

_CATEGORY_HINTS = {
    "intent": "the single main purpose/intent of this change",
    "new_api": "a new function, method, or class introduced (name + what it does)",
    "signature_change": "a changed function/method signature or type annotation",
    "behavior_change": "how the runtime behaviour differs after this change",
    "refactor": "why an existing construct was restructured, renamed, or simplified",
    "dependency": "an import or dependency change and why it was made",
    "scope": "which files/modules are touched and the nature of each change",
    "usage": "how a developer would call or use the changed/added code",
    "edge_case": "an edge case, default value, tolerance, or None/empty handling",
    "risk": "a potential side-effect, regression risk, or backward-compat concern",
}

SYSTEM_PROMPT = (
    "You are a senior software engineer writing a high-quality question-answering "
    "dataset about Git commits. You are given the production-code diff of one "
    "commit. Produce concise, factual question/answer pairs that can be answered "
    "*solely* from the diff (plus the repo name). Do not invent details that are "
    "not supported by the diff. Answers must be self-contained and specific "
    "(name the functions, files, parameters, or values involved). Avoid yes/no "
    "questions. Respond with JSON only."
)


def _select_categories(n: int) -> List[str]:
    """Pick ``n`` categories by priority (high-value first), in canonical order."""
    n = max(1, min(n, len(CATEGORY_PRIORITY)))
    chosen = set(CATEGORY_PRIORITY[:n])
    return [c for c in CATEGORIES if c in chosen]


def build_user_prompt(repo_id: str, commit_sha: str, diff: str,
                      categories: List[str]) -> str:
    """Instruction + the diff. JSON shape is described in-prompt (json mode)."""
    n = len(categories)
    cat_lines = [
        f"  {i + 1}. category \"{cat}\": ask about {_CATEGORY_HINTS[cat]}"
        for i, cat in enumerate(categories)
    ]
    cats = "\n".join(cat_lines)
    return (
        f"Repository: {repo_id}\n"
        f"Commit: {commit_sha}\n\n"
        f"Generate exactly {n} question/answer pairs about the change below. "
        f"Produce one pair for each of the following slots, in this order:\n"
        f"{cats}\n\n"
        f"Only ask questions that are clearly answerable from the diff; never "
        f"invent details. Each answer must be specific and self-contained.\n\n"
        f"Return a JSON object with this exact shape:\n"
        f'{{"qnas": [{{"category": "<one of the categories above>", '
        f'"question": "<question text>", "answer": "<answer text>"}}, ...]}}\n'
        f"The \"qnas\" array must contain exactly {n} items in the order above.\n\n"
        f"=== PRODUCTION CODE DIFF START ===\n"
        f"{diff}\n"
        f"=== PRODUCTION CODE DIFF END ==="
    )


# --------------------------------------------------------------------------- #
# Env / client
# --------------------------------------------------------------------------- #

def _load_dotenv(repo_root: Path) -> None:
    """Populate os.environ from <repo_root>/.env without extra deps."""
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
    except OSError:
        pass


def _repo_root() -> Path:
    # create_dataset/ -> repo root is the parent dir.
    return Path(__file__).resolve().parent.parent


def get_client():
    """Construct an OpenAI client, loading the key from env/.env."""
    _load_dotenv(_repo_root())
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set (looked in environment and repo-root .env)."
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise SystemExit("openai SDK missing. Install with: pip install openai")
    return OpenAI(api_key=api_key)


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise SystemExit(
            "pyarrow is required. On the cluster: `module load arrow` "
            "(or `pip install pyarrow`)."
        )


# --------------------------------------------------------------------------- #
# Input: iterate commits from the GRU v2 commit parquets
# --------------------------------------------------------------------------- #

_NEEDED_COLS = [
    "repo_id", "cross_repo_split", "commit_index",
    "commit_sha", "in_repo_split", "production_code_diff",
]


_TRUNC_NOTE = "\n... [diff truncated for length] ..."
_ENC = None
_ENC_TRIED = False


def _get_encoder():
    """Return a cached tiktoken encoder (o200k_base, gpt-5 family), or None."""
    global _ENC, _ENC_TRIED
    if _ENC_TRIED:
        return _ENC
    _ENC_TRIED = True
    try:
        import tiktoken  # type: ignore
        _ENC = tiktoken.get_encoding("o200k_base")
    except Exception:  # pragma: no cover - fall back to char heuristic
        _ENC = None
    return _ENC


def _count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)  # ~4 chars/token heuristic
    # Diffs can contain literal special-token text (e.g. <|endoftext|>);
    # treat everything as normal text rather than raising.
    return len(enc.encode(text, disallowed_special=()))


def _truncate_diff(diff: str, max_chars: int) -> str:
    """Coarse character pre-cap (keeps giant diffs out of the tokenizer)."""
    if max_chars <= 0 or len(diff) <= max_chars:
        return diff
    return diff[:max_chars] + _TRUNC_NOTE


def _fit_diff_to_tokens(diff: str, repo_id: str, commit_sha: str,
                        categories: List[str], max_input_tokens: int) -> str:
    """Trim the diff so the whole request (system + user) fits the token budget.

    The prompt overhead (system prompt + instruction template + repo/sha) is
    measured with an empty diff, then the remaining budget is spent on diff
    tokens. Uses tiktoken when available, otherwise a char-per-token heuristic.
    """
    if max_input_tokens <= 0:
        return diff
    overhead = (_count_tokens(SYSTEM_PROMPT)
                + _count_tokens(build_user_prompt(repo_id, commit_sha, "",
                                                   categories))
                + 8)  # per-message formatting slack
    budget = max_input_tokens - overhead
    if budget <= 0:
        return ""  # instructions alone exceed the budget (very small caps)
    enc = _get_encoder()
    if enc is None:
        approx_chars = budget * 4
        if len(diff) <= approx_chars:
            return diff
        return diff[:approx_chars] + _TRUNC_NOTE
    toks = enc.encode(diff, disallowed_special=())
    if len(toks) <= budget:
        return diff
    keep = max(0, budget - _count_tokens(_TRUNC_NOTE))
    return enc.decode(toks[:keep]) + _TRUNC_NOTE


# Diff-size thresholds (in tokens) for the adaptive question count. A tiny
# one-line fix doesn't support 10 distinct questions; a large refactor does.
_ADAPTIVE_LO_TOKENS = 120     # at/below -> qnas_min
_ADAPTIVE_HI_TOKENS = 1_800   # at/above -> qnas_max


def _target_qna_count(diff: str, args) -> int:
    """Resolve how many QnAs to ask for this commit.

    Fixed at ``--qnas-per-commit`` unless ``--adaptive-qnas`` is set, in which
    case it scales linearly with diff token count between ``--qnas-min`` and
    ``--qnas-max``.
    """
    if not getattr(args, "adaptive_qnas", False):
        return args.qnas_per_commit
    lo, hi = args.qnas_min, args.qnas_max
    if hi <= lo:
        return hi
    ntok = _count_tokens(diff)
    if ntok <= _ADAPTIVE_LO_TOKENS:
        return lo
    if ntok >= _ADAPTIVE_HI_TOKENS:
        return hi
    frac = (ntok - _ADAPTIVE_LO_TOKENS) / (_ADAPTIVE_HI_TOKENS - _ADAPTIVE_LO_TOKENS)
    return int(round(lo + frac * (hi - lo)))


def _even_sample_indices(n: int, k: int) -> List[int]:
    """Pick ``k`` indices in [0, n) spread evenly, always including 0 and n-1.

    Used to subsample a repo's chronologically-ordered commits so the kept set
    spans the whole timeline (early -> late) rather than clustering. Because
    ``in_repo_split`` is itself a chronological 80/10/10, even spacing also
    roughly preserves the train/val/test ratio.
    """
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


_IR_ORDER = ["train", "val", "test"]


def _sample_repo_commits(rows: List[Dict[str, Any]], cap: int,
                         stratify: bool, min_eval: int) -> List[Dict[str, Any]]:
    """Pick up to ``cap`` commits for one repo.

    Plain mode: even temporal sampling across the whole timeline.
    Stratified mode: guarantee up to ``min_eval`` commits from each of the
    ``val``/``test`` in-repo splits (when present), then fill the rest from the
    remaining budget proportionally; even-sample within each stratum so each
    keeps its chronological spread. This protects ir_val/ir_test coverage.
    """
    if cap <= 0 or len(rows) <= cap:
        return rows
    if not stratify:
        idxs = _even_sample_indices(len(rows), cap)
        return [rows[i] for i in idxs]

    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_split.setdefault(r["in_repo_split"], []).append(r)
    for s in by_split:
        by_split[s].sort(key=lambda r: r["commit_index"])

    alloc: Dict[str, int] = {}
    # Reserve a minimum for the eval strata first.
    for s in ("val", "test"):
        if by_split.get(s):
            alloc[s] = min(min_eval, len(by_split[s]))
    remaining = cap - sum(alloc.values())
    # Distribute the rest proportionally to leftover capacity per stratum.
    leftover = {s: len(rs) - alloc.get(s, 0) for s, rs in by_split.items()}
    total_left = sum(v for v in leftover.values() if v > 0)
    if remaining > 0 and total_left > 0:
        for s, cap_left in leftover.items():
            if cap_left <= 0:
                continue
            add = int(round(remaining * cap_left / total_left))
            alloc[s] = alloc.get(s, 0) + min(add, cap_left)
    # Clamp to stratum sizes and trim/grow to hit ``cap`` exactly where possible.
    for s in list(alloc):
        alloc[s] = min(alloc[s], len(by_split[s]))
    # Greedy fix-up so the totals land on cap (respecting capacities).
    def _capacity_left() -> int:
        return sum(len(by_split[s]) - alloc.get(s, 0) for s in by_split)
    while sum(alloc.values()) < cap and _capacity_left() > 0:
        for s in _IR_ORDER + [x for x in by_split if x not in _IR_ORDER]:
            if s in by_split and alloc.get(s, 0) < len(by_split[s]):
                alloc[s] = alloc.get(s, 0) + 1
                if sum(alloc.values()) >= cap:
                    break
    while sum(alloc.values()) > cap:
        for s in reversed(_IR_ORDER):
            if alloc.get(s, 0) > 0 and sum(alloc.values()) > cap:
                # never drop reserved eval minimums
                floor = min_eval if s in ("val", "test") else 0
                if alloc[s] > floor:
                    alloc[s] -= 1

    picked: List[Dict[str, Any]] = []
    for s, rs in by_split.items():
        k = alloc.get(s, 0)
        if k <= 0:
            continue
        idxs = _even_sample_indices(len(rs), k)
        picked.extend(rs[i] for i in idxs)
    picked.sort(key=lambda r: r["commit_index"])
    return picked


def iter_commits(
    commits_dir: Path,
    splits: List[str],
    *,
    skip_empty_diff: bool,
    max_diff_chars: int,
    repos_filter: Optional[set],
    limit: int,
    max_commits_per_repo: int = 0,
    stratify_splits: bool = False,
    min_eval_commits: int = 2,
) -> Iterator[Dict[str, Any]]:
    """Yield one work item per kept commit across the requested split files.

    When ``max_commits_per_repo > 0``, each repo is capped to that many
    commits. With ``stratify_splits`` the cap reserves coverage for the
    val/test in-repo splits; otherwise commits are sampled evenly across the
    repo's chronological timeline.
    """
    _require_pyarrow()
    emitted = 0
    seen: set = set()  # (split_file, repo_id, commit_sha) de-dup guard
    for split in splits:
        path = commits_dir / f"{split}.parquet"
        if not path.is_file():
            print(f"[warn] missing split file: {path}", file=sys.stderr)
            continue
        table = pq.read_table(path, columns=_NEEDED_COLS)

        # Gather qualifying rows, grouped per repo (so we can cap/sample).
        per_repo: Dict[str, List[Dict[str, Any]]] = {}
        for row in table.to_pylist():
            repo_id = row["repo_id"]
            if repos_filter is not None and repo_id not in repos_filter:
                continue
            diff = row["production_code_diff"] or ""
            if skip_empty_diff and not diff.strip():
                continue
            key = (split, repo_id, row["commit_sha"])
            if key in seen:
                continue
            seen.add(key)
            per_repo.setdefault(repo_id, []).append({
                "split_file": split,
                "repo_id": repo_id,
                "cross_repo_split": row["cross_repo_split"],
                "commit_index": int(row["commit_index"]),
                "commit_sha": row["commit_sha"],
                "in_repo_split": row["in_repo_split"],
                "diff": _truncate_diff(diff, max_diff_chars),
            })

        for repo_id in sorted(per_repo):
            rows = sorted(per_repo[repo_id], key=lambda r: r["commit_index"])
            if max_commits_per_repo > 0:
                rows = _sample_repo_commits(
                    rows, max_commits_per_repo,
                    stratify_splits, min_eval_commits,
                )
            for item in rows:
                yield item
                emitted += 1
                if limit and emitted >= limit:
                    return


# --------------------------------------------------------------------------- #
# Request body
# --------------------------------------------------------------------------- #

def build_request_body(item: Dict[str, Any], args) -> "Tuple[Dict[str, Any], int]":
    n = _target_qna_count(item["diff"], args)
    categories = _select_categories(n)
    diff = _fit_diff_to_tokens(
        item["diff"], item["repo_id"], item["commit_sha"],
        categories, getattr(args, "max_input_tokens", 4000),
    )
    body: Dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    item["repo_id"], item["commit_sha"], diff, categories,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": args.max_completion_tokens,
    }
    if args.reasoning_effort and args.reasoning_effort.lower() != "none":
        body["reasoning_effort"] = args.reasoning_effort
    return body, len(categories)


def _parse_cap(args) -> int:
    """Upper bound on QnAs kept per commit (parser cap)."""
    if getattr(args, "adaptive_qnas", False):
        return args.qnas_max
    return args.qnas_per_commit


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #

def _manifest_path(work_dir: Path) -> Path:
    return work_dir / "manifest.json"


def _meta_path(work_dir: Path) -> Path:
    return work_dir / "requests_meta.jsonl"


def _load_manifest(work_dir: Path) -> Dict[str, Any]:
    p = _manifest_path(work_dir)
    if not p.is_file():
        raise SystemExit(f"No manifest at {p}. Run `submit` first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _save_manifest(work_dir: Path, manifest: Dict[str, Any]) -> None:
    _manifest_path(work_dir).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #

def cmd_submit(args) -> None:
    work_dir = Path(args.work_dir)
    req_dir = work_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    repos_filter = set(args.repos) if args.repos else None
    items = iter_commits(
        Path(args.commits_dir), args.splits,
        skip_empty_diff=not args.keep_empty_diff,
        max_diff_chars=args.max_diff_chars,
        repos_filter=repos_filter,
        limit=args.limit,
        max_commits_per_repo=args.max_commits_per_repo,
        stratify_splits=args.stratify_splits,
        min_eval_commits=args.min_eval_commits,
    )

    # Write request JSONL shards + a parallel meta JSONL (custom_id -> identity).
    shards: List[Dict[str, Any]] = []
    meta_f = _meta_path(work_dir).open("w", encoding="utf-8")
    seq = 0
    shard_idx = 0
    cur_path: Optional[Path] = None
    cur_f = None
    cur_count = 0
    cur_bytes = 0

    def _open_shard(idx: int):
        p = req_dir / f"requests_{idx:04d}.jsonl"
        return p, p.open("w", encoding="utf-8")

    def _close_shard():
        nonlocal cur_f
        if cur_f is not None:
            cur_f.close()
            shards.append({"shard": shard_idx, "path": str(cur_path),
                           "n_requests": cur_count})

    total = 0
    expected_qnas = 0
    per_split: Dict[str, int] = {}
    for item in items:
        custom_id = f"r{seq:08d}"
        body, n_target = build_request_body(item, args)
        expected_qnas += n_target
        per_split[item["split_file"]] = per_split.get(item["split_file"], 0) + 1
        line = json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }) + "\n"
        encoded = line.encode("utf-8")

        # Roll over to a new shard on count or size limits.
        if (cur_f is None or cur_count >= args.max_requests_per_batch
                or cur_bytes + len(encoded) > args.max_batch_file_bytes):
            _close_shard()
            shard_idx = len(shards)
            cur_path, cur_f = _open_shard(shard_idx)
            cur_count = 0
            cur_bytes = 0

        cur_f.write(line)
        cur_bytes += len(encoded)
        cur_count += 1

        meta_f.write(json.dumps({
            "custom_id": custom_id,
            "split_file": item["split_file"],
            "repo_id": item["repo_id"],
            "cross_repo_split": item["cross_repo_split"],
            "commit_index": item["commit_index"],
            "commit_sha": item["commit_sha"],
            "in_repo_split": item["in_repo_split"],
        }) + "\n")

        seq += 1
        total += 1

    _close_shard()
    meta_f.close()

    mode = (f"adaptive {args.qnas_min}-{args.qnas_max}"
            if getattr(args, "adaptive_qnas", False)
            else f"fixed {args.qnas_per_commit}")
    avg = (expected_qnas / total) if total else 0
    print(f"[submit] prepared {total} requests across {len(shards)} shard(s) "
          f"in {req_dir}")
    print(f"[submit] qnas/commit={mode}; per-split commits={per_split}")

    base_manifest = {
        "model": args.model,
        "qnas_per_commit": args.qnas_per_commit,
        "adaptive_qnas": getattr(args, "adaptive_qnas", False),
        "qnas_min": getattr(args, "qnas_min", args.qnas_per_commit),
        "qnas_max": getattr(args, "qnas_max", args.qnas_per_commit),
        "parse_cap": _parse_cap(args),
        "splits": args.splits,
        "n_requests": total,
        "expected_qnas": expected_qnas,
        "per_split_commits": per_split,
        "shards": shards,
    }

    if args.dry_run:
        print(f"[dry-run] would create {len(shards)} batch(es); "
              f"~{expected_qnas} QnA pairs (avg {avg:.1f}/commit, "
              f"model={args.model}). No API calls made.")
        manifest = {**base_manifest, "dry_run": True, "batches": []}
        _save_manifest(work_dir, manifest)
        return

    client = get_client()
    batches: List[Dict[str, Any]] = []
    for sh in shards:
        path = Path(sh["path"])
        print(f"[submit] uploading {path.name} ({sh['n_requests']} requests)...")
        with path.open("rb") as fh:
            up = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "purpose": "code2lora-qa",
                "shard": str(sh["shard"]),
                "model": args.model,
            },
        )
        print(f"[submit]   -> batch {batch.id} (status={batch.status})")
        batches.append({
            "shard": sh["shard"],
            "input_file_id": up.id,
            "batch_id": batch.id,
            "n_requests": sh["n_requests"],
        })

    manifest = {**base_manifest, "dry_run": False, "batches": batches}
    _save_manifest(work_dir, manifest)
    print(f"[submit] created {len(batches)} batch(es). Manifest: "
          f"{_manifest_path(work_dir)}")
    print("[submit] next: poll for completion, then fetch.")


# --------------------------------------------------------------------------- #
# poll
# --------------------------------------------------------------------------- #

_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def cmd_poll(args) -> None:
    work_dir = Path(args.work_dir)
    manifest = _load_manifest(work_dir)
    if manifest.get("dry_run"):
        raise SystemExit("Manifest is from a --dry-run submit; nothing to poll.")
    client = get_client()

    while True:
        all_terminal = True
        updated = False
        for b in manifest["batches"]:
            batch = client.batches.retrieve(b["batch_id"])
            b["status"] = batch.status
            rc = getattr(batch, "request_counts", None)
            b["output_file_id"] = getattr(batch, "output_file_id", None)
            b["error_file_id"] = getattr(batch, "error_file_id", None)
            counts = ""
            if rc is not None:
                counts = (f" ({rc.completed}/{rc.total} done, "
                          f"{rc.failed} failed)")
            print(f"[poll] batch {b['batch_id']}: {batch.status}{counts}")
            if batch.status not in _TERMINAL:
                all_terminal = False
            updated = True
        if updated:
            _save_manifest(work_dir, manifest)
        if all_terminal or not args.wait:
            break
        print(f"[poll] waiting {args.interval}s ...")
        time.sleep(args.interval)

    done = sum(1 for b in manifest["batches"] if b.get("status") == "completed")
    print(f"[poll] {done}/{len(manifest['batches'])} batch(es) completed.")


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

QNA_SCHEMA = [
    ("repo_id", "string"),
    ("cross_repo_split", "string"),
    ("commit_index", "int32"),
    ("commit_sha", "string"),
    ("in_repo_split", "string"),
    ("task", "string"),
    ("question_index", "int32"),
    ("category", "string"),
    ("prefix", "large_string"),
    ("target", "large_string"),
]


def _load_meta(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    with _meta_path(work_dir).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            meta[rec["custom_id"]] = rec
    return meta


def _parse_content(content: str, qnas_per_commit: int) -> List[Dict[str, str]]:
    """Parse the model's JSON content into a clean list of QnA dicts."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    raw = obj.get("qnas") if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for i, q in enumerate(raw[:qnas_per_commit]):
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        answer = str(q.get("answer", "")).strip()
        if not question or not answer:
            continue
        category = str(q.get("category", "")).strip()
        if category not in CATEGORIES:
            category = CATEGORIES[i % len(CATEGORIES)]
        out.append({"category": category, "question": question, "answer": answer})
    return out


def _qnas_to_rows(ident: Dict[str, Any],
                  qnas: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Expand one commit's parsed QnAs into output parquet rows."""
    rows: List[Dict[str, Any]] = []
    for qi, q in enumerate(qnas):
        rows.append({
            "repo_id": ident["repo_id"],
            "cross_repo_split": ident["cross_repo_split"],
            "commit_index": int(ident["commit_index"]),
            "commit_sha": ident["commit_sha"],
            "in_repo_split": ident["in_repo_split"],
            "task": TASK_ID,
            "question_index": qi,
            "category": q["category"],
            "prefix": q["question"],
            "target": q["answer"],
        })
    return rows


def _write_split_parquet(out_dir: Path, split: str, rows: List[Dict[str, Any]]) -> None:
    _require_pyarrow()
    out_dir.mkdir(parents=True, exist_ok=True)
    type_map = {
        "string": pa.string(),
        "large_string": pa.large_string(),
        "int32": pa.int32(),
    }
    arrays, names = [], []
    for name, t in QNA_SCHEMA:
        arrays.append(pa.array([r[name] for r in rows], type=type_map[t]))
        names.append(name)
    table = pa.Table.from_arrays(arrays, names=names)
    path = out_dir / f"{split}.parquet"
    pq.write_table(table, path, compression="snappy")
    print(f"[fetch] wrote {len(rows)} rows -> {path}")


def _write_review_markdown(path: Path, rows_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    """Write a human-readable Markdown file grouped by repo + commit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_commits = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("# Generated QA dataset review\n\n")
        for split in sorted(rows_by_split):
            rows = rows_by_split[split]
            # group by (repo_id, commit_index, commit_sha), preserving order
            groups: Dict[tuple, List[Dict[str, Any]]] = {}
            for r in rows:
                key = (r["repo_id"], r["commit_index"], r["commit_sha"],
                       r["in_repo_split"])
                groups.setdefault(key, []).append(r)
            f.write(f"## split: {split}  ({len(groups)} commits)\n\n")
            for (repo_id, cidx, sha, irs), qnas in groups.items():
                n_commits += 1
                f.write(f"### {repo_id} @ commit #{cidx} `{sha[:10]}` "
                        f"(in_repo_split={irs})\n\n")
                for q in sorted(qnas, key=lambda x: x["question_index"]):
                    f.write(f"- **[{q['category']}] Q:** {q['prefix']}\n")
                    f.write(f"  **A:** {q['target']}\n\n")
    print(f"[fetch] wrote review markdown ({n_commits} commits) -> {path}")


def cmd_fetch(args) -> None:
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(work_dir)
    if manifest.get("dry_run"):
        raise SystemExit("Manifest is from a --dry-run submit; nothing to fetch.")
    qnas_per_commit = manifest.get("parse_cap",
                                   manifest.get("qnas_per_commit",
                                                args.qnas_per_commit))
    client = get_client()
    meta = _load_meta(work_dir)

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    fail_f = (work_dir / "failures.jsonl").open("w", encoding="utf-8")
    n_ok = n_fail = n_pairs = 0

    for b in manifest["batches"]:
        batch = client.batches.retrieve(b["batch_id"])
        if batch.status != "completed":
            print(f"[fetch] skip batch {b['batch_id']} (status={batch.status})")
            continue
        out_id = getattr(batch, "output_file_id", None)
        if not out_id:
            print(f"[fetch] batch {b['batch_id']} has no output file; skip")
            continue
        text = client.files.content(out_id).text
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            ident = meta.get(cid)
            resp = rec.get("response") or {}
            err = rec.get("error")
            if ident is None or err is not None or resp.get("status_code") != 200:
                n_fail += 1
                fail_f.write(json.dumps({"custom_id": cid, "error": err,
                                         "status": resp.get("status_code")}) + "\n")
                continue
            try:
                content = resp["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                n_fail += 1
                fail_f.write(json.dumps({"custom_id": cid,
                                         "error": "no content"}) + "\n")
                continue
            qnas = _parse_content(content, qnas_per_commit)
            if not qnas:
                n_fail += 1
                fail_f.write(json.dumps({"custom_id": cid,
                                         "error": "empty/unparseable",
                                         "content": content[:500]}) + "\n")
                continue
            split = ident["split_file"]
            bucket = rows_by_split.setdefault(split, [])
            for qi, q in enumerate(qnas):
                bucket.append({
                    "repo_id": ident["repo_id"],
                    "cross_repo_split": ident["cross_repo_split"],
                    "commit_index": int(ident["commit_index"]),
                    "commit_sha": ident["commit_sha"],
                    "in_repo_split": ident["in_repo_split"],
                    "task": TASK_ID,
                    "question_index": qi,
                    "category": q["category"],
                    "prefix": q["question"],
                    "target": q["answer"],
                })
                n_pairs += 1
            n_ok += 1
    fail_f.close()

    for split, rows in rows_by_split.items():
        _write_split_parquet(out_dir, split, rows)

    review_out = getattr(args, "review_out", None)
    if review_out:
        _write_review_markdown(Path(review_out), rows_by_split)

    print(f"[fetch] commits ok={n_ok}, failed={n_fail}, qna pairs={n_pairs}")
    if n_fail:
        print(f"[fetch] failure details: {work_dir / 'failures.jsonl'}")


# --------------------------------------------------------------------------- #
# sync (no batch: synchronous calls with a concurrency pool)
# --------------------------------------------------------------------------- #

def cmd_sync(args) -> None:
    """Generate QnAs synchronously (instant results, no batch discount).

    Intended for quick verification on one/few repos before committing to a
    full batch run. Uses a thread pool so a small repo finishes in seconds.
    """
    out_dir = Path(args.out_dir)
    client = get_client()
    repos_filter = set(args.repos) if args.repos else None
    items = list(iter_commits(
        Path(args.commits_dir), args.splits,
        skip_empty_diff=not args.keep_empty_diff,
        max_diff_chars=args.max_diff_chars,
        repos_filter=repos_filter,
        limit=args.limit,
        max_commits_per_repo=args.max_commits_per_repo,
        stratify_splits=args.stratify_splits,
        min_eval_commits=args.min_eval_commits,
    ))
    total = len(items)
    print(f"[sync] {total} commits -> {args.concurrency} workers, model={args.model}")
    if total == 0:
        print("[sync] nothing to do.")
        return

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    counters = {"ok": 0, "fail": 0, "pairs": 0}
    lock = threading.Lock()

    parse_cap = _parse_cap(args)

    def work(item: Dict[str, Any]):
        body, _ = build_request_body(item, args)
        last = ("error", "no attempt")
        for attempt in range(args.max_retries + 1):
            try:
                resp = client.chat.completions.create(**body)
                content = resp.choices[0].message.content
            except Exception as e:  # network / API error (often transient)
                last = ("error", str(e)[:300])
                time.sleep(1.5 * (attempt + 1))
                continue
            qnas = _parse_content(content, parse_cap)
            if qnas:
                return ("ok", item, qnas)
            # Empty/unparseable content is usually transient; retry.
            last = ("empty", (content or "")[:300])
            time.sleep(1.0 * (attempt + 1))
        return (last[0], item, last[1])

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, it) for it in items]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, item, payload = fut.result()
            with lock:
                if status == "ok":
                    rows = _qnas_to_rows(item, payload)
                    rows_by_split.setdefault(item["split_file"], []).extend(rows)
                    counters["ok"] += 1
                    counters["pairs"] += len(rows)
                else:
                    counters["fail"] += 1
                    failures.append({
                        "repo_id": item["repo_id"],
                        "commit_sha": item["commit_sha"],
                        "reason": status,
                        "detail": payload,
                    })
            if i % 20 == 0 or i == total:
                print(f"[sync] {i}/{total} done "
                      f"(ok={counters['ok']}, fail={counters['fail']})")

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda r: (r["repo_id"], r["commit_index"],
                                 r["question_index"]))
        _write_split_parquet(out_dir, split, rows)

    review_out = getattr(args, "review_out", None)
    if review_out:
        _write_review_markdown(Path(review_out), rows_by_split)

    if failures:
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / "failures.jsonl"
        with fp.open("w", encoding="utf-8") as fh:
            for rec in failures:
                fh.write(json.dumps(rec) + "\n")
        print(f"[sync] failure details: {fp}")

    print(f"[sync] commits ok={counters['ok']}, failed={counters['fail']}, "
          f"qna pairs={counters['pairs']}")


# --------------------------------------------------------------------------- #
# run (submit -> poll --wait -> fetch)
# --------------------------------------------------------------------------- #

def cmd_run(args) -> None:
    cmd_submit(args)
    if args.dry_run:
        return
    args.wait = True
    cmd_poll(args)
    cmd_fetch(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _add_common_submit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--commits-dir", required=True,
                   help="dir with {split}.parquet GRU v2 commit files")
    p.add_argument("--splits", nargs="+",
                   default=["train", "cr_val", "cr_test"],
                   help="split files to process (default: train cr_val cr_test)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--qnas-per-commit", type=int, default=DEFAULT_QNAS_PER_COMMIT,
                   help="fixed number of QnAs per commit (when not adaptive)")
    p.add_argument("--adaptive-qnas", action="store_true",
                   help="scale QnAs/commit with diff size between --qnas-min and "
                        "--qnas-max (high-value categories first)")
    p.add_argument("--qnas-min", type=int, default=3,
                   help="min QnAs/commit for tiny diffs when --adaptive-qnas")
    p.add_argument("--qnas-max", type=int, default=10,
                   help="max QnAs/commit for large diffs when --adaptive-qnas")
    p.add_argument("--max-diff-chars", type=int, default=24_000,
                   help="coarse char pre-cap on each diff (0 = no limit)")
    p.add_argument("--max-input-tokens", type=int, default=4_000,
                   help="hard cap on total input tokens per request; the diff is "
                        "token-trimmed to fit (0 = no token cap)")
    p.add_argument("--max-completion-tokens", type=int, default=6_000)
    p.add_argument("--reasoning-effort", default="low",
                   help="gpt-5 family reasoning effort: none|low|medium|high|xhigh "
                        "('none' omits the param entirely)")
    p.add_argument("--keep-empty-diff", action="store_true",
                   help="include commits whose production_code_diff is empty")
    p.add_argument("--repos", nargs="*", default=None,
                   help="optional repo_id allow-list (e.g. MrPowers/chispa)")
    p.add_argument("--max-commits-per-repo", type=int, default=0,
                   help="cap commits per repo, sampled evenly across the repo's "
                        "timeline (0 = no cap). ~15 yields ~70k QnAs at 10/commit.")
    p.add_argument("--stratify-splits", action="store_true",
                   help="when capping, reserve coverage for val/test in-repo "
                        "splits so ir_val/ir_test are not starved per repo")
    p.add_argument("--min-eval-commits", type=int, default=2,
                   help="min commits kept from each of val/test per repo when "
                        "--stratify-splits and the repo has them")
    p.add_argument("--limit", type=int, default=0,
                   help="global cap on number of commits (0 = all); for testing")
    p.add_argument("--max-requests-per-batch", type=int,
                   default=MAX_REQUESTS_PER_BATCH)
    p.add_argument("--max-batch-file-bytes", type=int,
                   default=MAX_BATCH_FILE_BYTES)
    p.add_argument("--dry-run", action="store_true",
                   help="build shards + manifest only; no uploads/API calls")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit", help="build + upload requests, create batches")
    p_sub.add_argument("--work-dir", required=True)
    _add_common_submit_args(p_sub)
    p_sub.set_defaults(func=cmd_submit)

    p_poll = sub.add_parser("poll", help="check / wait on batch status")
    p_poll.add_argument("--work-dir", required=True)
    p_poll.add_argument("--wait", action="store_true",
                        help="loop until all batches reach a terminal state")
    p_poll.add_argument("--interval", type=int, default=60,
                        help="seconds between polls when --wait")
    p_poll.set_defaults(func=cmd_poll)

    p_fetch = sub.add_parser("fetch", help="download outputs -> qna/{split}.parquet")
    p_fetch.add_argument("--work-dir", required=True)
    p_fetch.add_argument("--out-dir", required=True)
    p_fetch.add_argument("--qnas-per-commit", type=int,
                         default=DEFAULT_QNAS_PER_COMMIT)
    p_fetch.add_argument("--review-out", default=None,
                         help="also write a human-readable Markdown file of the "
                              "generated QnAs for manual review")
    p_fetch.set_defaults(func=cmd_fetch)

    p_sync = sub.add_parser(
        "sync", help="NO batch: synchronous calls, instant results (no discount)")
    p_sync.add_argument("--out-dir", required=True)
    p_sync.add_argument("--review-out", default=None,
                        help="also write a human-readable Markdown file for review")
    p_sync.add_argument("--concurrency", type=int, default=8,
                        help="number of parallel synchronous requests")
    p_sync.add_argument("--max-retries", type=int, default=2,
                        help="retries per commit on API error / empty response")
    _add_common_submit_args(p_sync)
    p_sync.set_defaults(func=cmd_sync)

    p_run = sub.add_parser("run", help="submit -> poll --wait -> fetch")
    p_run.add_argument("--work-dir", required=True)
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--interval", type=int, default=60)
    p_run.add_argument("--review-out", default=None,
                       help="also write a human-readable Markdown file for review")
    _add_common_submit_args(p_run)
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
