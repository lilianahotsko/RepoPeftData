#!/usr/bin/env python3
"""
Build a commit-level Parquet dataset for Code2LoRA training.

Per repository, keep ONLY commits that introduce new or changed assertions
in test files. For each such kept commit:

  * ``production_code_diff`` = unified diff of non-test source files vs the
    previous kept commit (empty-tree base for the first kept commit).
  * one QnA pair per new/changed assertion, whose prefix is the FULL content
    of the test file at that commit up to the assertion's cut line, and whose
    target is the assertion's RHS (same extraction as in
    ``create_qnas.py``).

Kept commits are then split chronologically per repo 80/10/10 into
``in_repo_split`` ∈ {train, val, test}.

Output: two Parquet files joined by (repo_id, commit_index):

    $OUT_DIR/commits.parquet
    $OUT_DIR/qna_pairs.parquet

Per-repo shards are written first under ``$OUT_DIR/shards/`` so the job is
restartable and embarrassingly parallel.

Example:
    python create_dataset/build_commit_parquet_db.py \\
        --splits-dir  $SCRATCH/REPO_DATASET \\
        --repos-root  $SCRATCH/REPO_DATASET/repositories \\
        --out-dir     $SCRATCH/REPO_DATASET/commit_parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_commit_assertion_db import (  # noqa: E402
    EMPTY_TREE_SHA,
    SKIP_DIRS,
    _is_test_path,
    filter_diff,
    get_diff,
)
from create_qnas import (  # noqa: E402
    SELF_ASSERT_METHODS,
    AssertionPair,
    _parse_assert_underscore,
    _parse_plain_assert,
    _parse_pytest_approx,
    _parse_pytest_raises,
    _parse_self_assert,
    _find_enclosing_test,
    flatten_to_oneliner,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QNA_FILENAME = "QNA_HYPERNET.json"
REPO_METADATA_FILENAME = "REPO_METADATA.json"

CROSS_REPO_SPLIT_FILES = {
    "train": "train.json",
    "cr_val": "cr_val.json",
    "cr_test": "cr_test.json",
}

SHARDS_SUBDIR = "shards"
DONE_FILENAME = "_done.jsonl"


# ---------------------------------------------------------------------------
# Path filtering for "is this a test file"
# ---------------------------------------------------------------------------

def _is_python_test_file(path: str) -> bool:
    """Heuristic matching ``2_separate_tests.py``: any path component with 'test'
    or a .py file importing pytest/unittest / defining test_* functions.

    Here we use the lighter path-based rule combined with a .py-only filter,
    since the REPO_METADATA.test_files list already captures the
    content-based decision at HEAD; we additionally accept any .py file whose
    path contains "test" (covers test files added in later commits that are
    not in REPO_METADATA).
    """
    p = Path(path)
    if p.suffix.lower() != ".py":
        return False
    if any(part in SKIP_DIRS for part in p.parts):
        return False
    return _is_test_path(path)


# ---------------------------------------------------------------------------
# Fast git helpers
#
# The hot path used to spawn ``git diff --name-only`` + N × ``git show`` per
# commit, which is dominated by ``fork+exec`` overhead for repos with 100s of
# commits. We replace that with:
#
#   1. ONE ``git log --raw`` streaming call that yields every commit's
#      changed-file list together with the new blob SHA per file.
#   2. A persistent ``git cat-file --batch`` subprocess fed blob SHAs on stdin,
#      so reading file contents costs one write+read, not a full fork+exec.
#
# Semantics are preserved exactly — we still see every commit, every changed
# path, and every file's content at that commit. Renames (status ``R``) are
# handled by migrating the previous-assertion cache from the old path to the
# new one, matching git's default rename-detection behavior.
# ---------------------------------------------------------------------------

# --- per-commit changed-files stream -----------------------------------------

_NULL_SHA = "0000000000000000000000000000000000000000"


@dataclass
class RawFileChange:
    """One entry from ``git log --raw``.

    * ``status`` is a single-letter status (``A``/``M``/``D``/``R``/``C``/``T``)
      or the multi-letter form (``R100`` etc.).
    * ``new_blob`` is the new blob SHA (all zeros if the file was deleted).
    * ``path`` is the post-change path. ``old_path`` is set only for renames.
    """
    status: str
    new_blob: str
    path: str
    old_path: Optional[str] = None


@dataclass
class RawCommit:
    sha: str
    parent_sha: Optional[str]
    timestamp: str
    changes: List[RawFileChange]


def stream_commit_changes(repo_dir: Path) -> Iterator[RawCommit]:
    """Yield one ``RawCommit`` per first-parent commit, oldest-first.

    Uses a single streaming ``git log --first-parent --reverse --raw``
    subprocess so we don't pay fork+exec per commit.
    """
    proc = subprocess.Popen(
        [
            "git", "-C", str(repo_dir),
            "log", "--first-parent", "--reverse",
            "--raw", "--no-abbrev",
            "-M",  # detect renames (same as git's default for log)
            "--format=COMMITHDR %H %P %aI",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None

    cur: Optional[RawCommit] = None
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue

            if line.startswith("COMMITHDR "):
                if cur is not None:
                    yield cur
                # Format: ``COMMITHDR <sha> [<parent>...] <timestamp>``
                # where the parent list may be empty (root commit) or contain
                # multiple entries (merge). We parse by peeling off the SHA
                # from the left and the timestamp from the right.
                tokens = line.split()
                if len(tokens) < 3:
                    cur = None
                    continue
                sha = tokens[1]
                timestamp = tokens[-1]
                parents = tokens[2:-1]
                parent_sha = parents[0] if parents else None
                cur = RawCommit(
                    sha=sha,
                    parent_sha=parent_sha,
                    timestamp=timestamp,
                    changes=[],
                )
                continue

            if cur is None or not line.startswith(":"):
                continue

            # Format:
            #   :<mode_a> <mode_b> <sha_a> <sha_b> <status>\t<path>
            # For renames/copies:
            #   :<mode_a> <mode_b> <sha_a> <sha_b> R<score>\t<old_path>\t<new_path>
            try:
                meta, _, tail = line.partition("\t")
                if not tail:
                    continue
                fields = meta.split()
                if len(fields) < 5:
                    continue
                new_blob = fields[3].lstrip(":")
                status = fields[4]
                parts = tail.split("\t")
                if status.startswith("R") or status.startswith("C"):
                    old_path = parts[0]
                    new_path = parts[1] if len(parts) > 1 else parts[0]
                    cur.changes.append(RawFileChange(
                        status=status,
                        new_blob=new_blob,
                        path=new_path,
                        old_path=old_path,
                    ))
                else:
                    cur.changes.append(RawFileChange(
                        status=status,
                        new_blob=new_blob,
                        path=parts[0],
                    ))
            except Exception:  # noqa: BLE001
                continue

        if cur is not None:
            yield cur
    finally:
        try:
            proc.stdout.close()
        finally:
            proc.wait(timeout=5)


# --- persistent cat-file --batch ---------------------------------------------

class GitCatFileBatch:
    """Wrapper around a persistent ``git cat-file --batch`` subprocess.

    Reading a blob costs one write + one header read + one bounded read from
    stdout — no fork+exec per file.
    """

    def __init__(self, repo_dir: Path):
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo_dir), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def read_blob(self, blob_sha: str) -> Optional[bytes]:
        """Return blob bytes, or None if missing / wrong type."""
        if not blob_sha or blob_sha == _NULL_SHA:
            return None
        p = self._proc
        assert p.stdin is not None and p.stdout is not None
        try:
            p.stdin.write((blob_sha + "\n").encode("ascii"))
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            return None

        header = p.stdout.readline()
        if not header:
            return None
        parts = header.decode("utf-8", errors="replace").strip().split()
        # <sha> <type> <size>  OR  <sha> missing
        if len(parts) < 3 or parts[1] != "blob":
            # missing / tag / tree / commit — consume no body
            return None
        try:
            size = int(parts[2])
        except ValueError:
            return None
        body = b""
        remaining = size
        while remaining > 0:
            chunk = p.stdout.read(remaining)
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        # trailing LF
        try:
            p.stdout.read(1)
        except Exception:  # noqa: BLE001
            pass
        return body

    def read_blob_text(self, blob_sha: str) -> Optional[str]:
        data = self.read_blob(blob_sha)
        if data is None:
            return None
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "GitCatFileBatch":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Assertion extraction at a given commit (reuses create_qnas.py parsers)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedAssertion:
    """Full-prefix assertion extracted from a file at a given commit."""
    assertion_type: str
    prefix: str
    target: str
    lineno: int
    col_offset: int
    test_function: str


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_id(s: str) -> str:
    """Collapse whitespace so formatting changes don't create spurious
    'new assertion' events.
    """
    return _WHITESPACE_RE.sub(" ", (s or "").strip())


def _full_prefix_for(source_lines: List[str], pair: AssertionPair) -> str:
    """Full test-file content up to (but not including) the assertion cut,
    plus the in-line prefix emitted by the parser (e.g. ``assert foo == ``).
    """
    lineno = pair.lineno
    col = pair.col_offset

    if lineno < 1:
        return pair.prefix

    pre = source_lines[: lineno - 1]
    last_line = source_lines[lineno - 1] if lineno - 1 < len(source_lines) else ""
    if col < 0:
        col = 0
    last_line_prefix = last_line[: col] if col <= len(last_line) else last_line

    return "".join(pre) + last_line_prefix + pair.prefix


def extract_assertions_from_source(source: str) -> List[ExtractedAssertion]:
    """Run the same AST walk as ``create_qnas.extract_assertion_pairs`` but
    build a FULL-file prefix (no 300-line cap) and return a flat list.
    """
    if not source:
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    lines = source.splitlines(keepends=True)
    out: List[ExtractedAssertion] = []

    for node in ast.walk(tree):
        pair: Optional[AssertionPair] = None

        if isinstance(node, ast.Assert):
            pair = _parse_plain_assert(source, node, lines)

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                val = func.value
                val_id = getattr(val, "id", None) if isinstance(val, ast.Name) else None
                if val_id == "self" and func.attr in SELF_ASSERT_METHODS:
                    pair = _parse_self_assert(source, node, lines)
                elif val_id == "pytest":
                    if func.attr == "raises":
                        pair = _parse_pytest_raises(source, node, lines)
                    elif func.attr == "approx":
                        pair = _parse_pytest_approx(source, node, lines)
                    else:
                        pair = _parse_assert_underscore(source, node, lines)
                else:
                    pair = _parse_assert_underscore(source, node, lines)
            elif isinstance(func, ast.Name) and func.id.startswith("assert_"):
                pair = _parse_assert_underscore(source, node, lines)

        if pair is None or not pair.target.strip():
            continue

        target = pair.target
        if target.lstrip().startswith(","):
            continue
        if "\n" in target:
            target = flatten_to_oneliner(target)
        if not target.strip():
            continue

        full_prefix = _full_prefix_for(lines, pair)
        test_node = _find_enclosing_test(tree, node)
        test_name = test_node.name if test_node else ""

        out.append(ExtractedAssertion(
            assertion_type=pair.assertion_type,
            prefix=full_prefix,
            target=target,
            lineno=pair.lineno,
            col_offset=pair.col_offset,
            test_function=test_name,
        ))
    return out


# ---------------------------------------------------------------------------
# Per-repo pipeline
# ---------------------------------------------------------------------------

def _repo_dir_for(repos_root: Path, repo_id: str) -> Path:
    parts = repo_id.split("/")
    if len(parts) == 2:
        return repos_root / parts[0] / parts[1]
    return repos_root / repo_id


def _load_test_files_from_metadata(repo_dir: Path) -> Set[str]:
    meta_path = repo_dir / REPO_METADATA_FILENAME
    if not meta_path.exists():
        return set()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    files = meta.get("test_files") or []
    return {str(f) for f in files if isinstance(f, str)}


def _chronological_splits(n_kept: int) -> List[str]:
    """80/10/10 chronological split of ``n_kept`` kept commits."""
    if n_kept <= 0:
        return []
    if n_kept < 3:
        return ["train"] * n_kept
    n_train = int(n_kept * 0.8)
    n_val = int(n_kept * 0.1)
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n_kept - n_train - n_val
    if n_test < 1:
        n_val = max(1, n_val - 1) if n_val > 1 else n_val
        n_test = n_kept - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n_kept - n_train - n_val
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


@dataclass
class CommitRow:
    repo_id: str
    cross_repo_split: str
    commit_index: int
    commit_sha: str
    commit_timestamp: str
    in_repo_split: str
    production_code_diff: str
    n_new_assertions: int


@dataclass
class QnaRow:
    repo_id: str
    cross_repo_split: str
    commit_index: int
    commit_sha: str
    in_repo_split: str
    test_file: str
    lineno: int
    col_offset: int
    assertion_type: str
    test_function: str
    prefix: str
    target: str


def process_repo(
    repo_id: str,
    cross_repo_split: str,
    repo_dir: Path,
    diff_timeout: int = 600,
    verbose: bool = False,
) -> Tuple[List[CommitRow], List[QnaRow], Dict[str, Any]]:
    """Process a single repository.

    Returns (commit_rows, qna_rows, stats).
    """
    stats: Dict[str, Any] = {
        "repo_id": repo_id,
        "total_commits": 0,
        "kept_commits": 0,
        "n_assertions": 0,
    }

    metadata_test_files = _load_test_files_from_metadata(repo_dir)

    # ---- Phase 1: streaming walk, detect kept commits ----------------------
    # One `git log --raw` produces every commit's changed-files + new blob SHAs.
    # Deletions ("D") are first-class (status D, null new_blob). Renames ("R")
    # migrate the previous-assertion cache to the new path so rename-only
    # commits don't look like brand-new assertions.
    prev_assert_keys: Dict[str, Set[Tuple[str, str, str]]] = {}
    kept: List[Dict[str, Any]] = []
    total_commits = 0

    commit_stream = stream_commit_changes(repo_dir)
    if verbose:
        commit_stream = tqdm(commit_stream, desc=f"  {repo_id} scan", leave=False)

    with GitCatFileBatch(repo_dir) as cat:
        for ci, raw in enumerate(commit_stream):
            total_commits += 1
            sha = raw.sha
            parent_sha = raw.parent_sha
            timestamp = raw.timestamp

            # Apply rename / delete bookkeeping to the cache FIRST so that a
            # rename-only commit stays "no new assertions".
            candidate_changes: List[RawFileChange] = []
            for ch in raw.changes:
                # Rename: migrate cache entry old_path -> new_path.
                if ch.old_path and ch.path != ch.old_path:
                    if ch.old_path in prev_assert_keys:
                        prev_assert_keys[ch.path] = prev_assert_keys.pop(
                            ch.old_path
                        )

                if ch.status.startswith("D"):
                    prev_assert_keys.pop(ch.path, None)
                    continue

                if ch.path in metadata_test_files or _is_python_test_file(ch.path):
                    candidate_changes.append(ch)

            if not candidate_changes:
                continue

            per_file_new: List[Tuple[str, List[ExtractedAssertion]]] = []
            for ch in candidate_changes:
                content = cat.read_blob_text(ch.new_blob)
                if content is None:
                    prev_assert_keys.pop(ch.path, None)
                    continue

                extracted = extract_assertions_from_source(content)
                current_keys: Set[Tuple[str, str, str]] = set()
                for ex in extracted:
                    current_keys.add((
                        ex.assertion_type,
                        _normalize_for_id(ex.prefix),
                        _normalize_for_id(ex.target),
                    ))

                prev_keys = prev_assert_keys.get(ch.path, set())
                new_keys = current_keys - prev_keys

                if new_keys:
                    new_extracted = [
                        ex for ex in extracted
                        if (ex.assertion_type,
                            _normalize_for_id(ex.prefix),
                            _normalize_for_id(ex.target)) in new_keys
                    ]
                    if new_extracted:
                        per_file_new.append((ch.path, new_extracted))

                prev_assert_keys[ch.path] = current_keys

            if not per_file_new:
                continue

            kept.append({
                "original_idx": ci,
                "sha": sha,
                "parent_sha": parent_sha,
                "timestamp": timestamp,
                "per_file_new": per_file_new,
            })

    stats["total_commits"] = total_commits
    stats["kept_commits"] = len(kept)
    if not kept:
        return [], [], stats

    # ---- Phase 2: build diffs between kept commits + chronological split
    splits = _chronological_splits(len(kept))
    commit_rows: List[CommitRow] = []
    qna_rows: List[QnaRow] = []

    prev_kept_sha: Optional[str] = None
    for ki, kc in enumerate(kept):
        sha = kc["sha"]
        base = prev_kept_sha  # None on first iteration -> EMPTY_TREE_SHA inside get_diff
        raw_diff = get_diff(repo_dir, base, sha, timeout=diff_timeout)
        filtered = filter_diff(raw_diff) if raw_diff else ""

        n_new = sum(len(e) for _, e in kc["per_file_new"])
        in_split = splits[ki]

        commit_rows.append(CommitRow(
            repo_id=repo_id,
            cross_repo_split=cross_repo_split,
            commit_index=ki,
            commit_sha=sha,
            commit_timestamp=kc.get("timestamp", ""),
            in_repo_split=in_split,
            production_code_diff=filtered,
            n_new_assertions=n_new,
        ))

        for tfile, extracted in kc["per_file_new"]:
            for ex in extracted:
                qna_rows.append(QnaRow(
                    repo_id=repo_id,
                    cross_repo_split=cross_repo_split,
                    commit_index=ki,
                    commit_sha=sha,
                    in_repo_split=in_split,
                    test_file=tfile,
                    lineno=ex.lineno,
                    col_offset=ex.col_offset,
                    assertion_type=ex.assertion_type,
                    test_function=ex.test_function,
                    prefix=ex.prefix,
                    target=ex.target,
                ))

        prev_kept_sha = sha

    stats["n_assertions"] = len(qna_rows)
    return commit_rows, qna_rows, stats


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

COMMIT_SCHEMA = [
    ("repo_id", "string"),
    ("cross_repo_split", "string"),
    ("commit_index", "int32"),
    ("commit_sha", "string"),
    ("commit_timestamp", "string"),
    ("in_repo_split", "string"),
    ("production_code_diff", "large_string"),
    ("n_new_assertions", "int32"),
]

QNA_SCHEMA = [
    ("repo_id", "string"),
    ("cross_repo_split", "string"),
    ("commit_index", "int32"),
    ("commit_sha", "string"),
    ("in_repo_split", "string"),
    ("test_file", "string"),
    ("lineno", "int32"),
    ("col_offset", "int32"),
    ("assertion_type", "string"),
    ("test_function", "string"),
    ("prefix", "large_string"),
    ("target", "string"),
]


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise SystemExit(
            "pyarrow is required. Install it with: pip install pyarrow"
        )


def _rows_to_table(rows: List, schema: List[Tuple[str, str]]) -> "pa.Table":
    """Convert a list of dataclasses to a pyarrow Table with an explicit schema."""
    columns: Dict[str, List[Any]] = {name: [] for name, _ in schema}
    for r in rows:
        d = r.__dict__
        for name, _ in schema:
            columns[name].append(d[name])

    type_map = {
        "string": pa.string(),
        "large_string": pa.large_string(),
        "int32": pa.int32(),
        "int64": pa.int64(),
    }
    arrays = []
    names = []
    for name, t in schema:
        arrays.append(pa.array(columns[name], type=type_map[t]))
        names.append(name)
    return pa.Table.from_arrays(arrays, names=names)


def write_shard(
    out_dir: Path,
    repo_id: str,
    commit_rows: List[CommitRow],
    qna_rows: List[QnaRow],
) -> Tuple[Path, Path]:
    """Write per-repo Parquet shards; returns the two paths written."""
    _require_pyarrow()
    safe = repo_id.replace("/", "__")
    shards = out_dir / SHARDS_SUBDIR
    shards.mkdir(parents=True, exist_ok=True)

    commits_path = shards / f"{safe}.commits.parquet"
    qna_path = shards / f"{safe}.qna.parquet"

    ctab = _rows_to_table(commit_rows, COMMIT_SCHEMA)
    pq.write_table(ctab, commits_path, compression="snappy")

    qtab = _rows_to_table(qna_rows, QNA_SCHEMA)
    pq.write_table(qtab, qna_path, compression="snappy")

    return commits_path, qna_path


def concat_shards(out_dir: Path) -> None:
    """Concatenate all per-repo shards into commits.parquet + qna_pairs.parquet."""
    _require_pyarrow()
    shards = out_dir / SHARDS_SUBDIR
    if not shards.is_dir():
        raise SystemExit(f"No shards dir at {shards}")

    commit_files = sorted(shards.glob("*.commits.parquet"))
    qna_files = sorted(shards.glob("*.qna.parquet"))

    print(f"Concatenating {len(commit_files)} commit shards...")
    if commit_files:
        commits_out = out_dir / "commits.parquet"
        writer: Optional[pq.ParquetWriter] = None
        try:
            for fp in tqdm(commit_files, desc="commits"):
                t = pq.read_table(fp)
                if writer is None:
                    writer = pq.ParquetWriter(
                        commits_out, t.schema, compression="snappy",
                    )
                writer.write_table(t)
        finally:
            if writer is not None:
                writer.close()
        print(f"  Wrote {commits_out}")

    print(f"Concatenating {len(qna_files)} qna shards...")
    if qna_files:
        qna_out = out_dir / "qna_pairs.parquet"
        writer = None
        try:
            for fp in tqdm(qna_files, desc="qna"):
                t = pq.read_table(fp)
                if writer is None:
                    writer = pq.ParquetWriter(
                        qna_out, t.schema, compression="snappy",
                    )
                writer.write_table(t)
        finally:
            if writer is not None:
                writer.close()
        print(f"  Wrote {qna_out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_repo_list(path: Path) -> Set[str]:
    """One repo_id per line; blank lines and ``#`` comments ignored."""
    text = path.read_text(encoding="utf-8")
    out: Set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def _load_cross_repo_assignments(splits_dir: Path) -> Dict[str, str]:
    """Map repo_id -> cross_repo_split using the existing train/cr_val/cr_test
    split JSONs. Repos present in multiple files are assigned train > cr_val > cr_test.
    """
    priority = ["train", "cr_val", "cr_test"]
    assignment: Dict[str, str] = {}
    for split_name in priority:
        fname = CROSS_REPO_SPLIT_FILES[split_name]
        fp = splits_dir / fname
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for repo_id in (data.get("repositories") or {}).keys():
            assignment.setdefault(repo_id, split_name)
    return assignment


def _iter_done_files(out_dir: Path) -> Iterator[Path]:
    """Yield every ``_done*.jsonl`` file. When running sharded jobs each
    worker writes to its own ``_done.<shard>.jsonl`` to avoid racing."""
    if (out_dir / DONE_FILENAME).exists():
        yield out_dir / DONE_FILENAME
    yield from sorted(out_dir.glob("_done.*.jsonl"))


def _load_done_set(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for dp in _iter_done_files(out_dir):
        try:
            text = dp.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("repo_id")
            if rid:
                done.add(rid)
    return done


def _append_done(
    out_dir: Path,
    repo_id: str,
    stats: Dict[str, Any],
    shard_index: Optional[int] = None,
) -> None:
    """Append one record. Writes go to a per-shard file when sharding is on."""
    if shard_index is not None:
        fname = f"_done.{shard_index:04d}.jsonl"
    else:
        fname = DONE_FILENAME
    dp = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"repo_id": repo_id, **stats}
    with dp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Multiprocessing worker — one Python process can drive N workers locally
# ---------------------------------------------------------------------------

def _mp_worker(task: Tuple[str, str, str, str, int, int]
               ) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    """Pool worker: process one repo and write its per-repo Parquet shard.

    Returns ``(repo_id, status, err_or_none, stats_or_none)`` where
    ``status`` is one of ``ok`` / ``no_kept`` / ``error`` / ``write_error``.
    We write Parquet inside the worker (no large payloads crossing the
    pickle boundary) and stream a one-line record into
    ``_done.wNNNN.jsonl`` keyed by worker id.
    """
    repo_id, cross_split, repo_dir_str, out_dir_str, diff_timeout, worker_id = task
    repo_dir = Path(repo_dir_str)
    out_dir = Path(out_dir_str)

    try:
        commit_rows, qna_rows, stats = process_repo(
            repo_id=repo_id,
            cross_repo_split=cross_split,
            repo_dir=repo_dir,
            diff_timeout=diff_timeout,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return repo_id, "error", repr(exc), None

    if not commit_rows:
        _append_done(out_dir, repo_id, stats, shard_index=worker_id)
        return repo_id, "no_kept", None, stats

    try:
        write_shard(out_dir, repo_id, commit_rows, qna_rows)
    except Exception as exc:  # noqa: BLE001
        return repo_id, "write_error", repr(exc), stats

    _append_done(out_dir, repo_id, stats, shard_index=worker_id)
    return repo_id, "ok", None, stats


def _run_pool(
    repos: List[str],
    assignment: Dict[str, str],
    repos_root: Path,
    out_dir: Path,
    n_workers: int,
    diff_timeout: int,
) -> Tuple[int, int, int]:
    """Drive ``n_workers`` subprocesses over ``repos``; returns the counters."""
    import multiprocessing as mp

    tasks: List[Tuple[str, str, str, str, int, int]] = []
    # Bucket repos round-robin across workers so each gets a mix of small and
    # large repos (Parquet shards already have unique names per repo).
    for i, repo_id in enumerate(repos):
        rdir = _repo_dir_for(repos_root, repo_id)
        if not rdir.is_dir() or not (rdir / ".git").exists():
            print(f"  skip (no git): {repo_id}", flush=True)
            continue
        worker_id = i % n_workers
        tasks.append((
            repo_id,
            assignment[repo_id],
            str(rdir),
            str(out_dir),
            diff_timeout,
            worker_id,
        ))

    ctx = mp.get_context("fork")  # much faster startup; no __main__ pickling
    n_ok = n_no_kept = n_errors = 0
    with ctx.Pool(processes=n_workers) as pool:
        it = pool.imap_unordered(_mp_worker, tasks, chunksize=1)
        for repo_id, status, err, _stats in tqdm(
            it, total=len(tasks), desc=f"repos (x{n_workers})",
        ):
            if status == "ok":
                n_ok += 1
            elif status == "no_kept":
                n_no_kept += 1
            elif status == "error":
                n_errors += 1
                print(f"  ERROR on {repo_id}: {err}", flush=True)
            elif status == "write_error":
                n_errors += 1
                print(f"  SHARD WRITE ERROR {repo_id}: {err}", flush=True)
    return n_ok, n_no_kept, n_errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build commit-level Parquet dataset (per-repo chronological split)",
    )
    default_root = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "REPO_DATASET",
    )
    ap.add_argument("--splits-dir", type=str, default=default_root)
    ap.add_argument("--repos-root", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Default: <splits-dir>/commit_parquet")
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument(
        "--repo-list",
        type=str,
        default=None,
        help="Text file: one repo_id per line (# comments allowed). Only these "
        "repos are processed (must appear in train/cr_val/cr_test splits).",
    )
    ap.add_argument("--resume", action="store_true",
                    help="Skip repos already recorded in _done.jsonl")
    ap.add_argument("--no-concat", action="store_true",
                    help="Skip the final shard concatenation step")
    ap.add_argument("--concat-only", action="store_true",
                    help="Only run the shard concatenation step (assumes shards already exist)")
    ap.add_argument("--diff-timeout", type=int, default=600)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--shard-index", type=int, default=None,
        help="Worker index in [0, num-shards). If set, only process repos "
             "whose index mod --num-shards equals --shard-index. Enables "
             "array-job parallelism without coordination.",
    )
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Total number of workers for sharded processing.")
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Number of local worker processes (single host). Orthogonal "
             "to --shard-index: e.g. --shard-index 3 --num-shards 4 "
             "--workers 8 means this process handles shard 3 of 4 (spread "
             "across all submissions) using 8 local subprocesses.",
    )
    args = ap.parse_args()

    if (args.shard_index is None) != (args.num_shards is None):
        raise SystemExit("--shard-index and --num-shards must be used together")
    if args.shard_index is not None and args.num_shards is not None:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(
                f"--shard-index must be in [0, {args.num_shards})"
            )

    _require_pyarrow()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    repos_root = Path(
        args.repos_root or (splits_dir / "repositories"),
    ).expanduser().resolve()
    out_dir = Path(
        args.out_dir or (splits_dir / "commit_parquet"),
    ).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.concat_only:
        concat_shards(out_dir)
        return

    assignment = _load_cross_repo_assignments(splits_dir)
    if not assignment:
        raise SystemExit(f"No cross-repo split JSONs found under {splits_dir}")

    repos_sorted = sorted(assignment.keys())
    if args.repo_list:
        repo_list_path = Path(args.repo_list).expanduser().resolve()
        if not repo_list_path.is_file():
            raise SystemExit(f"--repo-list not found: {repo_list_path}")
        allow = _load_repo_list(repo_list_path)
        assign_set = set(assignment.keys())
        unknown = sorted(allow - assign_set)
        if unknown:
            print(
                f"Warning: {len(unknown)} repo(s) in --repo-list are not in "
                f"train/cr_val/cr_test splits (skipped). First few: {unknown[:8]}",
                flush=True,
            )
        repos_sorted = sorted(allow & assign_set)
        if not repos_sorted:
            raise SystemExit("No repos left after --repo-list ∩ splits")
        print(f"Repo list filter: {len(repos_sorted)} repos from {repo_list_path}", flush=True)
    if args.limit_repos:
        repos_sorted = repos_sorted[: args.limit_repos]

    if args.shard_index is not None and args.num_shards is not None:
        # Deterministic round-robin shard assignment so every worker sees a
        # roughly equal distribution of big and small repos (vs a contiguous
        # slice which can be heavily skewed by sort order).
        repos_sorted = [
            r for i, r in enumerate(repos_sorted)
            if i % args.num_shards == args.shard_index
        ]

    done_set: Set[str] = _load_done_set(out_dir) if args.resume else set()

    print(f"Splits dir:  {splits_dir}")
    print(f"Repos root:  {repos_root}")
    print(f"Out dir:     {out_dir}")
    if args.shard_index is not None:
        print(f"Shard:       {args.shard_index} / {args.num_shards}")
    if args.workers and args.workers > 1:
        print(f"Workers:     {args.workers}")
    print(f"Repos:       {len(repos_sorted)} "
          f"(already done: {len(done_set & set(repos_sorted))})")

    # Drop already-done repos up front.
    todo = [r for r in repos_sorted if not (args.resume and r in done_set)]
    n_skipped = len(repos_sorted) - len(todo)

    if args.workers and args.workers > 1:
        n_ok, n_no_kept, n_errors = _run_pool(
            repos=todo,
            assignment=assignment,
            repos_root=repos_root,
            out_dir=out_dir,
            n_workers=args.workers,
            diff_timeout=args.diff_timeout,
        )
    else:
        n_ok = 0
        n_errors = 0
        n_no_kept = 0

        for repo_id in tqdm(todo, desc="repos"):
            cross_split = assignment[repo_id]
            rdir = _repo_dir_for(repos_root, repo_id)
            if not rdir.is_dir() or not (rdir / ".git").exists():
                print(f"  skip (no git): {repo_id}", flush=True)
                n_errors += 1
                continue

            try:
                commit_rows, qna_rows, stats = process_repo(
                    repo_id=repo_id,
                    cross_repo_split=cross_split,
                    repo_dir=rdir,
                    diff_timeout=args.diff_timeout,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                print(f"  ERROR on {repo_id}: {exc!r}", flush=True)
                continue

            if not commit_rows:
                n_no_kept += 1
                _append_done(
                    out_dir, repo_id, stats, shard_index=args.shard_index,
                )
                continue

            try:
                write_shard(out_dir, repo_id, commit_rows, qna_rows)
            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                print(f"  SHARD WRITE ERROR {repo_id}: {exc!r}", flush=True)
                continue

            n_ok += 1
            _append_done(out_dir, repo_id, stats, shard_index=args.shard_index)

    print(
        f"\nRepo results: ok={n_ok}, no_kept_commits={n_no_kept}, "
        f"skipped_done={n_skipped}, errors={n_errors}"
    )

    # Do NOT concat shards while other workers may still be writing. Only the
    # "no-shard" (single-process) mode runs concat inline.
    if not args.no_concat and args.shard_index is None:
        concat_shards(out_dir)


if __name__ == "__main__":
    main()
