#!/usr/bin/env python3
"""
Parquet-backed commit-sequence loader for Code2LoRA-GRU training.

Loads the dataset produced by ``create_dataset/build_commit_parquet_db.py``
(``commits.parquet`` + ``qna_pairs.parquet`` and/or their per-repo shards
under ``shards/``), and reshapes it into the same per-repo sequence format
consumed by ``train_code2lora_gru_commits.py``:

    {
        "repo_id":              str,
        "cross_repo_split":     str,                  # train | cr_val | cr_test
        "commit_diffs":         List[str],            # ordered by commit_index
        "commit_indices":       List[int],            # strictly increasing
        "commit_shas":          List[str],
        "commit_in_repo_splits":List[str],            # per-commit in_repo split
        "n_new_assertions":     List[int],            # per-commit new assertions
        "assertions_by_commit": Dict[int, List[(prefix, target)]],
        "assertion_splits":     Dict[int, List[str]], # mirrors assertions_by_commit
        "max_commit_index":     int,
        "_lazy_qna_spec":       optional; present when deferring QnA load (HF/concat)
    }

The loader supports:

* cross-repo split filtering (``cross_repo_splits`` arg)
* in-repo split filtering of assertions (``in_repo_splits`` arg)
* hard per-repo and per-assertion-split subsampling
* loading from a concatenated parquet file OR a ``shards/`` directory
* optional explicit list of repo IDs (``repo_ids_filter``)

Design notes (conference-quality)
---------------------------------
* The dataset on disk already encodes *kept commits only* (commits that
  changed test files AND introduced new/changed assertion identities) with
  ``production_code_diff`` computed between consecutive kept commits. We
  therefore never reconstruct diffs or assertions at load time.
* Each kept commit also carries a chronological 80/10/10 ``in_repo_split``
  label. Training uses only ``in_repo_split == 'train'`` assertions. Held-out
  in-repo evaluation uses ``val``/``test`` — the same repositories but later
  commits (chronologically). Held-out cross-repo evaluation uses repos in
  ``cr_val`` / ``cr_test`` with all their in-repo assertions.
* For efficiency we restrict pyarrow column projection to the minimum set of
  columns required at each phase (commits table and qna table).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow.compute as pc  # type: ignore
    import pyarrow.dataset as pads  # type: ignore

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False


SHARDS_SUBDIR = "shards"
COMMITS_FILENAME = "commits.parquet"
QNA_FILENAME = "qna_pairs.parquet"
# HF-export layout produced by create_dataset/export_commit_parquet_to_hf.py
HF_COMMITS_SUBDIR = "commits"
HF_QNA_SUBDIR = "qna"

# Columns read from each parquet; keep minimal to reduce I/O on large corpora.
_COMMIT_COLS = [
    "repo_id",
    "cross_repo_split",
    "commit_index",
    "commit_sha",
    "in_repo_split",
    "production_code_diff",
    "n_new_assertions",
]
_QNA_COLS = [
    "repo_id",
    "cross_repo_split",
    "commit_index",
    "in_repo_split",
    "prefix",
    "target",
]

# When ``defer_qna_materialization=True`` (concat/HF layout), each item may
# carry ``_lazy_qna_spec`` until :func:`materialize_lazy_qna_for_repo` runs.
LAZY_QNA_SPEC_KEY = "_lazy_qna_spec"


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise RuntimeError(
            "pyarrow is required for the Parquet loader. "
            "Install with: pip install pyarrow"
        )


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

@dataclass
class ParquetSources:
    """Resolved parquet inputs for commit and qna tables."""

    commits_paths: List[Path] = field(default_factory=list)
    qna_paths: List[Path] = field(default_factory=list)
    source_kind: str = "unknown"  # "concat" | "shards" | "mixed"

    def is_valid(self) -> bool:
        return bool(self.commits_paths) and bool(self.qna_paths)


def resolve_parquet_sources(
    parquet_dir: Optional[str] = None,
    commits_path: Optional[str] = None,
    qna_path: Optional[str] = None,
    prefer: str = "auto",
) -> ParquetSources:
    """Return a :class:`ParquetSources` describing the files to read.

    Resolution rules (first match wins):

    1. If ``commits_path`` and ``qna_path`` are both given: use them verbatim.
    2. Else, if ``parquet_dir`` is set:
         * ``prefer='concat'``  -> require the concatenated files.
         * ``prefer='shards'``  -> require the ``shards/`` directory.
         * ``prefer='auto'``    -> concat if present, otherwise shards.
    3. Raise a clear error if nothing is found.
    """
    srcs = ParquetSources()

    if commits_path and qna_path:
        cp = Path(commits_path).expanduser().resolve()
        qp = Path(qna_path).expanduser().resolve()
        if not cp.exists():
            raise FileNotFoundError(f"commits parquet not found: {cp}")
        if not qp.exists():
            raise FileNotFoundError(f"qna parquet not found: {qp}")
        srcs.commits_paths = [cp]
        srcs.qna_paths = [qp]
        srcs.source_kind = "concat"
        return srcs

    if parquet_dir is None:
        raise ValueError(
            "resolve_parquet_sources: provide either parquet_dir or "
            "(commits_path, qna_path)."
        )

    root = Path(parquet_dir).expanduser().resolve()
    concat_commits = root / COMMITS_FILENAME
    concat_qna = root / QNA_FILENAME
    shards_dir = root / SHARDS_SUBDIR
    hf_commits_dir = root / HF_COMMITS_SUBDIR
    hf_qna_dir = root / HF_QNA_SUBDIR
    have_concat = concat_commits.exists() and concat_qna.exists()
    have_shards = shards_dir.is_dir() and any(
        shards_dir.glob("*.commits.parquet")
    )
    have_hf = (
        hf_commits_dir.is_dir()
        and hf_qna_dir.is_dir()
        and any(hf_commits_dir.glob("*.parquet"))
        and any(hf_qna_dir.glob("*.parquet"))
    )

    if prefer == "concat" or (prefer == "auto" and have_concat):
        if not have_concat:
            raise FileNotFoundError(
                f"Expected {concat_commits} and {concat_qna}; not found."
            )
        srcs.commits_paths = [concat_commits]
        srcs.qna_paths = [concat_qna]
        srcs.source_kind = "concat"
        return srcs

    if prefer == "shards" or (prefer == "auto" and have_shards):
        if not have_shards:
            raise FileNotFoundError(
                f"Expected shards dir at {shards_dir}; not found."
            )
        cfs = sorted(shards_dir.glob("*.commits.parquet"))
        qfs = sorted(shards_dir.glob("*.qna.parquet"))
        if not cfs or not qfs:
            raise FileNotFoundError(
                f"No per-repo parquet shards under {shards_dir}."
            )
        srcs.commits_paths = cfs
        srcs.qna_paths = qfs
        srcs.source_kind = "shards"
        return srcs

    if prefer == "hf" or (prefer == "auto" and have_hf):
        if not have_hf:
            raise FileNotFoundError(
                f"Expected {hf_commits_dir}/ and {hf_qna_dir}/ with "
                "per-split parquet files; not found."
            )
        cfs = sorted(hf_commits_dir.glob("*.parquet"))
        qfs = sorted(hf_qna_dir.glob("*.parquet"))
        srcs.commits_paths = cfs
        srcs.qna_paths = qfs
        srcs.source_kind = "hf"
        return srcs

    raise FileNotFoundError(
        f"No recognized parquet layout under {root}. Expected one of: "
        f"concat ({COMMITS_FILENAME} + {QNA_FILENAME}), shards dir "
        f"({SHARDS_SUBDIR}/), or HF per-split layout "
        f"({HF_COMMITS_SUBDIR}/ + {HF_QNA_SUBDIR}/)."
    )


# ---------------------------------------------------------------------------
# Low-level table loading
# ---------------------------------------------------------------------------

def _build_filter_expr(filters: Optional[List[Tuple[str, str, Any]]]):
    if not filters:
        return None
    exprs = []
    for col, op, val in filters:
        if op == "in":
            exprs.append(pc.is_in(pc.field(col), pa.array(list(val))))
        elif op == "=":
            exprs.append(pc.equal(pc.field(col), val))
        elif op == "!=":
            exprs.append(pc.not_equal(pc.field(col), val))
        else:
            raise ValueError(f"unsupported filter op: {op}")
    filt = exprs[0]
    for e in exprs[1:]:
        filt = filt & e
    return filt


def _apply_mask_filter(tbl: "pa.Table", filters: Optional[List[Tuple[str, str, Any]]]) -> "pa.Table":
    if not filters:
        return tbl
    mask = None
    for col, op, val in filters:
        col_arr = tbl.column(col)
        if op == "in":
            m = pc.is_in(col_arr, pa.array(list(val)))
        elif op == "=":
            m = pc.equal(col_arr, val)
        elif op == "!=":
            m = pc.not_equal(col_arr, val)
        else:
            raise ValueError(f"unsupported filter op: {op}")
        mask = m if mask is None else (mask & m)
    return tbl.filter(mask) if mask is not None else tbl


def _read_single_parquet(
    path: Path,
    columns: Sequence[str],
    filters: Optional[List[Tuple[str, str, Any]]] = None,
) -> "pa.Table":
    """Read a single parquet file with column projection + row filter.

    Prefers pyarrow.dataset filter pushdown; falls back to read-then-mask.
    """
    _require_pyarrow()
    try:
        ds = pads.dataset(str(path), format="parquet")
        filt = _build_filter_expr(filters)
        return ds.to_table(columns=list(columns), filter=filt)
    except Exception:
        t = pq.read_table(str(path), columns=list(columns))
        return _apply_mask_filter(t, filters)


def _append_qna_batch_to_map(
    assertions_by_rc: Dict[Tuple[str, int], List[Tuple[str, str, str]]],
    batch: "pa.RecordBatch",
) -> None:
    """Merge one Arrow record batch into ``assertions_by_rc`` (prefix/target
    validation matches the eager QnA table path in the parquet loader).
    """
    q_repo = batch.column("repo_id").to_pylist()
    q_ci = batch.column("commit_index").to_pylist()
    q_in = batch.column("in_repo_split").to_pylist()
    q_prefix = batch.column("prefix").to_pylist()
    q_target = batch.column("target").to_pylist()
    for j in range(batch.num_rows):
        prefix = q_prefix[j]
        target = q_target[j]
        if not prefix or not target:
            continue
        if str(target).lstrip().startswith(","):
            continue
        assertions_by_rc[(q_repo[j], int(q_ci[j]))].append(
            (q_in[j], prefix, target),
        )


def _merge_qna_parquet_table_into(
    assertions_by_rc: Dict[Tuple[str, int], List[Tuple[str, str, str]]],
    qtab: "pa.Table",
) -> None:
    if qtab.num_rows == 0:
        return
    q_repo = qtab.column("repo_id").to_pylist()
    q_ci = qtab.column("commit_index").to_pylist()
    q_in = qtab.column("in_repo_split").to_pylist()
    q_prefix = qtab.column("prefix").to_pylist()
    q_target = qtab.column("target").to_pylist()
    for j in range(qtab.num_rows):
        prefix = q_prefix[j]
        target = q_target[j]
        if not prefix or not target:
            continue
        if str(target).lstrip().startswith(","):
            continue
        assertions_by_rc[(q_repo[j], int(q_ci[j]))].append(
            (q_in[j], prefix, target),
        )


def materialize_lazy_qna_for_repo(repo: Dict[str, Any]) -> None:
    """Fill ``assertions_by_commit`` / ``assertion_splits`` from Parquet for
    one repo, consuming ``_lazy_qna_spec``.

    Used so the concat/HF layout never holds every training QnA row in RAM at
    once: call this once per repo immediately before embedding/tokenization
    (e.g. inside the trainer's cache precompute loop).
    """
    spec = repo.pop(LAZY_QNA_SPEC_KEY, None)
    if spec is None:
        return

    rid = repo["repo_id"]
    qna_paths: Tuple[Path, ...] = tuple(Path(p) for p in spec["qna_paths"])
    cross_set: Set[str] = set(spec["cross_repo_splits"])
    in_tuple = spec.get("in_repo_splits")
    in_set: Optional[Set[str]] = (
        None if in_tuple is None else set(in_tuple)
    )

    assertions_by_rc: Dict[Tuple[str, int], List[Tuple[str, str, str]]] = (
        defaultdict(list)
    )
    qna_filters: List[Tuple[str, str, Any]] = [
        ("cross_repo_split", "in", list(cross_set)),
        ("repo_id", "=", rid),
    ]
    if in_set is not None:
        qna_filters.append(("in_repo_split", "in", list(in_set)))
    filt = _build_filter_expr(qna_filters)

    _require_pyarrow()
    for qp in qna_paths:
        if not qp.exists():
            continue
        try:
            ds = pads.dataset(str(qp), format="parquet")
            scanner = ds.scanner(
                filter=filt,
                columns=list(_QNA_COLS),
                batch_size=65536,
            )
            for rb in scanner.to_batches():
                if rb.num_rows == 0:
                    continue
                _append_qna_batch_to_map(assertions_by_rc, rb)
        except Exception:
            qtab = _read_single_parquet(qp, _QNA_COLS, qna_filters)
            _merge_qna_parquet_table_into(assertions_by_rc, qtab)

    commit_indices = repo["commit_indices"]
    assertions_by_commit: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    assertion_splits: Dict[int, List[str]] = defaultdict(list)
    for ci in commit_indices:
        for (ins, pfx, tgt) in assertions_by_rc.get((rid, int(ci)), []):
            assertions_by_commit[int(ci)].append((pfx, tgt))
            assertion_splits[int(ci)].append(ins)

    repo["assertions_by_commit"] = dict(assertions_by_commit)
    repo["assertion_splits"] = dict(assertion_splits)


def estimate_total_qna_rows(
    sources: ParquetSources,
    *,
    cross_repo_splits: Iterable[str],
    in_repo_splits: Optional[Iterable[str]] = None,
    repo_ids: Optional[Sequence[str]] = None,
) -> int:
    """Row count for QnA rows matching the same filters as materialization.

    Uses dataset predicate pushdown (no Python list of all strings).
    """
    _require_pyarrow()
    cross_set = set(cross_repo_splits)
    in_set: Optional[Set[str]] = (
        set(in_repo_splits) if in_repo_splits is not None else None
    )
    filters: List[Tuple[str, str, Any]] = [
        ("cross_repo_split", "in", list(cross_set)),
    ]
    if repo_ids is not None:
        filters.append(("repo_id", "in", list(repo_ids)))
    if in_set is not None:
        filters.append(("in_repo_split", "in", list(in_set)))
    filt = _build_filter_expr(filters)
    total = 0
    for p in sources.qna_paths:
        if not Path(p).exists():
            continue
        ds = pads.dataset(str(p), format="parquet")
        try:
            total += int(ds.count_rows(filter=filt))
        except Exception:
            t = _read_single_parquet(Path(p), ["repo_id"], filters)
            total += t.num_rows
    return total


def _probe_shard_cross_split(path: Path) -> Optional[str]:
    """Read a tiny slice of a per-repo shard to discover its cross_repo_split.

    Assumes a single repo per shard (as produced by ``write_shard``). Returns
    None if the shard is empty or unreadable.
    """
    try:
        t = pq.read_table(
            str(path),
            columns=["repo_id", "cross_repo_split"],
        )
        if t.num_rows == 0:
            return None
        # All rows in a per-repo shard have the same cross_repo_split.
        return t.column("cross_repo_split")[0].as_py()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

_TRAIN_CROSS = ("train",)
_EVAL_CROSS = ("cr_val", "cr_test")


def load_commit_sequences_from_parquet(
    sources: ParquetSources,
    *,
    cross_repo_splits: Iterable[str] = _TRAIN_CROSS,
    in_repo_splits: Optional[Iterable[str]] = ("train",),
    repo_ids_filter: Optional[Iterable[str]] = None,
    limit_repos: Optional[int] = None,
    min_commits: int = 1,
    require_assertions: bool = True,
    shuffle_repos: bool = False,
    seed: int = 0,
    defer_qna_materialization: bool = False,
) -> List[Dict[str, Any]]:
    """Load per-repo commit sequences + assertions from parquet shards.

    Parameters
    ----------
    sources:
        ParquetSources returned by :func:`resolve_parquet_sources`.
    cross_repo_splits:
        Keep only repos whose ``cross_repo_split`` is in this set.
    in_repo_splits:
        Keep only assertions whose ``in_repo_split`` is in this set. Use
        ``None`` to keep all assertions regardless of split.
    repo_ids_filter:
        Optional iterable of repo IDs to further restrict to.
    limit_repos:
        Hard cap on number of repos returned (for debugging / smoke tests).
    min_commits:
        Require at least this many kept commits for the repo to be returned.
    require_assertions:
        If True, drop repos with zero assertions after in-repo filtering.
        Ignored for assertion emptiness when ``defer_qna_materialization`` is
        True (assertions are filled later via :func:`materialize_lazy_qna_for_repo`).
    shuffle_repos:
        Shuffle returned list deterministically (``random.Random(seed)``).
    defer_qna_materialization:
        For concatenated / HF per-split layouts only: build repo items from
        commits only and attach ``_lazy_qna_spec`` instead of loading all QnA
        rows into RAM. Call :func:`materialize_lazy_qna_for_repo` per repo
        before training/eval needs ``assertions_by_commit``.

    Returns
    -------
    list of repo items, see module docstring for schema.
    """
    _require_pyarrow()
    cross_set = set(cross_repo_splits)
    in_set: Optional[Set[str]] = (
        set(in_repo_splits) if in_repo_splits is not None else None
    )
    repo_filter_set: Optional[Set[str]] = (
        set(repo_ids_filter) if repo_ids_filter is not None else None
    )

    # Split paths strategy:
    # * For "shards" source: each file is per-repo; filter files up-front by a
    #   cheap probe on cross_repo_split (and optional repo list). This bounds
    #   peak memory at one shard at a time.
    # * For "concat" source: read the whole concatenated table with pushdown.
    items: List[Dict[str, Any]] = []
    eff_require_assertions = (
        require_assertions and not defer_qna_materialization
    )

    def _pair_shards() -> List[Tuple[Path, Path]]:
        """Zip commit/qna shard paths by base name."""
        q_by_stem: Dict[str, Path] = {
            p.name.replace(".qna.parquet", ""): p for p in sources.qna_paths
        }
        out: List[Tuple[Path, Path]] = []
        for cp in sources.commits_paths:
            stem = cp.name.replace(".commits.parquet", "")
            qp = q_by_stem.get(stem)
            if qp is None:
                continue
            out.append((cp, qp))
        return out

    def _process_commit_table(ctab: "pa.Table") -> Dict[str, Dict[str, Any]]:
        per_repo: Dict[str, Dict[str, Any]] = {}
        if ctab.num_rows == 0:
            return per_repo
        c_repo = ctab.column("repo_id").to_pylist()
        c_cross = ctab.column("cross_repo_split").to_pylist()
        c_ci = ctab.column("commit_index").to_pylist()
        c_sha = ctab.column("commit_sha").to_pylist()
        c_in = ctab.column("in_repo_split").to_pylist()
        c_diff = ctab.column("production_code_diff").to_pylist()
        c_nnew = ctab.column("n_new_assertions").to_pylist()
        for i in range(ctab.num_rows):
            rid = c_repo[i]
            rec = per_repo.setdefault(rid, {
                "repo_id": rid,
                "cross_repo_split": c_cross[i],
                "_rows": [],
            })
            rec["_rows"].append((
                int(c_ci[i]), c_sha[i], c_in[i],
                c_diff[i] or "", int(c_nnew[i] or 0),
            ))
        return per_repo

    def _process_qna_table(
        qtab: "pa.Table",
    ) -> Dict[Tuple[str, int], List[Tuple[str, str, str]]]:
        assertions_by_rc: Dict[Tuple[str, int], List[Tuple[str, str, str]]] = defaultdict(list)
        if qtab.num_rows == 0:
            return assertions_by_rc
        q_repo = qtab.column("repo_id").to_pylist()
        q_ci = qtab.column("commit_index").to_pylist()
        q_in = qtab.column("in_repo_split").to_pylist()
        q_prefix = qtab.column("prefix").to_pylist()
        q_target = qtab.column("target").to_pylist()
        for j in range(qtab.num_rows):
            prefix = q_prefix[j]
            target = q_target[j]
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue
            assertions_by_rc[(q_repo[j], int(q_ci[j]))].append(
                (q_in[j], prefix, target),
            )
        return assertions_by_rc

    def _shape_items(
        per_repo: Dict[str, Dict[str, Any]],
        assertions_by_rc: Dict[Tuple[str, int], List[Tuple[str, str, str]]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rid, rec in per_repo.items():
            rows = sorted(rec["_rows"], key=lambda r: r[0])
            if len(rows) < min_commits:
                continue
            commit_indices = [r[0] for r in rows]
            commit_shas = [r[1] for r in rows]
            commit_in_splits = [r[2] for r in rows]
            commit_diffs = [r[3] for r in rows]
            n_new_assertions = [r[4] for r in rows]
            assertions_by_commit: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
            assertion_splits: Dict[int, List[str]] = defaultdict(list)
            total_assertions = 0
            for ci in commit_indices:
                for (ins, pfx, tgt) in assertions_by_rc.get((rid, ci), []):
                    assertions_by_commit[ci].append((pfx, tgt))
                    assertion_splits[ci].append(ins)
                    total_assertions += 1
            if eff_require_assertions and total_assertions == 0:
                continue
            out.append({
                "repo_id": rid,
                "cross_repo_split": rec["cross_repo_split"],
                "commit_diffs": commit_diffs,
                "commit_indices": commit_indices,
                "commit_shas": commit_shas,
                "commit_in_repo_splits": commit_in_splits,
                "n_new_assertions": n_new_assertions,
                "assertions_by_commit": dict(assertions_by_commit),
                "assertion_splits": dict(assertion_splits),
                "max_commit_index": commit_indices[-1] if commit_indices else -1,
            })
        return out

    if sources.source_kind == "shards":
        pairs = _pair_shards()
        if not pairs:
            return []
        # Shard-level filtering: probe cross_repo_split to avoid reading shards
        # that cannot match. Each shard is per-repo, so one probe is enough.
        for cp, qp in pairs:
            # Filter by repo list first (cheapest).
            stem = cp.name.replace(".commits.parquet", "")
            rid_from_stem = stem.replace("__", "/", 1)
            if repo_filter_set is not None and rid_from_stem not in repo_filter_set:
                continue
            # Probe cross_repo_split
            cs = _probe_shard_cross_split(cp)
            if cs is None or cs not in cross_set:
                continue
            # Load this shard's commit table (all columns).
            commit_filters = []
            if repo_filter_set:
                commit_filters.append(("repo_id", "in", list(repo_filter_set)))
            ctab = _read_single_parquet(cp, _COMMIT_COLS, commit_filters)
            per_repo = _process_commit_table(ctab)
            if not per_repo:
                continue
            # Load the matching qna shard.
            qna_filters = []
            if repo_filter_set:
                qna_filters.append(("repo_id", "in", list(repo_filter_set)))
            if in_set is not None:
                qna_filters.append(("in_repo_split", "in", list(in_set)))
            qtab = _read_single_parquet(qp, _QNA_COLS, qna_filters)
            assertions_by_rc = _process_qna_table(qtab)

            items.extend(_shape_items(per_repo, assertions_by_rc))

            # Early-stop when enough repos have been collected.
            if limit_repos is not None and len(items) >= limit_repos * 2:
                break
    else:
        # Concatenated / HF-per-split source: read commits first (small), pick
        # the subset of repo_ids we actually want, then pushdown-filter the
        # (potentially multi-GB) qna file by that explicit repo list so we
        # never materialize the full qna table in RAM.
        commit_filters = [("cross_repo_split", "in", list(cross_set))]
        if repo_filter_set:
            commit_filters.append(("repo_id", "in", list(repo_filter_set)))
        ctab_list: List["pa.Table"] = []
        for p in sources.commits_paths:
            ctab_list.append(_read_single_parquet(p, _COMMIT_COLS, commit_filters))
        ctab = (
            ctab_list[0] if len(ctab_list) == 1
            else pa.concat_tables(ctab_list, promote=True)
        )
        per_repo = _process_commit_table(ctab)

        # Enforce min_commits / limit_repos *before* loading qna so we only
        # fetch the assertions we're actually going to use. This is the
        # critical memory-saving step for the HF per-split layout.
        # _process_commit_table stores unsorted per-commit tuples under "_rows".
        kept_repo_ids: List[str] = sorted(
            rid for rid, rec in per_repo.items()
            if len(rec.get("_rows", [])) >= min_commits
        )
        if limit_repos is not None and len(kept_repo_ids) > limit_repos:
            kept_repo_ids = kept_repo_ids[:limit_repos]
        kept_repo_set = set(kept_repo_ids)
        # Drop everything from per_repo that we did not keep.
        per_repo = {rid: rec for rid, rec in per_repo.items() if rid in kept_repo_set}

        if defer_qna_materialization:
            lazy_spec = {
                "qna_paths": tuple(sources.qna_paths),
                "cross_repo_splits": tuple(sorted(cross_set)),
                "in_repo_splits": (
                    None if in_set is None else tuple(sorted(in_set))
                ),
            }
            for it in _shape_items(per_repo, {}):
                it[LAZY_QNA_SPEC_KEY] = lazy_spec
                items.append(it)
        else:
            qna_filters = [("cross_repo_split", "in", list(cross_set))]
            # Always push a repo-id filter so a multi-GB qna/train.parquet is
            # scanned row-group-by-row-group and only the matching rows are
            # brought into memory.
            if kept_repo_ids:
                qna_filters.append(("repo_id", "in", kept_repo_ids))
            elif repo_filter_set:
                qna_filters.append(("repo_id", "in", list(repo_filter_set)))
            if in_set is not None:
                qna_filters.append(("in_repo_split", "in", list(in_set)))
            qtab_list: List["pa.Table"] = []
            for p in sources.qna_paths:
                qtab_list.append(
                    _read_single_parquet(p, _QNA_COLS, qna_filters),
                )
            qtab = (
                qtab_list[0] if len(qtab_list) == 1
                else pa.concat_tables(qtab_list, promote=True)
            )
            assertions_by_rc = _process_qna_table(qtab)
            items.extend(_shape_items(per_repo, assertions_by_rc))

    items.sort(key=lambda x: x["repo_id"])
    if shuffle_repos:
        import random

        rng = random.Random(seed)
        rng.shuffle(items)

    if limit_repos is not None:
        items = items[: limit_repos]
    return items


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_items(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_repos = len(items)
    n_commits = sum(len(it["commit_diffs"]) for it in items)
    n_assert = sum(
        sum(len(v) for v in it["assertions_by_commit"].values())
        for it in items
    )
    n_commits_w_asserts = sum(
        sum(1 for v in it["assertions_by_commit"].values() if v)
        for it in items
    )
    by_cross: Dict[str, int] = defaultdict(int)
    for it in items:
        by_cross[it["cross_repo_split"]] += 1
    return {
        "n_repos": n_repos,
        "n_commits": n_commits,
        "n_assertions": n_assert,
        "n_commits_with_assertions": n_commits_w_asserts,
        "by_cross_repo_split": dict(by_cross),
    }


__all__ = [
    "ParquetSources",
    "resolve_parquet_sources",
    "load_commit_sequences_from_parquet",
    "summarize_items",
    "LAZY_QNA_SPEC_KEY",
    "materialize_lazy_qna_for_repo",
    "estimate_total_qna_rows",
    "SHARDS_SUBDIR",
    "COMMITS_FILENAME",
    "QNA_FILENAME",
]
