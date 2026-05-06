#!/usr/bin/env python3
"""Repository commit/test/QnA event statistics for the commit-Parquet dataset.

This analysis answers:

* how many first-parent commits each repository originally has;
* how many original commits touch Python test files;
* how many kept commit steps produce QnA rows;
* how many QnAs are created per kept step, split, repo, and event type.

Outputs are written under ``analysis/output`` by default:

* ``repo_commit_qna_stats.json``
* ``repo_commit_qna_stats_summary.txt``
* ``repo_commit_qna_figures/*.png``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pyarrow is required: {exc}") from exc


SPLITS = ("train", "cr_val", "cr_test")
TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]+\.py$|(^|/)[^/]+_test\.py$|(^|/)conftest\.py$",
    re.IGNORECASE,
)


def repo_dir_for(repos_root: Path, repo_id: str) -> Path:
    parts = repo_id.split("/")
    if len(parts) == 2:
        return repos_root / parts[0] / parts[1]
    return repos_root / repo_id


def is_test_path(path: str) -> bool:
    if not path or not path.endswith(".py"):
        return False
    return bool(TEST_FILE_RE.search(path))


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def dist(values: Iterable[int | float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals_sorted),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
        "mean": mean(vals_sorted),
        "median": median(vals_sorted),
        "std": pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0,
        "p25": percentile(vals_sorted, 25),
        "p75": percentile(vals_sorted, 75),
        "p90": percentile(vals_sorted, 90),
        "p95": percentile(vals_sorted, 95),
        "p99": percentile(vals_sorted, 99),
    }


def git_first_parent_test_stats(
    repo_dir: Path,
    timeout: int = 600,
) -> Optional[Dict[str, Any]]:
    """Return full first-parent commit counts and test-file-touch counts."""
    if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
        return None

    cmd = [
        "git", "-C", str(repo_dir),
        "log", "--first-parent", "--reverse",
        "--name-only",
        "--format=COMMIT %H",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None

    total = 0
    test_touch = 0
    current_has_test = False

    def flush() -> None:
        nonlocal test_touch, current_has_test
        if current_has_test:
            test_touch += 1
        current_has_test = False

    seen_commit = False
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("COMMIT "):
            if seen_commit:
                flush()
            seen_commit = True
            total += 1
            continue
        if is_test_path(line):
            current_has_test = True

    if seen_commit:
        flush()

    return {
        "original_first_parent_commits": total,
        "original_commits_touching_test_files": test_touch,
        "original_non_test_touch_commits": max(total - test_touch, 0),
        "test_touch_commit_ratio": (test_touch / total) if total else 0.0,
    }


def _read_table(path: Path, columns: List[str]) -> "pa.Table":
    schema = pq.read_schema(path)
    present = [c for c in columns if c in schema.names]
    return pq.read_table(path, columns=present)


def load_parquet_tables(data_dir: Path) -> Tuple["pa.Table", "pa.Table"]:
    commits_dir = data_dir / "commits"
    qna_dir = data_dir / "qna"
    if commits_dir.is_dir() and qna_dir.is_dir():
        commits = []
        qnas = []
        for split in SPLITS:
            cpath = commits_dir / f"{split}.parquet"
            qpath = qna_dir / f"{split}.parquet"
            if cpath.exists():
                commits.append(_read_table(
                    cpath,
                    [
                        "repo_id",
                        "cross_repo_split",
                        "commit_index",
                        "commit_sha",
                        "in_repo_split",
                        "n_new_assertions",
                        "n_added_assertions",
                        "n_modified_assertions",
                    ],
                ))
            if qpath.exists():
                qnas.append(_read_table(
                    qpath,
                    [
                        "repo_id",
                        "cross_repo_split",
                        "commit_index",
                        "commit_sha",
                        "test_file",
                        "file_split",
                        "assertion_type",
                        "assertion_event_type",
                    ],
                ))
        if not commits or not qnas:
            raise SystemExit(f"No HF parquet split files under {data_dir}")
        return pa.concat_tables(commits, promote_options="default"), pa.concat_tables(qnas, promote_options="default")

    cpath = data_dir / "commits.parquet"
    qpath = data_dir / "qna_pairs.parquet"
    if cpath.exists() and qpath.exists():
        return (
            _read_table(cpath, [
                "repo_id",
                "cross_repo_split",
                "commit_index",
                "commit_sha",
                "in_repo_split",
                "n_new_assertions",
                "n_added_assertions",
                "n_modified_assertions",
            ]),
            _read_table(qpath, [
                "repo_id",
                "cross_repo_split",
                "commit_index",
                "commit_sha",
                "test_file",
                "file_split",
                "assertion_type",
                "assertion_event_type",
            ]),
        )

    raise SystemExit(
        f"Unrecognized parquet layout at {data_dir}; expected HF commits/qna dirs "
        "or commits.parquet + qna_pairs.parquet."
    )


def table_to_columns(table: "pa.Table") -> Dict[str, List[Any]]:
    return {name: table.column(name).to_pylist() for name in table.column_names}


def collect_parquet_stats(commits: "pa.Table", qna: "pa.Table") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    c = table_to_columns(commits)
    q = table_to_columns(qna)

    qna_by_repo_commit: Dict[Tuple[str, int], Counter[str]] = defaultdict(Counter)
    qna_by_repo: Counter[str] = Counter()
    qna_by_split: Counter[str] = Counter()
    qna_by_event_type: Counter[str] = Counter()
    qna_by_assertion_type: Counter[str] = Counter()
    qna_by_file_split: Counter[str] = Counter()
    test_files_by_repo: Dict[str, set[str]] = defaultdict(set)

    q_repo = q["repo_id"]
    q_ci = q["commit_index"]
    q_cross = q.get("cross_repo_split", [""] * qna.num_rows)
    q_event = q.get("assertion_event_type", ["unknown"] * qna.num_rows)
    q_assert = q.get("assertion_type", ["unknown"] * qna.num_rows)
    q_file_split = q.get("file_split", [""] * qna.num_rows)
    q_test_file = q.get("test_file", [""] * qna.num_rows)

    for repo, ci, split, event_type, assertion_type, file_split, test_file in zip(
        q_repo,
        q_ci,
        q_cross,
        q_event,
        q_assert,
        q_file_split,
        q_test_file,
    ):
        repo = str(repo)
        ci = int(ci)
        event_type = str(event_type or "unknown")
        qna_by_repo_commit[(repo, ci)][event_type] += 1
        qna_by_repo[repo] += 1
        qna_by_split[str(split)] += 1
        qna_by_event_type[event_type] += 1
        qna_by_assertion_type[str(assertion_type or "unknown")] += 1
        if file_split:
            qna_by_file_split[str(file_split)] += 1
        if test_file:
            test_files_by_repo[repo].add(str(test_file))

    c_repo = c["repo_id"]
    c_cross = c.get("cross_repo_split", [""] * commits.num_rows)
    c_ci = c["commit_index"]
    c_n = c["n_new_assertions"]
    c_added = c.get("n_added_assertions", [0] * commits.num_rows)
    c_modified = c.get("n_modified_assertions", [0] * commits.num_rows)

    per_repo_acc: Dict[str, Dict[str, Any]] = {}
    per_step: List[Dict[str, Any]] = []
    for repo, split, ci, n, added, modified in zip(
        c_repo,
        c_cross,
        c_ci,
        c_n,
        c_added,
        c_modified,
    ):
        repo = str(repo)
        ci = int(ci)
        n = int(n or 0)
        added = int(added or 0)
        modified = int(modified or 0)
        acc = per_repo_acc.setdefault(repo, {
            "repo_id": repo,
            "cross_repo_split": str(split),
            "kept_qna_steps": 0,
            "qna_events": 0,
            "added_events": 0,
            "modified_events": 0,
            "max_qna_events_in_step": 0,
        })
        acc["kept_qna_steps"] += 1
        acc["qna_events"] += n
        acc["added_events"] += added
        acc["modified_events"] += modified
        acc["max_qna_events_in_step"] = max(acc["max_qna_events_in_step"], n)

        counts = qna_by_repo_commit.get((repo, ci), Counter())
        per_step.append({
            "repo_id": repo,
            "cross_repo_split": str(split),
            "commit_index": ci,
            "qna_events": n,
            "added_events": added,
            "modified_events": modified,
            "qna_added_from_rows": int(counts.get("added", 0)),
            "qna_modified_from_rows": int(counts.get("modified", 0)),
        })

    summary = {
        "parquet": {
            "repos": len(per_repo_acc),
            "kept_qna_steps": commits.num_rows,
            "qna_events": qna.num_rows,
            "qna_by_cross_repo_split": dict(qna_by_split),
            "qna_by_event_type": dict(qna_by_event_type),
            "qna_by_file_split": dict(qna_by_file_split),
            "top_assertion_types": dict(qna_by_assertion_type.most_common(25)),
            "qna_events_per_kept_step": dist([r["qna_events"] for r in per_step]),
            "added_events_per_kept_step": dist([r["added_events"] for r in per_step]),
            "modified_events_per_kept_step": dist([r["modified_events"] for r in per_step]),
            "kept_qna_steps_per_repo": dist([r["kept_qna_steps"] for r in per_repo_acc.values()]),
            "qna_events_per_repo": dist([r["qna_events"] for r in per_repo_acc.values()]),
            "unique_test_files_per_repo": dist([len(v) for v in test_files_by_repo.values()]),
        }
    }
    return summary, list(per_repo_acc.values()), per_step


def add_git_stats(
    repos_root: Path,
    per_repo: List[Dict[str, Any]],
    *,
    timeout: int,
    limit_repos: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw):
            return x

    rows = per_repo if limit_repos is None else per_repo[:limit_repos]
    missing = 0
    for row in tqdm(rows, desc="git repos", unit="repo"):
        stats = git_first_parent_test_stats(
            repo_dir_for(repos_root, row["repo_id"]),
            timeout=timeout,
        )
        if stats is None:
            missing += 1
            row.update({
                "original_first_parent_commits": None,
                "original_commits_touching_test_files": None,
                "test_touch_commit_ratio": None,
            })
        else:
            row.update(stats)

    analyzed = [r for r in rows if r.get("original_first_parent_commits") is not None]
    return {
        "repos_root": str(repos_root),
        "repos_requested": len(rows),
        "repos_with_git_stats": len(analyzed),
        "repos_missing_or_git_error": missing,
        "original_first_parent_commits_per_repo": dist(
            r["original_first_parent_commits"] for r in analyzed
        ),
        "original_test_touch_commits_per_repo": dist(
            r["original_commits_touching_test_files"] for r in analyzed
        ),
        "original_test_touch_commit_ratio_per_repo": dist(
            r["test_touch_commit_ratio"] for r in analyzed
        ),
        "total_original_first_parent_commits": sum(
            int(r["original_first_parent_commits"]) for r in analyzed
        ),
        "total_original_commits_touching_test_files": sum(
            int(r["original_commits_touching_test_files"]) for r in analyzed
        ),
        "total_kept_qna_steps": sum(int(r["kept_qna_steps"]) for r in analyzed),
        "total_qna_events": sum(int(r["qna_events"]) for r in analyzed),
    }


def write_plots(out_dir: Path, per_repo: List[Dict[str, Any]], per_step: List[Dict[str, Any]]) -> None:
    fig_dir = out_dir / "repo_commit_qna_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save_hist(values: List[float], path: Path, title: str, xlabel: str, bins: int = 40, log_y: bool = False) -> None:
        vals = [v for v in values if v is not None and not math.isnan(float(v))]
        if not vals:
            return
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.hist(vals, bins=bins, color="#3b82f6", edgecolor="white")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("#")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    save_hist(
        [r.get("original_first_parent_commits") for r in per_repo if r.get("original_first_parent_commits") is not None],
        fig_dir / "original_commits_per_repo.png",
        "Original first-parent commits per repo",
        "# commits",
    )
    save_hist(
        [r.get("original_commits_touching_test_files") for r in per_repo if r.get("original_commits_touching_test_files") is not None],
        fig_dir / "test_touch_commits_per_repo.png",
        "Original commits touching test files per repo",
        "# commits touching test files",
    )
    save_hist(
        [r.get("test_touch_commit_ratio") for r in per_repo if r.get("test_touch_commit_ratio") is not None],
        fig_dir / "test_touch_commit_ratio_per_repo.png",
        "Fraction of original commits touching test files",
        "test-touch commits / original commits",
    )
    save_hist(
        [r["qna_events"] for r in per_step],
        fig_dir / "qna_events_per_kept_step.png",
        "QnA events created per kept commit step",
        "# QnA events in step",
        log_y=True,
    )
    save_hist(
        [r["kept_qna_steps"] for r in per_repo],
        fig_dir / "kept_qna_steps_per_repo.png",
        "Kept QnA-producing steps per repo",
        "# kept steps",
    )

    xs = [
        r.get("original_commits_touching_test_files")
        for r in per_repo
        if r.get("original_commits_touching_test_files") is not None
    ]
    ys = [
        r["kept_qna_steps"]
        for r in per_repo
        if r.get("original_commits_touching_test_files") is not None
    ]
    if xs and ys:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.scatter(xs, ys, s=14, alpha=0.6, color="#ef4444")
        ax.set_title("Original test-touch commits vs kept QnA steps")
        ax.set_xlabel("# original commits touching test files")
        ax.set_ylabel("# kept QnA-producing steps")
        fig.tight_layout()
        fig.savefig(fig_dir / "test_touch_vs_kept_qna_steps.png")
        plt.close(fig)

    event_counts = Counter()
    for row in per_step:
        event_counts["added"] += int(row.get("added_events", 0))
        event_counts["modified"] += int(row.get("modified_events", 0))
    if event_counts:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
        labels = list(event_counts.keys())
        vals = [event_counts[k] for k in labels]
        ax.bar(labels, vals, color=["#3b82f6", "#f97316"])
        ax.set_title("QnA event types")
        ax.set_ylabel("# QnA events")
        fig.tight_layout()
        fig.savefig(fig_dir / "qna_event_types.png")
        plt.close(fig)


def write_summary_text(out_path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    parquet = summary["parquet"]
    git = summary.get("git", {})
    lines = [
        "Repo commit/QnA statistics",
        "==========================",
        f"Dataset: {summary['dataset_dir']}",
        f"Repos: {parquet['repos']}",
        f"Kept QnA steps: {parquet['kept_qna_steps']:,}",
        f"QnA events: {parquet['qna_events']:,}",
        f"QnA by event type: {parquet['qna_by_event_type']}",
        "",
    ]
    if git:
        lines.extend([
            "Original git history",
            "--------------------",
            f"Repos with git stats: {git['repos_with_git_stats']}",
            f"Total original first-parent commits: {git['total_original_first_parent_commits']:,}",
            f"Total original commits touching test files: {git['total_original_commits_touching_test_files']:,}",
            f"Total kept QnA-producing steps: {git['total_kept_qna_steps']:,}",
            f"Total QnA events: {git['total_qna_events']:,}",
            "",
            f"Original commits/repo: {git['original_first_parent_commits_per_repo']}",
            f"Test-touch commits/repo: {git['original_test_touch_commits_per_repo']}",
            f"Test-touch ratio/repo: {git['original_test_touch_commit_ratio_per_repo']}",
            "",
        ])
    lines.extend([
        "QnA per step",
        "------------",
        json.dumps(parquet["qna_events_per_kept_step"], indent=2),
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default_root = Path(os.environ.get("SCRATCH", str(Path.home() / "scratch"))) / "REPO_DATASET"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=default_root / "commit_parquet_hf")
    ap.add_argument("--repos-root", type=Path, default=default_root / "repositories")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "output")
    ap.add_argument("--git-timeout", type=int, default=900)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument(
        "--per-step-jsonl",
        action="store_true",
        help="Also write per-step rows as JSONL; can be large but useful for audits.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.expanduser().resolve()
    repos_root = args.repos_root.expanduser().resolve()

    print(f"Loading parquet dataset: {data_dir}", flush=True)
    commits, qna = load_parquet_tables(data_dir)
    summary, per_repo, per_step = collect_parquet_stats(commits, qna)
    summary["dataset_dir"] = str(data_dir)

    print(f"Collecting git stats from: {repos_root}", flush=True)
    summary["git"] = add_git_stats(
        repos_root,
        per_repo,
        timeout=args.git_timeout,
        limit_repos=args.limit_repos,
    )

    payload = {
        "summary": summary,
        "per_repo": per_repo,
        "per_step_sample": per_step[:1000],
        "notes": {
            "per_step_sample": "First 1000 kept steps only. Use --per-step-jsonl for all rows.",
            "original_commits_touching_test_files": "First-parent commits whose changed path list includes Python test-file paths.",
            "kept_qna_steps": "Rows in commits parquet. Each kept step has at least one assertion event.",
        },
    }

    json_path = out_dir / "repo_commit_qna_stats.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    summary_path = out_dir / "repo_commit_qna_stats_summary.txt"
    write_summary_text(summary_path, payload)
    print(f"Wrote {summary_path}", flush=True)

    if args.per_step_jsonl:
        jsonl_path = out_dir / "repo_commit_qna_steps.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in per_step:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {jsonl_path}", flush=True)

    write_plots(out_dir, per_repo, per_step)
    print(f"Wrote figures under {out_dir / 'repo_commit_qna_figures'}", flush=True)


if __name__ == "__main__":
    main()
