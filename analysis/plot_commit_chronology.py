#!/usr/bin/env python3
"""
Chronological commit / QnA visualizations for RepoPeft commit-parquet data.

Requires cloned repositories under ``repos_root`` (same layout as
``create_dataset/build_commit_parquet_db.py``) and the Parquet dataset
produced by that script (concat, ``shards/``, or HF layout — see
``hypernetwork.parquet_commit_dataset.resolve_parquet_sources``).

Outputs (under ``--out-dir``):

1. ``commits_timeline_churn.png`` — x: time, y: one row per repo; each dot is
   a commit (``git log --reverse --first-parent``). Marker area scales with
   total lines touched in that commit (sum of ``git numstat`` additions +
   deletions, binary files skipped).

2. ``commits_timeline_qna.png`` — same axes; blue small dots when a commit has
   zero new training QnA rows at that SHA; red dots when ``qna_pairs.parquet``
   has at least one row for that (repo, commit_sha), with area scaling with
   that count.

3. ``histograms_per_repo/*.png`` — one figure per repo: bars along the full
   commit timeline (commit order / dates on x), bar height = number of new
   QnA pairs attributed to that commit (0 for commits with no new QnAs).

Example::

    python analysis/plot_commit_chronology.py \\
        --parquet-dir  $SCRATCH/REPO_DATASET/commit_parquet \\
        --repos-root   $SCRATCH/REPO_DATASET/repositories \\
        --out-dir      analysis/output/commit_chronology_figs

Optional: ``--limit-repos 30`` for a quick smoke plot.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}") from exc

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pyarrow is required: {exc}") from exc


@dataclass(frozen=True)
class ParquetSources:
    source_kind: str
    commits_paths: List[Path]
    qna_paths: List[Path]


def resolve_parquet_sources(
    *,
    parquet_dir: Optional[str],
    commits_path: Optional[str],
    qna_path: Optional[str],
    prefer: str,
) -> ParquetSources:
    """Resolve concat, shard, or HF-style commit/QnA parquet files."""
    if commits_path and qna_path:
        return ParquetSources(
            source_kind="concat",
            commits_paths=[Path(commits_path).expanduser().resolve()],
            qna_paths=[Path(qna_path).expanduser().resolve()],
        )
    if not parquet_dir:
        raise SystemExit("Provide --parquet-dir or both --commits-file and --qna-file.")

    root = Path(parquet_dir).expanduser().resolve()
    concat_commits = root / "commits.parquet"
    concat_qna = root / "qna_pairs.parquet"
    hf_commits = sorted((root / "commits").glob("*.parquet")) if (root / "commits").is_dir() else []
    hf_qna = sorted((root / "qna").glob("*.parquet")) if (root / "qna").is_dir() else []
    shard_root = root / "shards" if (root / "shards").is_dir() else root
    shard_commits = sorted(shard_root.glob("*.commits.parquet"))
    shard_qna = sorted(shard_root.glob("*.qna.parquet"))

    if prefer in ("concat", "auto") and concat_commits.exists() and concat_qna.exists():
        return ParquetSources("concat", [concat_commits], [concat_qna])
    if prefer in ("hf", "auto") and hf_commits and hf_qna:
        return ParquetSources("hf", hf_commits, hf_qna)
    if prefer in ("shards", "auto") and shard_commits and shard_qna:
        return ParquetSources("shards", shard_commits, shard_qna)

    raise SystemExit(f"No parquet sources found under {root} with prefer={prefer!r}.")


# ---------------------------------------------------------------------------
# Repo paths (match create_dataset/build_commit_parquet_db._repo_dir_for)
# ---------------------------------------------------------------------------


def repo_dir_for(repos_root: Path, repo_id: str) -> Path:
    parts = repo_id.split("/")
    if len(parts) == 2:
        return repos_root / parts[0] / parts[1]
    return repos_root / repo_id


def safe_repo_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id.replace("/", "__"))


# ---------------------------------------------------------------------------
# Git: full first-parent history + per-commit churn (numstat)
# ---------------------------------------------------------------------------

_COMMIT_HEADER = re.compile(r"^COMMIT ([0-9a-f]{40}) (.+)$")


def git_log_chronological_with_churn(
    repo_dir: Path,
    *,
    first_parent: bool = True,
    timeout: int = 600,
) -> List[Dict[str, Any]]:
    """Return oldest-first commits: sha, datetime, churn (added+deleted lines)."""
    if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
        return []

    cmd = ["git", "-C", str(repo_dir), "log", "--reverse"]
    if first_parent:
        cmd.append("--first-parent")
    cmd += ["--pretty=format:COMMIT %H %aI", "--numstat"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    if proc.returncode != 0:
        return []

    out: List[Dict[str, Any]] = []
    cur_sha: Optional[str] = None
    cur_ts: Optional[datetime] = None
    churn = 0

    def flush() -> None:
        nonlocal cur_sha, cur_ts, churn
        if cur_sha and cur_ts is not None:
            out.append({"sha": cur_sha, "datetime": cur_ts, "churn": churn})
        cur_sha, cur_ts = None, None
        churn = 0

    for raw in proc.stdout.splitlines():
        line = raw.rstrip("\n")
        m = _COMMIT_HEADER.match(line)
        if m:
            flush()
            cur_sha = m.group(1)
            try:
                cur_ts = datetime.fromisoformat(m.group(2).replace("Z", "+00:00"))
                if cur_ts.tzinfo is None:
                    cur_ts = cur_ts.replace(tzinfo=timezone.utc)
            except ValueError:
                cur_ts = None
            churn = 0
            continue
        if not line or cur_sha is None:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, b = parts[0], parts[1]
        if a == "-" or b == "-":
            continue
        try:
            churn += int(a) + int(b)
        except ValueError:
            continue

    flush()
    return out


# ---------------------------------------------------------------------------
# Parquet: QnA counts per (repo_id, commit_sha)
# ---------------------------------------------------------------------------


def _pair_shard_paths(commits_paths: Sequence[Path], qna_paths: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """Match ``{safe}.commits.parquet`` with ``{safe}.qna.parquet``."""
    q_by_stem: Dict[str, Path] = {
        p.name.replace(".qna.parquet", ""): p for p in qna_paths
    }
    pairs: List[Tuple[Path, Path]] = []
    for cp in commits_paths:
        stem = cp.name.replace(".commits.parquet", "")
        q = q_by_stem.get(stem)
        if q is not None:
            pairs.append((cp, q))
    return pairs


def load_qna_counts_per_commit(
    parquet_dir: Optional[str],
    commits_path: Optional[str],
    qna_path: Optional[str],
    prefer: str,
) -> Tuple[DefaultDict[Tuple[str, str], int], List[str]]:
    """Return (counts for (repo_id, sha), sorted repo_ids seen in parquet)."""
    src = resolve_parquet_sources(
        parquet_dir=parquet_dir,
        commits_path=commits_path,
        qna_path=qna_path,
        prefer=prefer,
    )

    counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    repos_set: set = set()

    def ingest_qna_table(tab) -> None:
        if tab.num_rows == 0:
            return
        rs = tab.column("repo_id").to_pylist()
        shas = tab.column("commit_sha").to_pylist()
        for r, s in zip(rs, shas):
            if not r or not s:
                continue
            counts[(str(r), str(s))] += 1
            repos_set.add(str(r))

    if src.source_kind == "concat":
        for qp in src.qna_paths:
            ingest_qna_table(pq.read_table(str(qp), columns=["repo_id", "commit_sha"]))
    elif src.source_kind == "shards":
        pairs = _pair_shard_paths(src.commits_paths, src.qna_paths)
        for _cp, qp in pairs:
            ingest_qna_table(pq.read_table(str(qp), columns=["repo_id", "commit_sha"]))
    else:
        # HF or mixed: read every qna file
        for qp in src.qna_paths:
            ingest_qna_table(pq.read_table(str(qp), columns=["repo_id", "commit_sha"]))

    return counts, sorted(repos_set)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _scale_sizes_churn(churn: List[int]) -> List[float]:
    """Map churn lines -> matplotlib scatter 's' (points**2)."""
    v = [math.log1p(max(int(c), 0)) for c in churn]
    if not v or max(v) <= 0:
        return [8.0] * len(churn)
    lo, hi = min(v), max(v)
    if hi <= lo:
        return [50.0] * len(churn)
    return [5.0 + 95.0 * ((x - lo) / (hi - lo + 1e-9)) for x in v]


def _scale_sizes_qna(n_qna: List[int]) -> List[float]:
    """Blue branch uses small constant; red scales with sqrt(n_qna)."""
    out: List[float] = []
    reds = [max(1, int(x)) for x in n_qna if int(x) > 0]
    if not reds:
        return [6.0] * len(n_qna)
    nr = [math.sqrt(x) for x in reds]
    lo, hi = min(nr), max(nr)
    for x in n_qna:
        xi = int(x)
        if xi <= 0:
            out.append(6.0)
        else:
            t = math.sqrt(max(xi, 1))
            if hi <= lo:
                out.append(40.0)
            else:
                u = (t - lo) / (hi - lo + 1e-9)
                out.append(15.0 + 120.0 * u)
    return out


def plot_global_timeline(
    *,
    repo_ids: List[str],
    series_per_repo: List[List[Dict[str, Any]]],
    qna_counts: DefaultDict[Tuple[str, str], int],
    out_path: Path,
    mode: str,
    title: str,
) -> None:
    """mode: 'churn' | 'qna'."""
    repo_ids = list(repo_ids)
    y_map = {rid: i for i, rid in enumerate(repo_ids)}
    xs: List[datetime] = []
    ys: List[float] = []
    churns: List[int] = []
    qnas: List[int] = []
    for rid, commits in zip(repo_ids, series_per_repo):
        yi = float(y_map[rid])
        for c in commits:
            sha = c["sha"]
            nq = int(qna_counts.get((rid, sha), 0))
            xs.append(c["datetime"])
            ys.append(yi)
            churns.append(int(c.get("churn", 0)))
            qnas.append(nq)

    if not xs:
        print("  (skip global plot: no commit points)", flush=True)
        return

    fig_h = max(6.0, min(120.0, 0.11 * len(repo_ids) + 4.0))
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=150)

    if mode == "churn":
        s = _scale_sizes_churn(churns)
        ax.scatter(xs, ys, s=s, c="#444444", alpha=0.55, linewidths=0, edgecolors="none")
    else:
        s = _scale_sizes_qna(qnas)
        c_arr = ["#d62728" if int(n) > 0 else "#1f77b4" for n in qnas]
        ax.scatter(xs, ys, s=s, c=c_arr, alpha=0.65, linewidths=0, edgecolors="none")

    ax.set_yticks(range(len(repo_ids)))
    ax.set_yticklabels(repo_ids, fontsize=max(4, min(9, 900 // max(len(repo_ids), 1))))
    ax.set_xlabel("Time (commit date, author ISO)")
    ax.set_ylabel("Repository")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_global_timeline_qna_with_stats(
    *,
    repo_ids: List[str],
    series_per_repo: List[List[Dict[str, Any]]],
    qna_counts: DefaultDict[Tuple[str, str], int],
    out_path: Path,
    title: str,
) -> None:
    """QnA timeline with per-repo stats aligned as right-side columns."""
    repo_ids = list(repo_ids)
    y_map = {rid: i for i, rid in enumerate(repo_ids)}
    xs: List[datetime] = []
    ys: List[float] = []
    qnas: List[int] = []
    per_repo_stats: List[Tuple[int, int, int, float, float]] = []

    for rid, commits in zip(repo_ids, series_per_repo):
        yi = float(y_map[rid])
        qnas_in_changed_commits: List[int] = []
        for c in commits:
            sha = c["sha"]
            nq = int(qna_counts.get((rid, sha), 0))
            xs.append(c["datetime"])
            ys.append(yi)
            qnas.append(nq)
            if nq > 0:
                qnas_in_changed_commits.append(nq)
        n_qna_commits = len(qnas_in_changed_commits)
        n_qnas_total = sum(qnas_in_changed_commits)
        mean_qnas = (n_qnas_total / n_qna_commits) if n_qna_commits else 0.0
        if n_qna_commits > 1:
            var_qnas = sum((n - mean_qnas) ** 2 for n in qnas_in_changed_commits) / n_qna_commits
            std_qnas = math.sqrt(var_qnas)
        else:
            std_qnas = 0.0
        per_repo_stats.append((len(commits), n_qna_commits, n_qnas_total, mean_qnas, std_qnas))

    if not xs:
        print("  (skip QnA stats SVG: no commit points)", flush=True)
        return

    fig_h = max(6.0, min(120.0, 0.11 * len(repo_ids) + 4.0))
    fig = plt.figure(figsize=(24, fig_h), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[14.5, 6.3], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1], sharey=ax)

    s = _scale_sizes_qna(qnas)
    c_arr = ["#d62728" if int(n) > 0 else "#1f77b4" for n in qnas]
    ax.scatter(xs, ys, s=s, c=c_arr, alpha=0.65, linewidths=0, edgecolors="none")

    ax.set_yticks(range(len(repo_ids)))
    ax.set_yticklabels(repo_ids, fontsize=max(4, min(9, 900 // max(len(repo_ids), 1))))
    ax.set_xlabel("Time (commit date, author ISO)")
    ax.set_ylabel("Repository")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(ax.get_ylim())
    ax_stats.set_xticks([])
    ax_stats.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax_stats.spines.values():
        spine.set_visible(False)

    row_font = max(3.8, min(8.0, 760 / max(len(repo_ids), 1)))
    header_font = max(6.0, min(10.0, row_font + 1.5))
    col_x = [0.03, 0.25, 0.47, 0.66, 0.84]
    headers = ["# commits", "# QnA\ncommits", "# QnAs", "mean\nQnA/commit", "std\nQnA/commit"]
    for x, h in zip(col_x, headers):
        ax_stats.text(
            x,
            1.01,
            h,
            transform=ax_stats.transAxes,
            ha="left",
            va="bottom",
            fontsize=header_font,
            fontweight="bold",
            linespacing=0.9,
        )

    for yi, (n_commits, n_qna_commits, n_qnas_total, mean_qnas, std_qnas) in enumerate(per_repo_stats):
        vals = [
            f"{n_commits:,}",
            f"{n_qna_commits:,}",
            f"{n_qnas_total:,}",
            f"{mean_qnas:.1f}",
            f"{std_qnas:.1f}",
        ]
        for x, val in zip(col_x, vals):
            ax_stats.text(
                x,
                yi,
                val,
                ha="left",
                va="center",
                fontsize=row_font,
                color="#222222",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_repo_histogram(
    repo_id: str,
    commits: List[Dict[str, Any]],
    qna_counts: DefaultDict[Tuple[str, str], int],
    out_path: Path,
) -> None:
    if not commits:
        return
    times = [c["datetime"] for c in commits]
    heights = [float(qna_counts.get((repo_id, c["sha"]), 0)) for c in commits]
    xnum = mdates.date2num(times)
    if len(xnum) == 1:
        widths = [1.0]
    else:
        gaps = [xnum[i + 1] - xnum[i] for i in range(len(xnum) - 1)]
        gaps_sorted = sorted(gaps)
        med = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else 1.0
        med = max(med, 1e-6)
        widths = [g * 0.85 for g in gaps] + [med * 0.85]
        widths = [max(w, 1e-4) for w in widths]

    fig, ax = plt.subplots(figsize=(14, 4.0), dpi=140)
    ax.bar(xnum, heights, width=widths, align="edge", color="#6baed6", edgecolor="#2171b5", linewidth=0.2)
    ax.set_title(f"{repo_id} — new QnA pairs per commit (full first-parent history)")
    ax.set_xlabel("Commit time (chronological)")
    ax.set_ylabel("New QnA pairs at commit")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_parquet = Path(
        os.environ.get("SCRATCH", str(Path.home() / "scratch"))
    ).expanduser() / "REPO_DATASET" / "commit_parquet"
    ap.add_argument("--parquet-dir", type=str, default=str(default_parquet))
    ap.add_argument(
        "--commits-file",
        type=str,
        default=None,
        help="Explicit commits.parquet (use with --qna-file; overrides --parquet-dir)",
    )
    ap.add_argument("--qna-file", type=str, default=None)
    ap.add_argument(
        "--prefer",
        choices=("auto", "concat", "shards", "hf"),
        default="auto",
        help="Parquet layout under parquet-dir",
    )
    ap.add_argument(
        "--repos-root",
        type=str,
        required=True,
        help="Root of cloned repos (author/repo subdirs or flat)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "analysis" / "output" / "commit_chronology_figs"),
    )
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument(
        "--no-first-parent",
        action="store_true",
        help="Use full git history instead of --first-parent",
    )
    ap.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Only write the two global timeline PNGs",
    )
    ap.add_argument("--git-timeout", type=int, default=600)
    args = ap.parse_args()

    if bool(args.commits_file) ^ bool(args.qna_file):
        raise SystemExit("Provide both --commits-file and --qna-file, or neither.")

    repos_root = Path(args.repos_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    hist_dir = out_dir / "histograms_per_repo"

    qna_counts, repo_ids_parquet = load_qna_counts_per_commit(
        parquet_dir=None if args.commits_file else args.parquet_dir,
        commits_path=args.commits_file,
        qna_path=args.qna_file,
        prefer=args.prefer,
    )
    if not repo_ids_parquet:
        raise SystemExit("No repo_id rows found in qna parquet — check paths.")

    repo_ids = list(repo_ids_parquet)
    if args.limit_repos is not None:
        repo_ids = repo_ids[: max(0, args.limit_repos)]

    first_parent = not args.no_first_parent
    series_per_repo: List[List[Dict[str, Any]]] = []
    skipped = 0
    for rid in repo_ids:
        rdir = repo_dir_for(repos_root, rid)
        hist = git_log_chronological_with_churn(
            rdir, first_parent=first_parent, timeout=args.git_timeout
        )
        if not hist:
            skipped += 1
            series_per_repo.append([])
            continue
        series_per_repo.append(hist)

    if skipped:
        print(f"  warning: {skipped} repos had empty/missing git history under {repos_root}", flush=True)

    plot_global_timeline(
        repo_ids=repo_ids,
        series_per_repo=series_per_repo,
        qna_counts=qna_counts,
        out_path=out_dir / "commits_timeline_churn.png",
        mode="churn",
        title="All commits (first-parent): dot area ∝ lines added+deleted (numstat)",
    )
    plot_global_timeline(
        repo_ids=repo_ids,
        series_per_repo=series_per_repo,
        qna_counts=qna_counts,
        out_path=out_dir / "commits_timeline_qna.png",
        mode="qna",
        title="Commits: blue = no new QnA rows; red area ∝ new QnA pairs (parquet)",
    )
    plot_global_timeline_qna_with_stats(
        repo_ids=repo_ids,
        series_per_repo=series_per_repo,
        qna_counts=qna_counts,
        out_path=out_dir / "commits_timeline_qna_with_stats.svg",
        title="Commits and QnA creation by repository: red area ∝ new QnA pairs",
    )

    if not args.skip_histograms:
        hist_dir.mkdir(parents=True, exist_ok=True)
        n_h = 0
        for rid, hist in zip(repo_ids, series_per_repo):
            if not hist:
                continue
            fn = hist_dir / f"{safe_repo_filename(rid)}.png"
            plot_repo_histogram(rid, hist, qna_counts, fn)
            n_h += 1
        print(f"  wrote {n_h} per-repo histograms under {hist_dir}", flush=True)


if __name__ == "__main__":
    main()
