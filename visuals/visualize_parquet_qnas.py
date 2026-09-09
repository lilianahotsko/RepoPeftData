#!/usr/bin/env python3
"""Visualize QnA rows from the commit-Parquet/HF dataset as interactive HTML.

Examples:
    python visualize_parquet_qnas.py \
        --data-dir $SCRATCH/REPO_DATASET/commit_parquet_hf \
        --output report_commit_parquet_qnas.html \
        --max-samples 300

    python visualize_parquet_qnas.py --event-type modified --repo psf/requests
"""

from __future__ import annotations

import argparse
import html
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pyarrow is required: {exc}") from exc


QNA_COLUMNS = [
    "repo_id",
    "cross_repo_split",
    "commit_index",
    "commit_sha",
    "in_repo_split",
    "test_file",
    "file_split",
    "split_group",
    "lineno",
    "col_offset",
    "assertion_type",
    "test_function",
    "prefix",
    "target",
    "assertion_event_type",
    "assertion_event_id",
    "assertion_anchor",
    "old_target",
    "old_lineno",
    "old_col_offset",
]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def qna_paths(data_dir: Path, cross_split: Optional[str]) -> List[Path]:
    qna_dir = data_dir / "qna"
    if qna_dir.is_dir():
        if cross_split:
            path = qna_dir / f"{cross_split}.parquet"
            if not path.exists():
                raise SystemExit(f"QnA split not found: {path}")
            return [path]
        return sorted(qna_dir.glob("*.parquet"))

    single = data_dir / "qna_pairs.parquet"
    if single.exists():
        return [single]
    raise SystemExit(
        f"No QnA parquet found under {data_dir}; expected qna/*.parquet or qna_pairs.parquet."
    )


def table_columns(table) -> Dict[str, List[Any]]:
    return {name: table.column(name).to_pylist() for name in table.column_names}


def matches_filters(
    row: Dict[str, Any],
    *,
    repo: Optional[str],
    event_type: Optional[str],
    file_split: Optional[str],
    in_repo_split: Optional[str],
) -> bool:
    if repo and row.get("repo_id") != repo:
        return False
    if event_type and row.get("assertion_event_type") != event_type:
        return False
    if file_split and row.get("file_split") != file_split:
        return False
    if in_repo_split and row.get("in_repo_split") != in_repo_split:
        return False
    return True


def load_sample_records(
    paths: Iterable[Path],
    *,
    max_samples: int,
    repo: Optional[str],
    event_type: Optional[str],
    file_split: Optional[str],
    in_repo_split: Optional[str],
    batch_size: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        schema = pq.read_schema(path)
        columns = [c for c in QNA_COLUMNS if c in schema.names]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            cols = table_columns(batch)
            n = batch.num_rows
            for i in range(n):
                row = {name: cols[name][i] for name in cols}
                if not matches_filters(
                    row,
                    repo=repo,
                    event_type=event_type,
                    file_split=file_split,
                    in_repo_split=in_repo_split,
                ):
                    continue
                records.append(row)
                if len(records) >= max_samples:
                    return records
    return records


def count_matching_rows(
    paths: Iterable[Path],
    *,
    repo: Optional[str],
    event_type: Optional[str],
    file_split: Optional[str],
    in_repo_split: Optional[str],
    batch_size: int,
) -> Dict[str, Any]:
    counts = Counter()
    repos = Counter()
    assertion_types = Counter()
    event_types = Counter()
    file_splits = Counter()
    in_repo_splits = Counter()
    qna_by_commit = Counter()
    n_total_seen = 0
    n_matched = 0

    columns = [
        "repo_id",
        "cross_repo_split",
        "commit_index",
        "commit_sha",
        "in_repo_split",
        "test_file",
        "file_split",
        "assertion_type",
        "assertion_event_type",
    ]
    for path in paths:
        schema = pq.read_schema(path)
        present = [c for c in columns if c in schema.names]
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=present):
            cols = table_columns(batch)
            n_total_seen += batch.num_rows
            for i in range(batch.num_rows):
                row = {name: cols[name][i] for name in cols}
                if not matches_filters(
                    row,
                    repo=repo,
                    event_type=event_type,
                    file_split=file_split,
                    in_repo_split=in_repo_split,
                ):
                    continue
                n_matched += 1
                rid = str(row.get("repo_id") or "")
                sha = str(row.get("commit_sha") or "")
                counts[str(row.get("cross_repo_split") or "")] += 1
                repos[rid] += 1
                assertion_types[str(row.get("assertion_type") or "unknown")] += 1
                event_types[str(row.get("assertion_event_type") or "unknown")] += 1
                file_splits[str(row.get("file_split") or "unknown")] += 1
                in_repo_splits[str(row.get("in_repo_split") or "unknown")] += 1
                qna_by_commit[(rid, sha)] += 1

    qna_per_commit = sorted(qna_by_commit.values())
    return {
        "total_rows_seen": n_total_seen,
        "matched_rows": n_matched,
        "matched_repos": len(repos),
        "matched_commits": len(qna_by_commit),
        "cross_repo_splits": counts.most_common(),
        "top_repos": repos.most_common(20),
        "assertion_types": assertion_types.most_common(),
        "event_types": event_types.most_common(),
        "file_splits": file_splits.most_common(),
        "in_repo_splits": in_repo_splits.most_common(),
        "qna_per_commit": numeric_summary(qna_per_commit),
    }


def numeric_summary(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def sample_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    prefix_chars = [len(str(r.get("prefix") or "")) for r in records]
    target_chars = [len(str(r.get("target") or "")) for r in records]
    prefix_lines = [str(r.get("prefix") or "").count("\n") + 1 for r in records]
    target_lines = [str(r.get("target") or "").count("\n") + 1 for r in records]
    return {
        "prefix_chars": numeric_summary(prefix_chars),
        "target_chars": numeric_summary(target_chars),
        "prefix_lines": numeric_summary(prefix_lines),
        "target_lines": numeric_summary(target_lines),
    }


def make_bar(items: List[tuple], max_width: int = 260) -> str:
    if not items:
        return "<p class='muted'>No rows.</p>"
    max_val = max(int(v) for _k, v in items) or 1
    rows = []
    for label, count in items:
        width = max(2, int(max_width * int(count) / max_val))
        rows.append(
            f"<div class='bar-row'><span class='bar-label'>{esc(label)}</span>"
            f"<span class='bar' style='width:{width}px'></span>"
            f"<span class='bar-value'>{int(count):,}</span></div>"
        )
    return "\n".join(rows)


def stat_table(d: Dict[str, float]) -> str:
    return (
        "<table class='stat-table'>"
        "<tr><th>Min</th><th>Max</th><th>Mean</th><th>Median</th><th>Std</th></tr>"
        f"<tr><td>{d['min']:.0f}</td><td>{d['max']:.0f}</td>"
        f"<td>{d['mean']:.1f}</td><td>{d['median']:.1f}</td><td>{d['std']:.1f}</td></tr>"
        "</table>"
    )


def compact_code(text: str, keep_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= keep_lines:
        return text
    omitted = len(lines) - keep_lines
    return f"# ... ({omitted} lines omitted) ...\n" + "\n".join(lines[-keep_lines:])


def generate_html(
    records: List[Dict[str, Any]],
    global_stats: Dict[str, Any],
    sample_summary: Dict[str, Any],
    *,
    data_dir: Path,
    max_prefix_lines: int,
) -> str:
    cards = []
    for i, r in enumerate(records, 1):
        prefix = str(r.get("prefix") or "")
        target = str(r.get("target") or "")
        old_target = str(r.get("old_target") or "")
        event_type = str(r.get("assertion_event_type") or "unknown")
        repo = str(r.get("repo_id") or "")
        file_path = str(r.get("test_file") or "")
        function = str(r.get("test_function") or "")
        commit_sha = str(r.get("commit_sha") or "")
        short_sha = commit_sha[:12]
        prefix_view = compact_code(prefix, max_prefix_lines)
        old_target_html = ""
        if old_target:
            old_target_html = (
                "<div class='pair-section old-section'>"
                "<div class='section-label'>OLD TARGET (before modification)</div>"
                f"<pre><code>{esc(old_target)}</code></pre></div>"
            )

        cards.append(f"""
<div class="card" data-repo="{esc(repo)}" data-event="{esc(event_type)}"
     data-file-split="{esc(r.get('file_split') or '')}" data-assertion-type="{esc(r.get('assertion_type') or '')}">
  <div class="card-header" onclick="toggleCard(this)">
    <span class="card-num">#{i}</span>
    <span class="card-repo">{esc(repo)}</span>
    <span class="badge event-{esc(event_type)}">{esc(event_type)}</span>
    <span class="badge">{esc(r.get('cross_repo_split') or '')}</span>
    <span class="badge file-split">file:{esc(r.get('file_split') or '')}</span>
    <span class="card-file">{esc(file_path)}</span>
    <span class="card-cut">{esc(r.get('assertion_type') or '')} @ line {esc(r.get('lineno') or '?')}</span>
    <span class="card-toggle">&#9660;</span>
  </div>
  <div class="card-body" style="display:none;">
    <div class="meta-info">
      <span><b>Commit:</b> <code>{esc(short_sha)}</code></span>
      <span><b>Commit index:</b> {esc(r.get('commit_index') or '')}</span>
      <span><b>In-repo split:</b> {esc(r.get('in_repo_split') or '')}</span>
      <span><b>Function:</b> {esc(function or 'N/A')}</span>
      <span><b>Event id:</b> <code>{esc(r.get('assertion_event_id') or '')}</code></span>
      <span><b>Anchor:</b> <code>{esc(r.get('assertion_anchor') or '')}</code></span>
    </div>
    <div class="pair-container">
      <div class="pair-section prefix-section">
        <div class="section-label">PREFIX (model input, showing last {max_prefix_lines} lines)</div>
        <pre><code>{esc(prefix_view)}</code></pre>
        <details><summary>Show full prefix ({prefix.count(chr(10)) + 1} lines)</summary><pre><code>{esc(prefix)}</code></pre></details>
      </div>
      <div class="pair-section target-section">
        <div class="section-label">TARGET (created QnA answer)</div>
        <pre><code>{esc(target)}</code></pre>
      </div>
      {old_target_html}
    </div>
  </div>
</div>
""")

    html_cards = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Commit-Parquet QnA Visualization</title>
<style>
:root {{
  --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9;
  --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --orange:#f0883e;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system, Segoe UI, sans-serif; padding:20px; line-height:1.45; }}
h1 {{ margin:0 0 4px; font-size:1.65rem; }}
h2 {{ color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:6px; margin-top:26px; }}
.subtitle,.muted {{ color:var(--muted); }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px; }}
.stats-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; }}
.stats-card h3 {{ color:var(--accent); margin:0 0 10px; font-size:1rem; }}
.bar-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:.86rem; }}
.bar-label {{ width:185px; text-align:right; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.bar {{ display:inline-block; height:15px; background:linear-gradient(90deg,#1f6feb,#58a6ff); border-radius:3px; }}
.bar-value {{ font-weight:600; }}
.stat-table {{ width:100%; border-collapse:collapse; }}
.stat-table th,.stat-table td {{ border-bottom:1px solid var(--border); padding:5px 8px; text-align:left; font-size:.9rem; }}
.filter-bar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; margin:12px 0; }}
select,input,button {{ background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:5px 8px; }}
button {{ cursor:pointer; color:var(--accent); }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; margin-bottom:8px; }}
.card:hover {{ border-color:var(--accent); }}
.card-header {{ display:flex; flex-wrap:wrap; align-items:center; gap:9px; padding:10px 14px; cursor:pointer; font-size:.88rem; }}
.card-num {{ color:var(--muted); font-weight:700; min-width:38px; }}
.card-repo {{ color:var(--accent); font-weight:700; }}
.card-file {{ color:var(--muted); }}
.card-cut {{ color:#d2a8ff; }}
.card-toggle {{ margin-left:auto; color:var(--muted); }}
.badge {{ background:#1f6feb30; color:#58a6ff; border-radius:999px; padding:2px 8px; font-size:.78rem; }}
.event-added {{ background:#23863640; color:var(--green); }}
.event-modified {{ background:#9e6a0340; color:var(--orange); }}
.file-split {{ color:#d2a8ff; background:#8957e530; }}
.card-body {{ padding:0 14px 14px; }}
.meta-info {{ display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:.84rem; margin:4px 0 12px; }}
.pair-container {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(260px,1fr) minmax(220px,.8fr); gap:12px; align-items:start; }}
.pair-section {{ min-width:0; }}
.section-label {{ font-size:.78rem; font-weight:800; letter-spacing:.04em; text-transform:uppercase; margin-bottom:6px; }}
.prefix-section .section-label {{ color:var(--accent); }}
.target-section .section-label {{ color:var(--green); }}
.old-section .section-label {{ color:var(--orange); }}
pre {{ background:#1c2128; border:1px solid var(--border); border-radius:6px; padding:12px; overflow:auto; max-height:420px; font-size:.82rem; }}
code {{ font-family:'Fira Code','JetBrains Mono',Consolas,monospace; }}
details summary {{ cursor:pointer; color:var(--accent); margin-top:6px; }}
@media (max-width:1100px) {{ .pair-container {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>Commit-Parquet QnA Visualization</h1>
<p class="subtitle">Dataset: <code>{esc(data_dir)}</code>. Showing {len(records):,} sampled rows from {global_stats['matched_rows']:,} matching QnAs.</p>

<h2>Summary</h2>
<div class="stats-grid">
  <div class="stats-card"><h3>Totals</h3>
    <p><b>Matched QnAs:</b> {global_stats['matched_rows']:,}</p>
    <p><b>Repos:</b> {global_stats['matched_repos']:,}</p>
    <p><b>Commits with QnAs:</b> {global_stats['matched_commits']:,}</p>
  </div>
  <div class="stats-card"><h3>Event Types</h3>{make_bar(global_stats['event_types'])}</div>
  <div class="stats-card"><h3>Cross-Repo Splits</h3>{make_bar(global_stats['cross_repo_splits'])}</div>
  <div class="stats-card"><h3>File Splits</h3>{make_bar(global_stats['file_splits'])}</div>
  <div class="stats-card"><h3>Assertion Types</h3>{make_bar(global_stats['assertion_types'])}</div>
  <div class="stats-card"><h3>Top Repositories</h3>{make_bar(global_stats['top_repos'])}</div>
  <div class="stats-card"><h3>QnAs per Commit</h3>{stat_table(global_stats['qna_per_commit'])}</div>
  <div class="stats-card"><h3>Sample Prefix/Target Lengths</h3>
    <p><b>Prefix chars</b></p>{stat_table(sample_summary['prefix_chars'])}
    <p><b>Target chars</b></p>{stat_table(sample_summary['target_chars'])}
  </div>
</div>

<h2>Sample QnA Rows</h2>
<div class="filter-bar">
  <label>Repo</label><select id="repoFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Event</label><select id="eventFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>File split</label><select id="fileSplitFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Search</label><input id="searchBox" type="text" oninput="filterCards()" placeholder="repo, file, target, code...">
  <button onclick="expandAll()">Expand all shown</button>
  <button onclick="collapseAll()">Collapse all</button>
</div>
<div id="cards">{html_cards}</div>

<script>
const cards = [...document.querySelectorAll('.card')];
function fillSelect(id, attr) {{
  const values = [...new Set(cards.map(c => c.dataset[attr]).filter(Boolean))].sort();
  const sel = document.getElementById(id);
  values.forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }});
}}
fillSelect('repoFilter', 'repo');
fillSelect('eventFilter', 'event');
fillSelect('fileSplitFilter', 'fileSplit');
function toggleCard(header) {{
  const card = header.parentElement;
  const body = card.querySelector('.card-body');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
}}
function matches(c) {{
  const repo = document.getElementById('repoFilter').value.toLowerCase();
  const event = document.getElementById('eventFilter').value.toLowerCase();
  const fileSplit = document.getElementById('fileSplitFilter').value.toLowerCase();
  const search = document.getElementById('searchBox').value.toLowerCase();
  return (!repo || c.dataset.repo.toLowerCase() === repo)
    && (!event || c.dataset.event.toLowerCase() === event)
    && (!fileSplit || c.dataset.fileSplit.toLowerCase() === fileSplit)
    && (!search || c.textContent.toLowerCase().includes(search));
}}
function filterCards() {{ cards.forEach(c => c.style.display = matches(c) ? '' : 'none'); }}
function expandAll() {{ cards.forEach(c => {{ if (matches(c)) c.querySelector('.card-body').style.display = 'block'; }}); }}
function collapseAll() {{ cards.forEach(c => c.querySelector('.card-body').style.display = 'none'); }}
</script>
</body>
</html>
"""


def main() -> int:
    default_data = Path(os.environ.get("SCRATCH", str(Path.home() / "scratch"))) / "REPO_DATASET" / "commit_parquet_hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--cross-split", choices=["train", "cr_val", "cr_test"], default=None)
    parser.add_argument("--repo", default=None, help="Only visualize one repo, e.g. owner/name")
    parser.add_argument("--event-type", choices=["added", "modified"], default=None)
    parser.add_argument("--file-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--in-repo-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-prefix-lines", type=int, default=60)
    parser.add_argument("--output", "-o", type=Path, default=Path("report_commit_parquet_qnas.html"))
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    paths = qna_paths(data_dir, args.cross_split)
    print(f"Reading QnA parquet from {data_dir}")
    print(f"Files: {len(paths)}")

    global_stats = count_matching_rows(
        paths,
        repo=args.repo,
        event_type=args.event_type,
        file_split=args.file_split,
        in_repo_split=args.in_repo_split,
        batch_size=args.batch_size,
    )
    print(f"Matched QnAs: {global_stats['matched_rows']:,}")

    records = load_sample_records(
        paths,
        max_samples=max(1, args.max_samples),
        repo=args.repo,
        event_type=args.event_type,
        file_split=args.file_split,
        in_repo_split=args.in_repo_split,
        batch_size=args.batch_size,
    )
    if not records:
        print("No matching QnA rows to visualize.")
        return 0

    report = generate_html(
        records,
        global_stats,
        sample_stats(records),
        data_dir=data_dir,
        max_prefix_lines=args.max_prefix_lines,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
