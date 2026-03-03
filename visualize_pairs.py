#!/usr/bin/env python3
"""
Visualize Q&A pairs from create_qnas.py (QNA_HYPERNET.json).
Produces an interactive HTML report with:
  - Summary statistics (repos, assertion types, length distributions)
  - Browsable, syntax-highlighted prefix → target pairs
  - Filtering and search

Usage:
    python visualize_pairs.py [--repos-dir DIR] [--output report.html]   # all repos with QNA
    python visualize_pairs.py --qna path/to/QNA_HYPERNET.json [--output report.html]
    python visualize_pairs.py --repo owner/repo_name [--repos-dir DIR] [--output report.html]
"""

import argparse
import html
import json
import os
import statistics
from collections import Counter

# Default repos dir (must match create_qnas.py)
REPOSITORIES_DIR = "/home/lhotsko/scratch/REPO_DATASET/repositories"
QNA_HYPERNET = "QNA_HYPERNET.json"


def iter_repos_with_qna(repos_root: str):
    """Yield (author, repo_name, qna_path) for repos that have QNA_HYPERNET.json."""
    if not os.path.isdir(repos_root):
        return
    for author in os.listdir(repos_root):
        author_path = os.path.join(repos_root, author)
        if not os.path.isdir(author_path):
            continue
        for repo_name in os.listdir(author_path):
            repo_path = os.path.join(author_path, repo_name)
            qna_path = os.path.join(repo_path, QNA_HYPERNET)
            if os.path.isfile(qna_path):
                yield author, repo_name, qna_path


def load_qna_json(path: str) -> list[dict]:
    """Load pairs from a QNA_HYPERNET.json file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    repo = data.get("repo", "")
    # Normalize to flat records for visualize_pairs
    records = []
    for p in pairs:
        meta = p.get("metadata", {})
        records.append({
            "prefix": p.get("prefix", ""),
            "target": p.get("target", ""),
            "repo": meta.get("repo", repo),
            "assertion_type": p.get("assertion_type", "unknown"),
            "metadata": {
                "file": meta.get("file", ""),
                "function": meta.get("test_function", ""),
                "cut_line": meta.get("lineno", ""),
                "cut_kind": p.get("assertion_type", ""),
                "was_multiline": meta.get("was_multiline", False),
            },
        })
    return records


def load_data(path: str, max_lines: int | None = None) -> list[dict]:
    """Load from JSONL (legacy) or QNA JSON. For JSONL, max_lines limits."""
    if path.endswith(".json"):
        return load_qna_json(path)
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            records.append(json.loads(line))
    return records


def compute_stats(records: list[dict]) -> dict:
    stats = {}
    stats["total"] = len(records)

    # Repos
    repo_counts = Counter(r["repo"] for r in records)
    stats["num_repos"] = len(repo_counts)
    stats["top_repos"] = repo_counts.most_common(20)

    # Assertion types (from create_qnas)
    at_counts = Counter(r.get("assertion_type", "unknown") for r in records)
    stats["assertion_types"] = at_counts.most_common()

    # Prefix / target lengths (in characters and lines)
    prefix_chars = [len(r.get("prefix", "")) for r in records]
    target_chars = [len(r.get("target", "")) for r in records]
    prefix_lines = [r.get("prefix", "").count("\n") + 1 for r in records]
    target_lines = [r.get("target", "").count("\n") + 1 for r in records]

    stats["prefix_chars"] = {
        "min": min(prefix_chars), "max": max(prefix_chars),
        "mean": statistics.mean(prefix_chars), "median": statistics.median(prefix_chars),
    }
    stats["target_chars"] = {
        "min": min(target_chars), "max": max(target_chars),
        "mean": statistics.mean(target_chars), "median": statistics.median(target_chars),
    }
    stats["prefix_lines"] = {
        "min": min(prefix_lines), "max": max(prefix_lines),
        "mean": statistics.mean(prefix_lines), "median": statistics.median(prefix_lines),
    }
    stats["target_lines"] = {
        "min": min(target_lines), "max": max(target_lines),
        "mean": statistics.mean(target_lines), "median": statistics.median(target_lines),
    }

    # Multiline
    multiline = sum(1 for r in records if r.get("metadata", {}).get("was_multiline", False))
    stats["multiline_count"] = multiline
    stats["multiline_pct"] = 100.0 * multiline / len(records) if records else 0

    return stats


def make_bar(items: list, max_bar_width: int = 300) -> str:
    """Generate an HTML horizontal bar chart from (label, count) pairs."""
    if not items:
        return ""
    max_val = max(v for _, v in items)
    rows = []
    for label, count in items:
        width = int(max_bar_width * count / max_val) if max_val else 0
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{html.escape(str(label))}</span>'
            f'<div class="bar" style="width:{width}px"></div>'
            f'<span class="bar-value">{count}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def make_stat_table(d: dict) -> str:
    return (
        f'<table class="stat-table">'
        f'<tr><th>Min</th><th>Max</th><th>Mean</th><th>Median</th></tr>'
        f'<tr><td>{d["min"]}</td><td>{d["max"]}</td>'
        f'<td>{d["mean"]:.1f}</td><td>{d["median"]:.1f}</td></tr>'
        f'</table>'
    )


def generate_html(records: list[dict], stats: dict, max_display: int = 200) -> str:
    """Generate a self-contained HTML report."""

    display_records = records[:max_display]
    sample_cards = []
    for i, r in enumerate(display_records):
        meta = r.get("metadata", {})
        prefix_code = html.escape(r.get("prefix", ""))
        target_code = html.escape(r.get("target", ""))
        prefix_full = r.get("prefix", "")
        prefix_lines_list = prefix_full.split("\n")
        if len(prefix_lines_list) > 30:
            prefix_short = "\n".join(
                ["... (truncated, showing last 30 lines) ...", ""] + prefix_lines_list[-30:]
            )
        else:
            prefix_short = prefix_full

        card = f"""
        <div class="card" data-repo="{html.escape(r.get('repo',''))}"
             data-assertion-type="{html.escape(r.get('assertion_type',''))}"
             data-idx="{i}">
          <div class="card-header" onclick="toggleCard(this)">
            <span class="card-num">#{i+1}</span>
            <span class="card-repo">{html.escape(r.get('repo',''))}</span>
            <span class="card-file">{html.escape(meta.get('file',''))}</span>
            <span class="card-func">{html.escape(str(meta.get('function','') or ''))}</span>
            <span class="card-cut">{html.escape(r.get('assertion_type',''))} @ line {meta.get('cut_line','?')}</span>
            <span class="card-fw badge">{html.escape(r.get('assertion_type',''))}</span>
            <span class="card-toggle">&#9660;</span>
          </div>
          <div class="card-body" style="display:none;">
            <div class="pair-container">
              <div class="pair-section prefix-section">
                <div class="section-label">PREFIX (input)</div>
                <pre><code>{html.escape(prefix_short)}</code></pre>
                {"<details><summary>Show full prefix (" + str(len(prefix_lines_list)) + " lines)</summary><pre><code>" + prefix_code + "</code></pre></details>" if len(prefix_lines_list) > 30 else ""}
              </div>
              <div class="arrow">&#10142;</div>
              <div class="pair-section target-section">
                <div class="section-label">TARGET (expected output)</div>
                <pre><code>{target_code}</code></pre>
              </div>
            </div>
            <div class="meta-info">
              <span><b>Assertion type:</b> {html.escape(r.get('assertion_type',''))}</span>
              <span><b>Was multiline:</b> {meta.get('was_multiline', False)}</span>
              <span><b>Prefix:</b> {len(r.get('prefix',''))} chars, {r.get('prefix','').count(chr(10))+1} lines</span>
              <span><b>Target:</b> {len(r.get('target',''))} chars, {r.get('target','').count(chr(10))+1} lines</span>
            </div>
          </div>
        </div>
        """
        sample_cards.append(card)

    samples_html = "\n".join(sample_cards)

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Q&A Pairs Visualization — create_qnas</title>
<style>
  :root {{
    --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --prefix-bg: #1c2128; --target-bg: #0f1a0f;
    --target-border: #2ea04370; --prefix-border: #58a6ff40;
    --badge-bg: #1f6feb30; --badge-text: #58a6ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }}
  h1 {{ color: #f0f6fc; margin-bottom: 5px; font-size: 1.6em; }}
  h2 {{ color: var(--accent); margin: 25px 0 10px; font-size: 1.2em; border-bottom: 1px solid var(--border); padding-bottom: 5px; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 20px; font-size: 0.95em; }}

  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-bottom: 30px; }}
  .stats-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .stats-card h3 {{ color: var(--accent); font-size: 1em; margin-bottom: 10px; }}
  .stat-table {{ border-collapse: collapse; width: 100%; }}
  .stat-table th, .stat-table td {{ text-align: left; padding: 4px 10px; border-bottom: 1px solid var(--border); font-size: 0.9em; }}
  .stat-table th {{ color: var(--text-dim); font-weight: 600; }}

  .bar-row {{ display: flex; align-items: center; margin: 3px 0; font-size: 0.85em; }}
  .bar-label {{ width: 220px; text-align: right; padding-right: 10px; color: var(--text-dim);
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar {{ height: 16px; background: linear-gradient(90deg, #1f6feb, #58a6ff); border-radius: 3px; min-width: 2px; }}
  .bar-value {{ padding-left: 8px; color: var(--text); font-weight: 500; }}

  .filter-bar {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
                 padding: 12px 16px; margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .filter-bar label {{ color: var(--text-dim); font-size: 0.9em; }}
  .filter-bar select, .filter-bar input {{ background: var(--bg); color: var(--text); border: 1px solid var(--border);
                                           border-radius: 4px; padding: 5px 8px; font-size: 0.9em; }}

  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px;
           transition: border-color 0.2s; }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{ display: flex; align-items: center; gap: 10px; padding: 10px 16px; cursor: pointer;
                  flex-wrap: wrap; font-size: 0.88em; }}
  .card-num {{ color: var(--text-dim); font-weight: 700; min-width: 40px; }}
  .card-repo {{ color: var(--accent); font-weight: 600; }}
  .card-file {{ color: var(--text-dim); }}
  .card-func {{ color: #d2a8ff; }}
  .card-cut {{ color: var(--text-dim); font-size: 0.85em; }}
  .card-toggle {{ margin-left: auto; color: var(--text-dim); transition: transform 0.2s; }}
  .card.open .card-toggle {{ transform: rotate(180deg); }}
  .badge {{ background: var(--badge-bg); color: var(--badge-text); padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }}

  .card-body {{ padding: 0 16px 16px; }}
  .pair-container {{ display: flex; gap: 12px; align-items: flex-start; }}
  .arrow {{ font-size: 2em; color: var(--accent); padding-top: 30px; flex-shrink: 0; }}
  .pair-section {{ flex: 1; min-width: 0; }}
  .prefix-section pre {{ background: var(--prefix-bg); border: 1px solid var(--prefix-border); }}
  .target-section pre {{ background: var(--target-bg); border: 1px solid var(--target-border); }}
  .section-label {{ font-size: 0.8em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
                    margin-bottom: 6px; }}
  .prefix-section .section-label {{ color: #58a6ff; }}
  .target-section .section-label {{ color: #3fb950; }}
  pre {{ padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; line-height: 1.45;
         max-height: 400px; overflow-y: auto; }}
  code {{ font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace; }}
  details {{ margin-top: 6px; }}
  details summary {{ color: var(--accent); cursor: pointer; font-size: 0.85em; }}
  .meta-info {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 10px; font-size: 0.82em; color: var(--text-dim); }}

  @media (max-width: 900px) {{
    .pair-container {{ flex-direction: column; }}
    .arrow {{ transform: rotate(90deg); padding: 0; text-align: center; }}
  }}
</style>
</head>
<body>

<h1>Q&A Pairs Visualization</h1>
<p class="subtitle">From <b>create_qnas.py</b> &mdash; {stats['total']:,} pairs, {stats['num_repos']} repo(s)</p>

<h2>Summary Statistics</h2>
<div class="stats-grid">
  <div class="stats-card">
    <h3>Repositories ({stats['num_repos']} total)</h3>
    {make_bar(stats['top_repos'])}
  </div>
  <div class="stats-card">
    <h3>Assertion Types</h3>
    {make_bar(stats['assertion_types'])}
  </div>
  <div class="stats-card">
    <h3>Prefix Length (characters)</h3>
    {make_stat_table(stats['prefix_chars'])}
    <h3 style="margin-top:12px">Prefix Length (lines)</h3>
    {make_stat_table(stats['prefix_lines'])}
  </div>
  <div class="stats-card">
    <h3>Target Length (characters)</h3>
    {make_stat_table(stats['target_chars'])}
    <h3 style="margin-top:12px">Target Length (lines)</h3>
    {make_stat_table(stats['target_lines'])}
  </div>
  <div class="stats-card">
    <h3>Multiline Targets</h3>
    <p>{stats['multiline_count']:,} / {stats['total']:,} targets were multiline ({stats['multiline_pct']:.1f}%)</p>
  </div>
</div>

<h2>Training Pairs (showing {min(max_display, len(records)):,} of {stats['total']:,})</h2>

<div class="filter-bar">
  <label>Filter repo:</label>
  <select id="repoFilter" onchange="filterCards()">
    <option value="">All repos</option>
  </select>
  <label>Assertion type:</label>
  <select id="typeFilter" onchange="filterCards()">
    <option value="">All</option>
  </select>
  <label>Search:</label>
  <input id="searchBox" type="text" placeholder="Search in code..." oninput="filterCards()">
  <button onclick="expandAll()" style="background:var(--bg);color:var(--accent);border:1px solid var(--border);
          border-radius:4px;padding:5px 10px;cursor:pointer;">Expand All</button>
  <button onclick="collapseAll()" style="background:var(--bg);color:var(--accent);border:1px solid var(--border);
          border-radius:4px;padding:5px 10px;cursor:pointer;">Collapse All</button>
</div>

<div id="cards">
{samples_html}
</div>

<script>
const cards = document.querySelectorAll('.card');
const repos = new Set(), types = new Set();
cards.forEach(c => {{ repos.add(c.dataset.repo); types.add(c.dataset.assertionType); }});
const repoSel = document.getElementById('repoFilter');
[...repos].sort().forEach(r => {{ const o = document.createElement('option'); o.value = r; o.textContent = r; repoSel.appendChild(o); }});
const typeSel = document.getElementById('typeFilter');
[...types].sort().forEach(t => {{ const o = document.createElement('option'); o.value = t; o.textContent = t; typeSel.appendChild(o); }});

function toggleCard(header) {{
  const card = header.parentElement;
  const body = card.querySelector('.card-body');
  if (body.style.display === 'none') {{ body.style.display = 'block'; card.classList.add('open'); }}
  else {{ body.style.display = 'none'; card.classList.remove('open'); }}
}}

function filterCards() {{
  const repo = repoSel.value.toLowerCase();
  const type = typeSel.value.toLowerCase();
  const search = document.getElementById('searchBox').value.toLowerCase();
  cards.forEach(c => {{
    const matchRepo = !repo || c.dataset.repo.toLowerCase() === repo;
    const matchType = !type || c.dataset.assertionType.toLowerCase() === type;
    const matchSearch = !search || c.textContent.toLowerCase().includes(search);
    c.style.display = (matchRepo && matchType && matchSearch) ? '' : 'none';
  }});
}}

function expandAll() {{ cards.forEach(c => {{ c.querySelector('.card-body').style.display = 'block'; c.classList.add('open'); }}); }}
function collapseAll() {{ cards.forEach(c => {{ c.querySelector('.card-body').style.display = 'none'; c.classList.remove('open'); }}); }}
</script>

</body>
</html>"""
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Q&A pairs from create_qnas.py (QNA_HYPERNET.json)"
    )
    parser.add_argument(
        "--qna",
        type=str,
        default=None,
        help="Path to QNA_HYPERNET.json file",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repo as owner/repo_name (loads from repos-dir/owner/repo_name/QNA_HYPERNET.json)",
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        default=REPOSITORIES_DIR,
        help=f"Root of repositories (default: {REPOSITORIES_DIR})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max number of sample pairs to display (default: 200). Use 0 for all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all pairs (overrides --max-samples)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output HTML file path (default: /tmp/qna_report.html)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of repos when iterating over all (for testing)",
    )
    args = parser.parse_args()

    records = []
    if args.qna:
        path = args.qna
        if not os.path.isfile(path):
            print(f"Error: File not found: {path}")
            return 1
        print(f"Loading from {path} ...")
        records = load_data(path)
        print(f"  Loaded {len(records):,} pairs.")
    elif args.repo:
        parts = args.repo.split("/", 1)
        if len(parts) != 2:
            print("Error: --repo must be owner/repo_name")
            return 1
        author, repo_name = parts
        path = os.path.join(args.repos_dir, author, repo_name, QNA_HYPERNET)
        if not os.path.isfile(path):
            print(f"Error: QNA file not found: {path}")
            print("Run: python create_dataset/create_qnas.py --repo", args.repo)
            return 1
        print(f"Loading from {path} ...")
        records = load_data(path)
        print(f"  Loaded {len(records):,} pairs.")
    else:
        # Iterate over all repos with QNA
        if not os.path.isdir(args.repos_dir):
            print(f"Error: Repos dir not found: {args.repos_dir}")
            return 1
        repos_list = list(iter_repos_with_qna(args.repos_dir))
        if args.limit:
            repos_list = repos_list[: args.limit]
        print(f"Scanning {args.repos_dir} ... found {len(repos_list)} repos with QNA")
        if not repos_list:
            print("No repos with QNA_HYPERNET.json found.")
            return 0
        for author, repo_name, qna_path in repos_list:
            try:
                recs = load_data(qna_path)
                for r in recs:
                    r["repo"] = f"{author}/{repo_name}"
                    if "metadata" in r:
                        r["metadata"]["repo"] = f"{author}/{repo_name}"
                records.extend(recs)
            except Exception as e:
                print(f"  Warning: skipped {author}/{repo_name}: {e}")
        print(f"  Loaded {len(records):,} pairs from {len(repos_list)} repos.")

    if not records:
        print("No pairs to visualize.")
        return 0

    print("Computing statistics ...")
    stats = compute_stats(records)

    max_display = len(records) if (args.all or args.max_samples == 0) else args.max_samples
    print("Generating HTML report ...")
    report_html = generate_html(records, stats, max_display=max_display)

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join("/tmp", "qna_report.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Report saved to: {out_path}")
    print(f"Open in browser:  firefox {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
