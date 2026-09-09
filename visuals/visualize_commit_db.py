#!/usr/bin/env python3
"""
Interactive HTML report for commits_assertions.db (same spirit as visualize_pairs_context.py).

Shows what is stored per row:
  - Assertion mode (default): each assertion with prefix/target, plus tabs for the
    unified diff and commit SHA at that (repo_id, commit_index).
  - Commits-only: if assertions are empty or --commits-only, one card per commit row.

Usage:
    python visualize_commit_db.py -o db_report.html --limit 40
    python visualize_commit_db.py --db-path $SCRATCH/REPO_DATASET/commits_assertions.db -o report.html
    python visualize_commit_db.py --commits-only --limit-repos 3 --limit 30 -o commits.html

On cluster scratch (Lustre/NFS), SQLite may raise "locking protocol"; this script tries
read-only immutable URI first, then copies the DB to a temp dir if needed.
Use --no-local-copy to skip the copy (and fail with a manual cp hint) for huge DBs.

Large TEXT columns: use --max-column-chars (default 120000) so the job is not OOM-killed;
use --limit 5 --limit-repos 1 for a tiny HTML report.
"""

from __future__ import annotations

import argparse
import html as html_mod
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.data_utils import get_default_splits_dir


def esc(s: str) -> str:
    return html_mod.escape(str(s))


def truncate_code(text: str, max_lines: int = 40) -> Tuple[str, bool]:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text, False
    return "\n".join(lines[:max_lines]), True


def default_db_path() -> Path:
    return Path(get_default_splits_dir()).expanduser().resolve() / "commits_assertions.db"


def _probe_conn(conn: sqlite3.Connection) -> None:
    conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()


def _copy_sqlite_files(src_db: Path, dst_db: Path) -> None:
    shutil.copy2(src_db, dst_db)
    for ext in ("-wal", "-shm"):
        side = Path(str(src_db) + ext)
        if side.is_file():
            shutil.copy2(side, Path(str(dst_db) + ext))


def open_db_for_visualize(
    db_path: Path,
    *,
    read_only_uri: bool,
    immutable_uri: bool,
    no_local_copy: bool,
) -> Tuple[sqlite3.Connection, Optional[Path]]:
    """Open SQLite for read-only inspection; handles cluster FS locking issues.

    Returns (connection, temp_dir). If temp_dir is set, remove it with
    shutil.rmtree after closing the connection (local copy fallback).
    """
    abs_path = db_path.resolve()
    posix = abs_path.as_posix()

    def try_open(
        label: str, factory: Callable[[], sqlite3.Connection]
    ) -> Optional[sqlite3.Connection]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = factory()
            _probe_conn(conn)
            print(f"Opened database ({label})", flush=True)
            return conn
        except sqlite3.Error:
            if conn is not None:
                conn.close()
            return None

    strategies: List[Tuple[str, Callable[[], sqlite3.Connection]]] = []

    if immutable_uri:
        strategies.append(
            (
                "read-only URI + immutable=1",
                lambda: sqlite3.connect(
                    f"file:{posix}?mode=ro&immutable=1", uri=True, timeout=120.0
                ),
            )
        )
    elif read_only_uri:
        strategies.append(
            (
                "read-only URI",
                lambda: sqlite3.connect(
                    f"file:{posix}?mode=ro", uri=True, timeout=120.0
                ),
            )
        )
        strategies.append(
            (
                "read-write (default locking)",
                lambda: sqlite3.connect(str(abs_path), timeout=120.0),
            )
        )
    else:
        strategies.extend(
            [
                (
                    "read-only URI + immutable=1 (no POSIX locks; OK for static snapshot)",
                    lambda: sqlite3.connect(
                        f"file:{posix}?mode=ro&immutable=1", uri=True, timeout=120.0
                    ),
                ),
                (
                    "read-only URI",
                    lambda: sqlite3.connect(
                        f"file:{posix}?mode=ro", uri=True, timeout=120.0
                    ),
                ),
                (
                    "read-write",
                    lambda: sqlite3.connect(str(abs_path), timeout=120.0),
                ),
            ]
        )

    for label, factory in strategies:
        c = try_open(label, factory)
        if c is not None:
            return c, None

    if no_local_copy:
        print(
            "Could not open database (locking protocol / unsupported locks on this "
            "filesystem). Re-run with a copy on local disk, e.g.\n"
            "  cp \"$DB\" /tmp/commits_assertions.db && "
            "python visualize_commit_db.py --db-path /tmp/commits_assertions.db",
            file=sys.stderr,
        )
        sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="commit_db_viz_"))
    tmp_db = tmp_dir / "commits_assertions.db"
    try:
        print(
            f"Copying database to temp dir for local open (cluster FS lock workaround): "
            f"{tmp_db}",
            flush=True,
        )
        _copy_sqlite_files(abs_path, tmp_db)
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Failed to copy database: {exc}", file=sys.stderr)
        sys.exit(1)

    c = try_open(
        "local temp copy",
        lambda: sqlite3.connect(str(tmp_db), timeout=120.0),
    )
    if c is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Could not open temp copy of database.", file=sys.stderr)
        sys.exit(1)
    return c, tmp_dir


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _commits_has_column(conn: sqlite3.Connection, column: str) -> bool:
    rows = conn.execute("PRAGMA table_info(commits)").fetchall()
    return any(r[1] == column for r in rows)


def db_stats(conn: sqlite3.Connection, repo_ids: List[str]) -> Dict[str, int]:
    cur = conn.cursor()
    has_assert = _has_table(conn, "assertions")
    if not repo_ids:
        row_c = cur.execute("SELECT COUNT(*) FROM commits").fetchone()
        r_c = cur.execute("SELECT COUNT(DISTINCT repo_id) FROM commits").fetchone()
        a_c = (0,)
        if has_assert:
            a_c = cur.execute("SELECT COUNT(*) FROM assertions").fetchone()
        return {
            "repos": int(r_c[0] or 0),
            "commits": int(row_c[0] or 0),
            "assertions": int(a_c[0] or 0),
        }
    ph = ",".join("?" * len(repo_ids))
    row_c = cur.execute(
        f"SELECT COUNT(*) FROM commits WHERE repo_id IN ({ph})", repo_ids
    ).fetchone()
    a_c = (0,)
    if has_assert:
        a_c = cur.execute(
            f"SELECT COUNT(*) FROM assertions WHERE repo_id IN ({ph})", repo_ids
        ).fetchone()
    return {
        "repos": len(repo_ids),
        "commits": int(row_c[0] or 0),
        "assertions": int(a_c[0] or 0),
    }


def load_repo_ids(conn: sqlite3.Connection, limit_repos: Optional[int]) -> List[str]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT repo_id FROM commits ORDER BY repo_id"
    ).fetchall()
    out = [r[0] for r in rows]
    if limit_repos is not None and limit_repos > 0:
        out = out[:limit_repos]
    return out


def _sql_truncation_suffix(full_len: int, cap: int) -> str:
    return (
        f"\n\n[… truncated for visualization: showing first {cap} of {full_len} "
        f"characters; raise --max-column-chars or use 0 to load full text (may use "
        f"a lot of RAM) …]"
    )


def load_assertion_items(
    conn: sqlite3.Connection,
    repo_ids: List[str],
    limit: Optional[int],
    max_column_chars: Optional[int],
) -> List[Dict[str, Any]]:
    if not repo_ids:
        return []
    ph = ",".join("?" * len(repo_ids))
    cur = conn.cursor()
    cap = max_column_chars if max_column_chars and max_column_chars > 0 else None
    items: List[Dict[str, Any]] = []
    sha_sel = "c.commit_sha" if _commits_has_column(conn, "commit_sha") else "''"
    if cap is None:
        q = f"""
            SELECT a.rowid, a.repo_id, a.commit_index,
                   a.assertion_prefix, a.assertion_target,
                   c.production_code_diff, {sha_sel}
            FROM assertions a
            JOIN commits c
              ON a.repo_id = c.repo_id AND a.commit_index = c.commit_index
            WHERE a.repo_id IN ({ph})
            ORDER BY a.repo_id, a.commit_index, a.rowid
        """
        for r in cur.execute(q, repo_ids).fetchall():
            items.append(
                {
                    "mode": "assertion",
                    "rowid": r[0],
                    "repo_id": r[1],
                    "commit_index": r[2],
                    "prefix": r[3] or "",
                    "target": r[4] or "",
                    "diff": r[5] or "",
                    "commit_sha": (r[6] or "") if r[6] is not None else "",
                }
            )
    else:
        q = f"""
            SELECT a.rowid, a.repo_id, a.commit_index,
                   a.assertion_prefix, a.assertion_target,
                   substr(c.production_code_diff, 1, ?),
                   length(c.production_code_diff),
                   {sha_sel}
            FROM assertions a
            JOIN commits c
              ON a.repo_id = c.repo_id AND a.commit_index = c.commit_index
            WHERE a.repo_id IN ({ph})
            ORDER BY a.repo_id, a.commit_index, a.rowid
        """
        for r in cur.execute(q, (cap, *repo_ids)).fetchall():
            diff_t, diff_len = r[5], r[6]
            diff_t = diff_t or ""
            if diff_len is not None and diff_len > cap:
                diff_t = diff_t + _sql_truncation_suffix(diff_len, cap)
            items.append(
                {
                    "mode": "assertion",
                    "rowid": r[0],
                    "repo_id": r[1],
                    "commit_index": r[2],
                    "prefix": r[3] or "",
                    "target": r[4] or "",
                    "diff": diff_t,
                    "commit_sha": (r[7] or "") if r[7] is not None else "",
                }
            )
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def load_commit_items(
    conn: sqlite3.Connection,
    repo_ids: List[str],
    limit: Optional[int],
    max_column_chars: Optional[int],
) -> List[Dict[str, Any]]:
    if not repo_ids:
        return []
    ph = ",".join("?" * len(repo_ids))
    cur = conn.cursor()
    cap = max_column_chars if max_column_chars and max_column_chars > 0 else None
    sha_sel = "commit_sha" if _commits_has_column(conn, "commit_sha") else "''"
    if cap is None:
        q = f"""
            SELECT repo_id, commit_index, production_code_diff, {sha_sel}
            FROM commits
            WHERE repo_id IN ({ph})
            ORDER BY repo_id, commit_index
        """
        rows = cur.execute(q, repo_ids).fetchall()
        items = []
        for row in rows:
            items.append(
                {
                    "mode": "commit",
                    "repo_id": row[0],
                    "commit_index": row[1],
                    "diff": row[2] or "",
                    "commit_sha": (row[3] or "") if row[3] is not None else "",
                }
            )
    else:
        q = f"""
            SELECT repo_id, commit_index,
                   substr(production_code_diff, 1, ?),
                   length(production_code_diff),
                   {sha_sel}
            FROM commits
            WHERE repo_id IN ({ph})
            ORDER BY repo_id, commit_index
        """
        rows = cur.execute(q, (cap, *repo_ids)).fetchall()
        items = []
        for row in rows:
            repo_id, ci, diff_t, diff_len, sha = row
            diff_t = diff_t or ""
            if diff_len is not None and diff_len > cap:
                diff_t = diff_t + _sql_truncation_suffix(diff_len, cap)
            items.append(
                {
                    "mode": "commit",
                    "repo_id": repo_id,
                    "commit_index": ci,
                    "diff": diff_t,
                    "commit_sha": (sha or "") if sha is not None else "",
                }
            )
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def _diff_tab_content(diff_text: str, idx: int) -> str:
    if not diff_text.strip():
        return f'<div class="context-block diff-block"><div class="ctx-label diff-label">COMMIT DIFF</div><p class="no-data">Empty diff (e.g. root commit or no path changes).</p></div>'
    dt, was_t = truncate_code(diff_text, max_lines=45)
    trunc_note = " (truncated)" if was_t else ""
    extra = ""
    if was_t:
        extra = f'<details><summary>Full diff ({diff_text.count(chr(10))+1} lines)</summary><pre><code>{esc(diff_text)}</code></pre></details>'
    return f"""
    <div class="context-block diff-block">
      <div class="ctx-label diff-label">COMMIT DIFF{trunc_note} ({len(diff_text)} chars, {diff_text.count(chr(10))+1} lines)</div>
      <pre><code>{esc(dt)}</code></pre>
      {extra}
    </div>"""


def _commit_sha_content(sha: str, idx: int) -> str:
    if not sha.strip():
        return '<div class="context-block code-block"><div class="ctx-label code-label">COMMIT SHA</div><p class="no-data">No SHA recorded.</p></div>'
    return f"""
    <div class="context-block code-block">
      <div class="ctx-label code-label">COMMIT SHA</div>
      <pre><code>{esc(sha)}</code></pre>
    </div>"""


def _assertion_pair_html(prefix: str, target: str) -> str:
    prefix_lines = prefix.split("\n")
    ctx_window = 20
    if len(prefix_lines) > ctx_window:
        visible = prefix_lines[-ctx_window:]
        ellipsis = f"... ({len(prefix_lines) - ctx_window} lines above) ..."
    else:
        visible = prefix_lines
        ellipsis = ""

    cut_parts = []
    if ellipsis:
        cut_parts.append(esc(ellipsis))
    for line in visible[:-1]:
        cut_parts.append(esc(line))
    last_line = visible[-1] if visible else ""
    cut_parts.append(
        esc(last_line)
        + '<span class="cut-marker">|</span>'
        + '<span class="target-hl">' + esc(target) + "</span>"
    )
    cut_html = "\n".join(cut_parts)

    return f"""
            <div class="section-label">Code context (cut point)</div>
            <pre><code>{cut_html}</code></pre>

            <div class="pair-row">
              <div class="pair-section prefix-section">
                <div class="section-label prefix-label">PREFIX ({len(prefix_lines)} lines, {len(prefix)} chars)</div>
                <details><summary>Show full prefix</summary><pre><code>{esc(prefix)}</code></pre></details>
              </div>
              <div class="arrow">&#10142;</div>
              <div class="pair-section target-section">
                <div class="section-label target-label">TARGET</div>
                <pre><code>{esc(target)}</code></pre>
              </div>
            </div>"""


def generate_html(
    items: List[Dict[str, Any]],
    stats: Dict[str, int],
    title_suffix: str,
) -> str:
    cards = []
    for i, item in enumerate(items):
        repo = item["repo_id"]
        ci = item["commit_index"]
        diff_text = item["diff"]
        sha = item.get("commit_sha", "")
        mode = item["mode"]

        if mode == "assertion":
            header_extra = f'<span class="card-line">commit {ci}</span><span class="badge">assertion rowid {item["rowid"]}</span>'
            body_top = _assertion_pair_html(item["prefix"], item["target"])
        else:
            header_extra = f'<span class="card-line">commit {ci}</span><span class="badge">commit row</span>'
            body_top = f"""
            <p class="commits-only-note">No assertion row; showing stored <code>production_code_diff</code> and <code>commit_sha</code> for this commit index.</p>
            <div class="meta-row">
              <span>diff: {len(diff_text)} chars</span>
              <span>sha: {esc(sha[:12])}</span>
            </div>"""

        diff_block = _diff_tab_content(diff_text, i)
        sha_block = _commit_sha_content(sha, i)

        card = f"""
        <div class="card" data-repo="{esc(repo)}" data-idx="{i}">
          <div class="card-header" onclick="toggleCard(this)">
            <span class="card-num">#{i+1}</span>
            <span class="card-repo">{esc(repo)}</span>
            {header_extra}
            <span class="card-toggle">&#9660;</span>
          </div>
          <div class="card-body" style="display:none;">
            {body_top}

            <h3 class="context-heading">Database columns</h3>
            <div class="tab-bar">
              <button class="tab active" onclick="showTab(this,'diff-{i}')">Commit diff</button>
              <button class="tab" onclick="showTab(this,'sha-{i}')">Commit SHA</button>
            </div>
            <div class="tab-content" id="diff-{i}">{diff_block}</div>
            <div class="tab-content" id="sha-{i}" style="display:none;">{sha_block}</div>
          </div>
        </div>"""
        cards.append(card)

    cards_html = "\n".join(cards)
    n_total = len(items)
    mode_label = stats.get("view_mode", "mixed")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Commit DB visualization — {esc(title_suffix)}</title>
<style>
  :root {{
    --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --diff-color: #f0883e; --code-color: #79c0ff;
    --prefix-bg: #1c2128; --target-bg: #0f1a0f; --icl-color: #3fb950;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }}
  h1 {{ color: #f0f6fc; margin-bottom: 4px; font-size: 1.5em; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 20px; }}
  h3.context-heading {{ color: var(--accent); margin: 18px 0 8px; font-size: 1em;
                         border-top: 1px solid var(--border); padding-top: 14px; }}

  .filter-bar {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
                 padding: 10px 16px; margin-bottom: 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .filter-bar label {{ color: var(--text-dim); font-size: 0.9em; }}
  .filter-bar select, .filter-bar input {{ background: var(--bg); color: var(--text); border: 1px solid var(--border);
                                           border-radius: 4px; padding: 5px 8px; font-size: 0.9em; }}
  button {{ background: var(--bg); color: var(--accent); border: 1px solid var(--border);
            border-radius: 4px; padding: 5px 10px; cursor: pointer; font-size: 0.85em; }}
  button:hover {{ border-color: var(--accent); }}

  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{ display: flex; align-items: center; gap: 10px; padding: 10px 16px; cursor: pointer;
                  flex-wrap: wrap; font-size: 0.88em; }}
  .card-num {{ color: var(--text-dim); font-weight: 700; min-width: 36px; }}
  .card-repo {{ color: var(--accent); font-weight: 600; }}
  .card-line {{ color: var(--text-dim); font-size: 0.85em; }}
  .badge {{ background: #1f6feb30; color: var(--accent); padding: 2px 8px; border-radius: 10px; font-size: 0.78em; }}
  .card-toggle {{ margin-left: auto; color: var(--text-dim); }}
  .card.open .card-toggle {{ transform: rotate(180deg); }}
  .card-body {{ padding: 0 16px 16px; }}

  .pair-row {{ display: flex; gap: 12px; align-items: flex-start; margin-top: 12px; }}
  .arrow {{ font-size: 2em; color: var(--accent); padding-top: 24px; flex-shrink: 0; }}
  .pair-section {{ flex: 1; min-width: 0; }}
  .prefix-section pre {{ background: var(--prefix-bg); border: 1px solid #58a6ff30; }}
  .target-section pre {{ background: var(--target-bg); border: 1px solid #3fb95040; }}
  .section-label {{ font-size: 0.78em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 5px; }}
  .prefix-label {{ color: var(--accent); }}
  .target-label {{ color: var(--icl-color); }}

  pre {{ padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 0.78em; line-height: 1.35;
         max-height: 400px; overflow-y: auto; }}
  code {{ font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace; }}
  details {{ margin-top: 5px; }} details summary {{ color: var(--accent); cursor: pointer; font-size: 0.83em; }}
  .cut-marker {{ color: #f85149; font-weight: 900; font-size: 1.2em; }}
  .target-hl {{ background: #2ea04340; color: var(--icl-color); border-radius: 3px; padding: 0 2px; }}
  .no-data {{ color: var(--text-dim); font-style: italic; font-size: 0.85em; padding: 8px 0; }}

  .tab-bar {{ display: flex; gap: 4px; margin-bottom: 0; }}
  .tab {{ padding: 6px 16px; font-size: 0.85em; border-bottom: 2px solid transparent; border-radius: 4px 4px 0 0; }}
  .tab.active {{ border-bottom-color: var(--accent); color: #f0f6fc; font-weight: 600; }}
  .tab-content {{ border: 1px solid var(--border); border-radius: 0 0 6px 6px; padding: 12px; }}

  .context-block {{ margin-bottom: 14px; }}
  .ctx-label {{ font-size: 0.78em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }}
  .diff-label {{ color: var(--diff-color); }}
  .code-label {{ color: var(--code-color); }}
  .diff-block pre {{ border-left: 3px solid var(--diff-color); }}
  .code-block pre {{ border-left: 3px solid var(--code-color); }}

  .commits-only-note {{ color: var(--text-dim); font-size: 0.88em; margin-bottom: 10px; }}
  .meta-row {{ display: flex; gap: 16px; color: var(--text-dim); font-size: 0.85em; margin-bottom: 12px; }}

  @media (max-width: 900px) {{
    .pair-row {{ flex-direction: column; }}
    .arrow {{ transform: rotate(90deg); padding: 0; text-align: center; }}
  }}
</style>
</head>
<body>
<h1>Commits + assertions DB</h1>
<p class="subtitle">
  <b>{esc(title_suffix)}</b> &mdash;
  DB scope: {stats["repos"]} repos, {stats["commits"]} commit rows, {stats["assertions"]} assertion rows
  &mdash; <b>{n_total}</b> cards ({esc(mode_label)})
</p>

<div class="filter-bar">
  <label>Repo:</label>
  <select id="repoFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Search:</label>
  <input id="searchBox" type="text" placeholder="Search..." oninput="filterCards()">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>

<div id="cards">{cards_html}</div>

<script>
const cards = document.querySelectorAll('.card');
const repos = new Set();
cards.forEach(c => repos.add(c.dataset.repo));
const sel = document.getElementById('repoFilter');
[...repos].sort().forEach(r => {{ const o = document.createElement('option'); o.value = r; o.textContent = r; sel.appendChild(o); }});

function toggleCard(h) {{
  const c = h.parentElement, b = c.querySelector('.card-body');
  if (b.style.display === 'none') {{ b.style.display = 'block'; c.classList.add('open'); }}
  else {{ b.style.display = 'none'; c.classList.remove('open'); }}
}}
function filterCards() {{
  const repo = sel.value.toLowerCase(), q = document.getElementById('searchBox').value.toLowerCase();
  cards.forEach(c => {{
    c.style.display = (!repo || c.dataset.repo.toLowerCase() === repo) && (!q || c.textContent.toLowerCase().includes(q)) ? '' : 'none';
  }});
}}
function expandAll() {{ cards.forEach(c => {{ c.querySelector('.card-body').style.display='block'; c.classList.add('open'); }}); }}
function collapseAll() {{ cards.forEach(c => {{ c.querySelector('.card-body').style.display='none'; c.classList.remove('open'); }}); }}

function showTab(btn, id) {{
  const card = btn.closest('.card-body');
  card.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
  card.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).style.display = 'block';
}}
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize commits_assertions.db in HTML (like visualize_pairs_context.py)"
    )
    ap.add_argument(
        "--db-path",
        type=str,
        default=str(default_db_path()),
        help="Path to commits_assertions.db",
    )
    ap.add_argument("--limit", type=int, default=50, help="Max cards in report")
    ap.add_argument(
        "--max-column-chars",
        type=int,
        default=120_000,
        help="Max characters loaded per TEXT column from SQLite (diff); avoids OOM on "
        "large diffs. Use 0 for no limit (may use huge RAM / get killed).",
    )
    ap.add_argument(
        "--limit-repos",
        type=int,
        default=None,
        help="Only first N repos (by name) that appear in commits table",
    )
    ap.add_argument(
        "--commits-only",
        action="store_true",
        help="Show one card per commit row (ignore assertions table)",
    )
    ap.add_argument(
        "--read-only-uri",
        action="store_true",
        help="Open with SQLite URI file:...?mode=ro (lighter locking; use with a static copy if needed)",
    )
    ap.add_argument(
        "--immutable-uri",
        action="store_true",
        help="Only try read-only URI with immutable=1 (fail if that does not work)",
    )
    ap.add_argument(
        "--no-local-copy",
        action="store_true",
        help="Do not copy DB to temp dir if open fails (exit with a hint instead)",
    )
    ap.add_argument("-o", "--output", type=str, default=None)
    args = ap.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn: Optional[sqlite3.Connection] = None
    tmp_viz_dir: Optional[Path] = None
    cap: Optional[int] = args.max_column_chars if args.max_column_chars > 0 else None
    try:
        conn, tmp_viz_dir = open_db_for_visualize(
            db_path,
            read_only_uri=args.read_only_uri,
            immutable_uri=args.immutable_uri,
            no_local_copy=args.no_local_copy,
        )
        repo_ids = load_repo_ids(conn, args.limit_repos)
        stats = db_stats(conn, repo_ids)
        n_assert = stats["assertions"]
        assertions_table = _has_table(conn, "assertions")
        if args.commits_only or n_assert == 0 or not assertions_table:
            items = load_commit_items(conn, repo_ids, args.limit, cap)
            if args.commits_only:
                view_mode = "commits only (--commits-only)"
            elif not assertions_table:
                view_mode = "commits only (no assertions table)"
            else:
                view_mode = "commits only (no assertion rows in DB)"
        else:
            items = load_assertion_items(conn, repo_ids, args.limit, cap)
            view_mode = "assertions + joined commit snapshot"

        stats["view_mode"] = view_mode
        title_suffix = db_path.name
        html_out = generate_html(items, stats, title_suffix)
    finally:
        if conn is not None:
            conn.close()
        if tmp_viz_dir is not None:
            shutil.rmtree(tmp_viz_dir, ignore_errors=True)

    out_path = args.output or f"db_report_{db_path.stem}.html"
    Path(out_path).write_text(html_out, encoding="utf-8")
    print(f"Wrote {len(items)} cards to {out_path}")
    print(f"  Mode: {view_mode}")
    print(f"  Repos in filter: {len(repo_ids)}, DB totals (filtered): commits={stats['commits']}, assertions={stats['assertions']}")
    if cap is not None:
        print(f"  max_column_chars={cap} (use 0 to disable cap; risk OOM)")


if __name__ == "__main__":
    main()
