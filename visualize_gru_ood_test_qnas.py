#!/usr/bin/env python3
"""Visualize the OOD (ood_test) QnAs with per-method scores AND predictions.

This is the OOD counterpart of ``visualize_gru_ir_test_qnas.py``. The data
source is the unified OOD parquet (one row per QnA) and the per-method
scores come from the v2 sharded evaluators' ``raw_samples`` lists, just
like the IR-test viewer. The novelty here is that each prediction card
shows BOTH the raw model output and the post-processed string (the one
actually fed into the EM / EditSim / CodeBLEU metrics by
``evaluation.metrics.compute_metrics``).

Three-step workflow:

  # 1. produce the keys file (no GPU) -- this also writes the HTML report
  #    without prediction text yet, useful as a sanity check:
  python visualize_gru_ood_test_qnas.py \\
      --write-selected-keys $SCRATCH/OOD_TEST_PRED_DUMP/selected_keys.jsonl

  # 2. submit the prediction-dump SLURM array (one job per method):
  KEYS_FILE=$SCRATCH/OOD_TEST_PRED_DUMP/selected_keys.jsonl \\
      OUT_DIR=$SCRATCH/OOD_TEST_PRED_DUMP \\
      sbatch --array=0-4 scripts/slurm/dump_ood_test_predictions.sh

  # 3. rebuild the report with prediction text + DRC/RAG contexts attached:
  python visualize_gru_ood_test_qnas.py \\
      --predictions-dir $SCRATCH/OOD_TEST_PRED_DUMP \\
      --output report_gru_ood_test_qnas.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

# Ensure ``evaluation`` is importable when the script is run from the repo root.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from evaluation.metrics import (  # noqa: E402  (path tweak above is intentional)
    postprocess_prediction,
    strip_comments,
)


QNA_COLUMNS = [
    "repo_id", "cross_repo_split", "in_repo_split", "commit_index",
    "commit_sha", "test_file", "lineno", "col_offset",
    "assertion_type", "test_function", "prefix", "target",
]

DEFAULT_METHODS: Dict[str, Dict[str, str]] = {
    # label  ->  shard dir + filename prefix used by the v2 evaluators.
    "code2lora":     {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_STATIC_EVAL_V2/h100_v2_ood_run5",
                      "prefix": "static_v2"},
    "code2lora_gru": {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_ood",
                      "prefix": "gru_v2"},
    "rag":           {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/rag_h100_v2_prefix256_ood",
                      "prefix": "baseline_rag"},
    "drc":           {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/drc_h100_v2_prefix256_ood",
                      "prefix": "baseline_drc"},
    "text2lora":     {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/text2lora_h100_v2_full7_ood_sharded",
                      "prefix": "baseline_text2lora"},
    "pretrained":    {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/pretrained_h100_v2_ood",
                      "prefix": "baseline_pretrained"},
    "slora":         {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/slora_h100_v2_anchor_ood",
                      "prefix": "baseline_slora"},
    "doc2lora":      {"dir": "/scratch/lhotsko/TRAINING_CHECKPOINTS/BASELINES_V2/doc2lora_h100_v2_ood",
                      "prefix": "baseline_doc2lora"},
}


@dataclass
class QnaScore:
    em: float = float("nan")
    edit: float = float("nan")
    cb: float = float("nan")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def load_method_scores(eval_dir: Path, prefix: str, suite: str
                       ) -> Dict[Tuple[str, str], List[QnaScore]]:
    """Return {(repo, sha): [QnaScore, ...]} for this method."""
    out: Dict[Tuple[str, str], List[QnaScore]] = {}
    shards = sorted(eval_dir.glob(f"{prefix}_{suite}_shard*of*.json"))
    if not shards:
        # Fall back to non-sharded layout used by the local prediction-dump jobs.
        shards = sorted(eval_dir.glob(f"{prefix}_{suite}.json"))
    if not shards:
        print(f"  [warn] no shards under {eval_dir} matching {prefix}_{suite}_shard*")
        return out
    for sp in shards:
        try:
            d = json.loads(sp.read_text())
        except Exception as e:
            print(f"  [warn] failed to read {sp}: {e}")
            continue
        pc = d.get("per_commit", [])
        rs = d.get("raw_samples", {}) or {}
        em_list = rs.get("exact_match", [])
        ed_list = rs.get("edit_similarity", [])
        cb_list = rs.get("code_bleu", [])
        offset = 0
        for rec in pc:
            n = int(rec.get("n_qnas", 0))
            key = (rec.get("repo_id"), rec.get("commit_sha"))
            group: List[QnaScore] = []
            for j in range(n):
                k = offset + j
                group.append(QnaScore(
                    em=float(em_list[k]) if k < len(em_list) else float("nan"),
                    edit=float(ed_list[k]) if k < len(ed_list) else float("nan"),
                    cb=float(cb_list[k]) if k < len(cb_list) else float("nan"),
                ))
            out[key] = group
            offset += n
    return out


_META_COLS = [
    "repo_id", "commit_sha", "commit_index", "test_file",
    "assertion_type", "lineno",
]


def scan_groups(path: Path, batch_size: int
                ) -> Tuple[Dict[Tuple[str, str], int],
                           Dict[Tuple[str, str], int],
                           Counter, Counter, Counter]:
    """First pass: count QnAs per (repo, commit) and tally stats. No
    prefix/target loaded -- keeps memory low even for 500k+ row parquets.
    Returns (counts, commit_indices, assertion_types, repos,
    cross_repo_splits).
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    commit_index: Dict[Tuple[str, str], int] = {}
    assertion_types: Counter = Counter()
    repos: Counter = Counter()
    cross_repo_splits: Counter = Counter()
    schema = pq.read_schema(path)
    cols = [c for c in _META_COLS + ["cross_repo_split"] if c in schema.names]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = {c: batch.column(c).to_pylist() for c in batch.column_names}
        n = batch.num_rows
        for i in range(n):
            key = (d["repo_id"][i], d["commit_sha"][i])
            counts[key] += 1
            if key not in commit_index:
                commit_index[key] = int(d.get("commit_index", [0] * n)[i] or 0)
            assertion_types[d.get("assertion_type", [None] * n)[i] or "unknown"] += 1
            repos[d["repo_id"][i]] += 1
            cross_repo_splits[d.get("cross_repo_split", [None] * n)[i] or "unknown"] += 1
    return counts, commit_index, assertion_types, repos, cross_repo_splits


def load_selected_rows(path: Path, batch_size: int,
                       wanted: Dict[Tuple[str, str], List[int]]
                       ) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Second pass: load full row text only for the requested
    (repo, sha, qna_pos) coordinates.
    """
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    schema = pq.read_schema(path)
    cols = [c for c in QNA_COLUMNS if c in schema.names]
    pf = pq.ParquetFile(path)
    seen: Dict[Tuple[str, str], int] = defaultdict(int)
    wanted_keys = set(wanted.keys())
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = {c: batch.column(c).to_pylist() for c in batch.column_names}
        n = batch.num_rows
        for i in range(n):
            key = (d["repo_id"][i], d["commit_sha"][i])
            if key not in wanted_keys:
                continue
            pos = seen[key]
            seen[key] += 1
            if pos in wanted[key]:
                out[(key[0], key[1], pos)] = {c: d[c][i] for c in d}
    return out


def numeric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    return {
        "min": float(min(values)), "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def make_bar(items: List[Tuple[str, int]], max_width: int = 260) -> str:
    if not items:
        return "<p class='muted'>No rows.</p>"
    mx = max(int(v) for _, v in items) or 1
    return "\n".join(
        f"<div class='bar-row'><span class='bar-label'>{esc(k)}</span>"
        f"<span class='bar' style='width:{max(2, int(max_width*v/mx))}px'></span>"
        f"<span class='bar-value'>{int(v):,}</span></div>"
        for k, v in items
    )


def stat_table(d: Dict[str, float]) -> str:
    return (
        "<table class='stat-table'>"
        "<tr><th>Min</th><th>Max</th><th>Mean</th><th>Median</th><th>Std</th></tr>"
        f"<tr><td>{d['min']:.2f}</td><td>{d['max']:.2f}</td>"
        f"<td>{d['mean']:.2f}</td><td>{d['median']:.2f}</td><td>{d['std']:.2f}</td></tr>"
        "</table>"
    )


def compact_code(text: str, keep_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= keep_lines:
        return text
    omitted = len(lines) - keep_lines
    return f"# ... ({omitted} lines omitted) ...\n" + "\n".join(lines[-keep_lines:])


def fmt_score(s: Optional[QnaScore]) -> str:
    if s is None:
        return ("<span class='score-cell missing' title='no score for this QnA'>"
                "&mdash;</span>")
    em_cls = "ok" if s.em >= 0.5 else "ko"
    icon = "&#10003;" if s.em >= 0.5 else "&#10007;"
    return (
        f"<span class='score-cell {em_cls}' "
        f"title='EM={s.em:.0f}  EditSim={s.edit:.3f}  CodeBLEU={s.cb:.3f}'>"
        f"{icon} "
        f"<span class='score-sub'>{s.edit:.2f}/{s.cb:.2f}</span>"
        f"</span>"
    )


def render_pred_block(method: str, s: Optional[QnaScore],
                      rec: Dict[str, Any], target: str) -> str:
    """Render one per-method prediction block including:
    - the augmented context that was injected (RAG / DRC etc., as a
      collapsed <details>);
    - the RAW prediction text exactly as emitted by the model;
    - the POST-PROCESSED prediction string (same string used to score EM /
      EditSim / CodeBLEU);
    - the cleaned target for direct visual diffing.
    """
    raw_pred = str(rec.get("prediction") or "")
    aug_prompt = str(rec.get("augmented_prompt") or "")
    base_prefix = str(rec.get("prefix") or "")

    ctx_html = ""
    if aug_prompt and aug_prompt != base_prefix:
        added = aug_prompt
        if base_prefix and aug_prompt.endswith(base_prefix):
            added = aug_prompt[:-len(base_prefix)].rstrip("\n")
        if added:
            ctx_html = (
                f"<details class='ctx-block'><summary>Injected context "
                f"({added.count(chr(10)) + 1} lines, "
                f"{len(added):,} chars)</summary>"
                f"<pre><code>{esc(added)}</code></pre></details>"
            )

    pp_pred = postprocess_prediction(raw_pred, target)
    target_clean = strip_comments(target).strip()
    pp_matches = pp_pred.strip() == target_clean.strip()
    pp_changed = pp_pred != raw_pred

    em_cls = "ok" if (s and s.em >= 0.5) else "ko"
    status_label = "EM" if (s and s.em >= 0.5) else "no-EM"

    # Highlight whether postprocessing flipped the match outcome.
    pp_flag = ""
    if pp_changed:
        pp_flag = (" <span class='pp-flag changed' title='postprocessing "
                   "modified the raw output'>postprocessed</span>")
    pp_match_cls = "ok" if pp_matches else "ko"
    pp_match_lbl = "matches target" if pp_matches else "differs from target"

    return (
        f"<div class='pred-block'>"
        f"<div class='pred-header'>"
        f"<span class='pred-method'>{esc(method)}</span>"
        f"<span class='pred-status {em_cls}'>{status_label}</span>"
        f"<span class='pred-status {pp_match_cls}' "
        f"title='compares post-processed pred vs strip_comments(target)'>"
        f"{pp_match_lbl}</span>"
        f"{pp_flag}"
        f"</div>"
        f"{ctx_html}"
        f"<div class='pred-pair'>"
        f"  <div class='pred-col'>"
        f"    <div class='pred-col-label'>Raw model output</div>"
        f"    <pre class='pred-text'><code>"
        f"{esc(raw_pred) if raw_pred else '<em>(empty)</em>'}"
        f"</code></pre>"
        f"  </div>"
        f"  <div class='pred-col'>"
        f"    <div class='pred-col-label'>Post-processed (scored)</div>"
        f"    <pre class='pred-text pp'><code>"
        f"{esc(pp_pred) if pp_pred else '<em>(empty)</em>'}"
        f"</code></pre>"
        f"  </div>"
        f"</div>"
        f"</div>"
    )


def render_card(idx: int, row: Dict[str, Any], qna_pos: int,
                method_scores: Dict[str, Optional[QnaScore]],
                method_preds: Dict[str, Optional[Dict[str, Any]]],
                max_prefix_lines: int, suite: str) -> str:
    prefix = str(row.get("prefix") or "")
    target = str(row.get("target") or "")
    repo = str(row.get("repo_id") or "")
    file_path = str(row.get("test_file") or "")
    function = str(row.get("test_function") or "")
    commit_sha = str(row.get("commit_sha") or "")
    cross_repo_split = str(row.get("cross_repo_split") or "")
    short_sha = commit_sha[:12]

    prefix_view = compact_code(prefix, max_prefix_lines)
    target_clean = strip_comments(target).strip()

    method_html_parts: List[str] = []
    pred_block_parts: List[str] = []
    n_correct = 0
    n_total = 0
    for m, s in method_scores.items():
        if s is not None:
            n_total += 1
            if s.em >= 0.5:
                n_correct += 1
        method_html_parts.append(
            f"<div class='m-row'><span class='m-name'>{esc(m)}</span>{fmt_score(s)}</div>"
        )
        rec = method_preds.get(m)
        if rec is not None:
            pred_block_parts.append(render_pred_block(m, s, rec, target))
    method_html = "\n".join(method_html_parts)
    preds_html = ("\n".join(pred_block_parts)) if pred_block_parts else ""

    return f"""
<div class="card" data-repo="{esc(repo)}"
     data-assertion-type="{esc(row.get('assertion_type') or '')}"
     data-cross-split="{esc(cross_repo_split)}"
     data-correct="{n_correct}" data-scored="{n_total}">
  <div class="card-header" onclick="toggleCard(this)">
    <span class="card-num">#{idx}</span>
    <span class="card-repo">{esc(repo)}</span>
    <span class="badge">{esc(suite)}</span>
    <span class="badge cross-split">cross:{esc(cross_repo_split)}</span>
    <span class="card-file">{esc(file_path)}</span>
    <span class="card-cut">{esc(row.get('assertion_type') or '')} @ line {esc(row.get('lineno') or '?')}</span>
    <span class="score-summary">{n_correct}/{n_total} methods &#10003;</span>
    <span class="card-toggle">&#9660;</span>
  </div>
  <div class="card-body" style="display:none;">
    <div class="meta-info">
      <span><b>Commit:</b> <code>{esc(short_sha)}</code></span>
      <span><b>Commit index:</b> {esc(row.get('commit_index') or '')}</span>
      <span><b>QnA in commit:</b> {qna_pos}</span>
      <span><b>Function:</b> {esc(function or 'N/A')}</span>
    </div>
    <div class="pair-container">
      <div class="pair-section prefix-section">
        <div class="section-label">PREFIX (model input, last {max_prefix_lines} lines)</div>
        <pre><code>{esc(prefix_view)}</code></pre>
        <details><summary>Show full prefix ({prefix.count(chr(10)) + 1} lines)</summary><pre><code>{esc(prefix)}</code></pre></details>
      </div>
      <div class="pair-section target-section">
        <div class="section-label">EXPECTED TARGET</div>
        <pre><code>{esc(target)}</code></pre>
        {('<div class="section-label" style="margin-top:8px;">TARGET (cleaned)</div><pre class="pp"><code>' + esc(target_clean) + '</code></pre>') if target_clean != target.strip() else ''}
      </div>
      <div class="pair-section methods-section">
        <div class="section-label">PER-METHOD SCORES</div>
        <div class="m-list">{method_html}</div>
        <p class="muted score-help">
          &#10003; = exact match; sub-numbers show EditSim/CodeBLEU. Hover for raw values.
        </p>
      </div>
    </div>
    {('<div class="preds-grid"><div class="section-label">PER-METHOD PREDICTIONS &amp; CONTEXTS (raw / post-processed)</div>' + preds_html + '</div>') if preds_html else ''}
  </div>
</div>
"""


def generate_html(records: List[Tuple[int, Dict[str, Any], int,
                                       Dict[str, Optional[QnaScore]],
                                       Dict[str, Optional[Dict[str, Any]]]]],
                  *,
                  global_stats: Dict[str, Any],
                  method_summaries: Dict[str, Dict[str, float]],
                  qna_parquet: Path,
                  max_prefix_lines: int,
                  suite: str) -> str:
    cards_html = "\n".join(
        render_card(i, r, pos, m, p, max_prefix_lines, suite)
        for i, r, pos, m, p in records
    )
    methods = list(method_summaries.keys())
    method_summary_html = "".join(
        f"<div class='stats-card'><h3>{esc(m)}</h3>"
        f"<p><b>Scored QnAs:</b> {method_summaries[m]['n']:.0f}</p>"
        f"<p><b>EM:</b> {100*method_summaries[m]['em']:.2f}% "
        f"<span class='muted'>"
        f"(EditSim {method_summaries[m]['edit']:.3f}, "
        f"CodeBLEU {method_summaries[m]['cb']:.3f})</span></p>"
        f"</div>"
        for m in methods
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OOD Dataset &middot; {suite} &middot; per-method predictions</title>
<style>
:root {{
  --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9;
  --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149;
  --orange:#f0883e;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,Segoe UI,sans-serif; padding:20px; line-height:1.45; }}
h1 {{ margin:0 0 4px; font-size:1.65rem; }}
h2 {{ color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:6px; margin-top:26px; }}
.subtitle,.muted {{ color:var(--muted); }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:14px; }}
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
.cross-split {{ color:#d2a8ff; background:#8957e530; }}
.score-summary {{ font-weight:600; color:var(--muted); margin-left:8px; }}
.card-body {{ padding:0 14px 14px; }}
.meta-info {{ display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:.84rem; margin:4px 0 12px; }}
.pair-container {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(220px,.8fr) minmax(250px,.9fr); gap:12px; align-items:start; }}
.pair-section {{ min-width:0; }}
.section-label {{ font-size:.78rem; font-weight:800; letter-spacing:.04em; text-transform:uppercase; margin-bottom:6px; }}
.prefix-section .section-label {{ color:var(--accent); }}
.target-section .section-label {{ color:var(--green); }}
.methods-section .section-label {{ color:var(--orange); }}
pre {{ background:#1c2128; border:1px solid var(--border); border-radius:6px; padding:12px; overflow:auto; max-height:420px; font-size:.82rem; }}
pre.pp {{ border-left:3px solid var(--green); }}
code {{ font-family:'Fira Code','JetBrains Mono',Consolas,monospace; }}
details summary {{ cursor:pointer; color:var(--accent); margin-top:6px; }}
.m-list {{ display:flex; flex-direction:column; gap:5px; }}
.m-row {{ display:flex; align-items:center; gap:10px; background:#1c2128; border:1px solid var(--border); border-radius:5px; padding:6px 9px; }}
.m-name {{ flex:1; font-weight:600; color:var(--accent); font-size:.86rem; }}
.score-cell {{ font-family:'Fira Code',monospace; font-size:.85rem; padding:2px 7px; border-radius:4px; min-width:75px; text-align:right; }}
.score-cell.ok {{ background:#23863640; color:var(--green); }}
.score-cell.ko {{ background:#f8514922; color:var(--red); }}
.score-cell.missing {{ color:var(--muted); }}
.score-sub {{ color:var(--muted); font-size:.75rem; margin-left:4px; }}
.score-help {{ font-size:.78rem; margin-top:6px; }}
.preds-grid {{ margin-top:14px; border-top:1px dashed var(--border); padding-top:10px; display:flex; flex-direction:column; gap:10px; }}
.preds-grid .section-label {{ color:#f0d28e; }}
.pred-block {{ background:#1c2128; border:1px solid var(--border); border-radius:6px; padding:8px 10px; }}
.pred-header {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:4px; font-size:.86rem; }}
.pred-method {{ color:var(--accent); font-weight:700; }}
.pred-status {{ font-family:'Fira Code',monospace; padding:1px 7px; border-radius:4px; font-size:.75rem; }}
.pred-status.ok {{ background:#23863640; color:var(--green); }}
.pred-status.ko {{ background:#f8514922; color:var(--red); }}
.pp-flag {{ font-family:'Fira Code',monospace; padding:1px 7px; border-radius:4px; font-size:.7rem; background:#f0883e30; color:var(--orange); }}
.pp-flag.changed {{ background:#f0883e30; color:var(--orange); }}
.pred-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.pred-col {{ min-width:0; }}
.pred-col-label {{ font-size:.7rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin-bottom:2px; }}
.pred-text {{ margin:2px 0 0; max-height:220px; }}
.ctx-block {{ margin-bottom:6px; }}
@media (max-width:1100px) {{ .pair-container {{ grid-template-columns:1fr; }} .pred-pair {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>OOD Dataset &middot; per-method predictions on <code>{esc(suite)}</code></h1>
<p class="subtitle">
  Dataset: <code>{esc(qna_parquet)}</code> &middot;
  Showing {len(records):,} sampled QnAs from {global_stats['matched_rows']:,} matching rows
  across {global_stats['matched_repos']:,} repos and {global_stats['matched_commits']:,}
  (repo, commit) groups.
  Per-method EM / EditSim / CodeBLEU come from each method's
  <code>raw_samples</code> shard JSONs. Each card shows BOTH the raw model
  output and the post-processed string that <code>evaluation.metrics</code>
  feeds into the EM / EditSim / CodeBLEU computations.
</p>

<h2>Per-method headline scores</h2>
<div class="stats-grid">{method_summary_html}</div>

<h2>Dataset stats (matched rows)</h2>
<div class="stats-grid">
  <div class="stats-card"><h3>Totals</h3>
    <p><b>QnAs:</b> {global_stats['matched_rows']:,}</p>
    <p><b>Repos:</b> {global_stats['matched_repos']:,}</p>
    <p><b>Commits:</b> {global_stats['matched_commits']:,}</p>
  </div>
  <div class="stats-card"><h3>Cross-Repo Splits</h3>{make_bar(global_stats['cross_repo_splits'])}</div>
  <div class="stats-card"><h3>Assertion Types</h3>{make_bar(global_stats['assertion_types'])}</div>
  <div class="stats-card"><h3>Top Repositories</h3>{make_bar(global_stats['top_repos'])}</div>
  <div class="stats-card"><h3>QnAs per Commit</h3>{stat_table(global_stats['qna_per_commit'])}</div>
</div>

<h2>Sample QnAs ({len(records):,})</h2>
<div class="filter-bar">
  <label>Repo</label><select id="repoFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Cross-split</label><select id="crossFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Assertion</label><select id="assertionFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Show</label>
  <select id="correctFilter" onchange="filterCards()">
    <option value="">All</option>
    <option value="all">All methods &#10003;</option>
    <option value="none">All methods &#10007;</option>
    <option value="mixed">Mixed</option>
  </select>
  <label>Search</label><input id="searchBox" type="text" oninput="filterCards()" placeholder="repo, file, target, code...">
  <button onclick="expandAll()">Expand all shown</button>
  <button onclick="collapseAll()">Collapse all</button>
</div>
<div id="cards">{cards_html}</div>

<script>
const cards = [...document.querySelectorAll('.card')];
function fillSelect(id, attr) {{
  const values = [...new Set(cards.map(c => c.dataset[attr]).filter(Boolean))].sort();
  const sel = document.getElementById(id);
  values.forEach(v => {{ const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }});
}}
fillSelect('repoFilter', 'repo');
fillSelect('crossFilter', 'crossSplit');
fillSelect('assertionFilter', 'assertionType');
function toggleCard(header) {{
  const body = header.parentElement.querySelector('.card-body');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
}}
function correctMatches(c) {{
  const mode = document.getElementById('correctFilter').value;
  if (!mode) return true;
  const correct = parseInt(c.dataset.correct || 0);
  const scored  = parseInt(c.dataset.scored  || 0);
  if (mode === 'all')   return scored > 0 && correct === scored;
  if (mode === 'none')  return scored > 0 && correct === 0;
  if (mode === 'mixed') return scored > 1 && correct > 0 && correct < scored;
  return true;
}}
function matches(c) {{
  const repo   = document.getElementById('repoFilter').value.toLowerCase();
  const cross  = document.getElementById('crossFilter').value.toLowerCase();
  const at     = document.getElementById('assertionFilter').value.toLowerCase();
  const search = document.getElementById('searchBox').value.toLowerCase();
  return (!repo  || c.dataset.repo.toLowerCase() === repo)
      && (!cross || (c.dataset.crossSplit || '').toLowerCase() === cross)
      && (!at    || (c.dataset.assertionType || '').toLowerCase() === at)
      && (!search || c.textContent.toLowerCase().includes(search))
      && correctMatches(c);
}}
function filterCards() {{ cards.forEach(c => c.style.display = matches(c) ? '' : 'none'); }}
function expandAll() {{ cards.forEach(c => {{ if (matches(c)) c.querySelector('.card-body').style.display = 'block'; }}); }}
function collapseAll() {{ cards.forEach(c => c.querySelector('.card-body').style.display = 'none'); }}
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qna-parquet", type=Path,
                   default=Path("/scratch/lhotsko/REPO_DATASET/commit_parquet_ood/qna_pairs.parquet"))
    p.add_argument("--suite", default="ood_test",
                   help="Suite name used to look up shard files "
                        "(default: ood_test).")
    p.add_argument("--qnas-per-commit-cap", type=int, default=8,
                   help="Match the evaluator's --qnas-per-commit-limit "
                        "default (8); set 0 to keep all rows.")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--max-prefix-lines", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--output", "-o", type=Path,
                   default=Path("report_gru_ood_test_qnas.html"))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--write-selected-keys", type=Path, default=None,
                   help="Also dump the (repo_id, commit_sha) of every sampled "
                        "card as a JSONL. Feed this file to the evaluators' "
                        "--restrict-keys to dump predictions only for the "
                        "shown commits.")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="Directory holding per-method JSONLs produced by the "
                        "evaluators' --predictions-out flag. Looked up as "
                        "<dir>/<method>.jsonl for each method in "
                        "DEFAULT_METHODS.")
    args = p.parse_args()

    if not args.qna_parquet.exists():
        print(f"[error] QnA parquet not found: {args.qna_parquet}", flush=True)
        return 1

    print(f"Pass 1: scanning {args.qna_parquet} ...")
    counts, commit_idx, assertion_types, repos, cross_repo_splits = \
        scan_groups(args.qna_parquet, args.batch_size)
    n_rows = sum(counts.values())
    print(f"  found {n_rows:,} qnas across {len(counts):,} (repo, commit) groups")

    print("Pass 2: loading per-method scores ...")
    method_scores: Dict[str, Dict[Tuple[str, str], List[QnaScore]]] = {}
    for label, cfg in DEFAULT_METHODS.items():
        d = Path(cfg["dir"])
        if not d.is_dir():
            print(f"  [warn] {label}: directory not found: {d}")
            continue
        ms = load_method_scores(d, cfg["prefix"], args.suite)
        method_scores[label] = ms
        n_total_scores = sum(len(v) for v in ms.values())
        print(f"  {label}: {n_total_scores:,} per-QnA scores across "
              f"{len(ms):,} (repo, commit) groups")

    cap = args.qnas_per_commit_cap or 0
    effective_n = lambda k: (min(counts[k], cap) if cap else counts[k])

    method_summaries: Dict[str, Dict[str, float]] = {}
    for label, ms in method_scores.items():
        em_vals: List[float] = []
        ed_vals: List[float] = []
        cb_vals: List[float] = []
        for key, scores in ms.items():
            n_in_data = effective_n(key)
            for i in range(min(n_in_data, len(scores))):
                s = scores[i]
                em_vals.append(s.em); ed_vals.append(s.edit); cb_vals.append(s.cb)
        if em_vals:
            method_summaries[label] = {
                "n": len(em_vals),
                "em": sum(em_vals) / len(em_vals),
                "edit": sum(ed_vals) / len(ed_vals),
                "cb": sum(cb_vals) / len(cb_vals),
            }
        else:
            method_summaries[label] = {"n": 0, "em": 0, "edit": 0, "cb": 0}

    qna_per_commit_counts = [float(effective_n(k)) for k in counts]
    global_stats = {
        "matched_rows": int(sum(qna_per_commit_counts)),
        "matched_repos": len(repos),
        "matched_commits": len(counts),
        "cross_repo_splits": cross_repo_splits.most_common(),
        "assertion_types": assertion_types.most_common(20),
        "top_repos": repos.most_common(20),
        "qna_per_commit": numeric_summary(qna_per_commit_counts),
    }

    print("Pass 3: sampling cards ...")
    import random as _r
    rng = _r.Random(args.seed)
    keys = sorted(counts.keys())
    rng.shuffle(keys)

    wanted: Dict[Tuple[str, str], List[int]] = {}
    chosen: List[Tuple[Tuple[str, str], int]] = []
    for key in keys:
        n_in_data = effective_n(key)
        for pos in range(n_in_data):
            chosen.append((key, pos))
            if len(chosen) >= args.max_samples:
                break
        if len(chosen) >= args.max_samples:
            break
    for (key, pos) in chosen:
        wanted.setdefault(key, []).append(pos)
    print(f"  selected {len(chosen)} qnas across {len(wanted)} commits")

    if args.write_selected_keys:
        args.write_selected_keys.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_selected_keys, "w") as fh:
            for (repo_id, sha) in sorted(wanted.keys()):
                fh.write(json.dumps({"repo_id": repo_id,
                                     "commit_sha": sha}) + "\n")
        print(f"  wrote selected keys -> {args.write_selected_keys}")

    pred_index: Dict[str, Dict[Tuple[str, str, int], Dict[str, Any]]] = {}
    if args.predictions_dir:
        print(f"Pass 3b: loading per-method predictions from {args.predictions_dir} ...")
        for label in DEFAULT_METHODS:
            jp = args.predictions_dir / f"{label}.jsonl"
            if not jp.exists():
                print(f"  [warn] missing {jp}")
                continue
            idx: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
            with open(jp) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    key = (rec.get("repo_id"), rec.get("commit_sha"),
                           int(rec.get("qna_pos", 0)))
                    idx[key] = rec
            pred_index[label] = idx
            print(f"  {label}: {len(idx)} prediction records")

    print("Pass 4: loading prefix/target text for selected rows ...")
    row_map = load_selected_rows(args.qna_parquet, args.batch_size, wanted)
    print(f"  loaded {len(row_map)} rows")

    selected: List[Tuple[int, Dict[str, Any], int,
                          Dict[str, Optional[QnaScore]],
                          Dict[str, Optional[Dict[str, Any]]]]] = []
    idx = 0
    for (key, pos) in chosen:
        row = row_map.get((key[0], key[1], pos))
        if row is None:
            continue
        idx += 1
        ms: Dict[str, Optional[QnaScore]] = {}
        ps: Dict[str, Optional[Dict[str, Any]]] = {}
        for label, m in method_scores.items():
            grp = m.get(key, [])
            ms[label] = grp[pos] if pos < len(grp) else None
            ps[label] = pred_index.get(label, {}).get((key[0], key[1], pos))
        selected.append((idx, row, pos, ms, ps))

    print(f"Rendering {len(selected)} cards -> {args.output}")
    report = generate_html(
        selected,
        global_stats=global_stats,
        method_summaries=method_summaries,
        qna_parquet=args.qna_parquet,
        max_prefix_lines=args.max_prefix_lines,
        suite=args.suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report saved to: {args.output.resolve()}")

    print("\nHeadline:")
    for m, s in method_summaries.items():
        print(f"  {m:14s}  EM={100*s['em']:.2f}%  EditSim={s['edit']:.3f}  "
              f"CodeBLEU={s['cb']:.3f}  (n={s['n']:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
