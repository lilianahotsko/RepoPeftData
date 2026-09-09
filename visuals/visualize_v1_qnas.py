#!/usr/bin/env python3
"""Visualize the V1 (table-main_results) QnAs with per-method scores,
predictions (raw + post-processed) AND the actual context that was used
by the context-injection baselines (DRC oracle context, RAG top-k chunks).

Sources
-------

The V1 prediction inventory lives at::

    $SCRATCH/BASELINES/<method>_<split>.json
      └─ entries[i] = {repo, expected, got, got_raw, exact_match,
                       code_bleu, edit_similarity, ...method-specific extras}

``entries`` is in 1-to-1 alignment with the deterministic
``evaluation.data_utils.load_split($SCRATCH/REPO_DATASET, <split>)`` order
(sorted-repo, in-pair order, skipping targets that start with ``,``).
This means we can recover the prefix, target, test_file and lineno for
each scored QnA simply by joining on the global row index.

Contexts
--------

* DRC / oracle context (used by ``oracle_context*``, ``*_drc*``,
  ``fft_oracle``, ``hypernet_oracle``, ``hypernet_paw_oracle`` ...) is
  read straight from ``$SCRATCH/ORACLE_CONTEXT_CACHE_V*`` keyed by
  ``"<test_file>::<lineno>"``. The viewer extracts the ``extracted_code``
  string used at eval time.

* RAG retrieved chunks (used by ``rag_top*``, ``rag256_k*``,
  ``fft_rag256_k5``, ``slora_rag256_k5``) are NOT stored alongside the
  predictions in V1, but the chunk indices live in
  ``$SCRATCH/RAG_CHUNK_CACHE*/<repo_slug>.pt``. We do NOT re-run
  retrieval in the viewer (would require the embedding model on a GPU);
  instead, point ``--rag-context-dir`` at the JSON dump produced by
  ``scripts/dump_v1_rag_context.py`` and the top-k chunks per QnA will
  appear in the matching prediction blocks.

Usage
-----

    # Quick render on the IR-test split using the default method set.
    python visualize_v1_qnas.py --split ir_test \\
        --output report_v1_ir_test_qnas.html

    # Pre-dump RAG retrievals once, then re-render with chunk text:
    python scripts/dump_v1_rag_context.py --split ir_test \\
        --output-dir $SCRATCH/V1_RAG_CONTEXT_DUMP/ir_test
    python visualize_v1_qnas.py --split ir_test \\
        --rag-context-dir $SCRATCH/V1_RAG_CONTEXT_DUMP/ir_test \\
        --output report_v1_ir_test_qnas.html
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

# Ensure ``evaluation`` is importable when run from any cwd.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from evaluation.data_utils import load_split  # noqa: E402
from evaluation.metrics import postprocess_prediction, strip_comments  # noqa: E402


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Sensible default set of V1 methods that appear in
# ``RepoPeft_Paper/tables/table-main_results.tex``. Only methods whose
# ``$SCRATCH/BASELINES/<method>_<split>.json`` exists are actually loaded.
DEFAULT_METHODS: List[str] = [
    "pretrained",
    "rag_top3",
    "rag256_k5",
    "icl_3shot",
    "oracle_context_v2",
    "fft_no_oracle",
    "fft_rag256_k5",
    "fft_drc4k",
    "fft_oracle",
    "single_lora_no_oracle",
    "slora4k",
    "slora_rag256_k5",
    "slora_drc4k",
    "slora_drc_v4",
    "text2lora_code",
    "hypernet_no_oracle",
    "hypernet_drc8k",
    "per_repo_lora",   # synthesised from PER_REPO_LORA tree, see below
]

# Method -> context kind. Methods missing here are assumed context-free.
DRC_METHODS = {
    "oracle_context", "oracle_context_v2",
    "pretrained_drc_v3_8k", "pretrained_drc_v4_8k",
    "fft_drc4k", "fft_oracle",
    "slora_drc4k", "slora_drc_v4",
    "hypernet_drc8k", "hypernet_oracle", "hypernet_paw_oracle",
}
RAG_METHODS = {
    "rag_top3", "rag_top5", "rag_top10",
    "rag256_k5", "rag256_k10",
    "fft_rag256_k5", "slora_rag256_k5",
}

# Default oracle / RAG cache dirs (override via CLI).
DEFAULT_ORACLE_CACHE = "/scratch/lhotsko/ORACLE_CONTEXT_CACHE_V4"
DEFAULT_RAG_CHUNK_CACHE = "/scratch/lhotsko/RAG_CHUNK_CACHE_256"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class V1Entry:
    """One scored prediction from a BASELINES/<method>_<split>.json file."""
    got: str = ""
    got_raw: Optional[str] = None
    expected: str = ""
    em: bool = False
    edit: float = 0.0
    cb: float = 0.0
    extras: Optional[Dict[str, Any]] = None


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_v1_items(splits_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Reproduce the V1 evaluator's row ordering. Each item carries
    {repo, prefix, target, assertion_type, metadata}.
    """
    items = load_split(splits_dir, split)
    if not items:
        raise FileNotFoundError(
            f"Could not load V1 split JSON: {splits_dir}/{split}.json"
        )
    return items


def load_method_entries(method: str, split: str, baselines_dir: Path
                        ) -> Optional[List[V1Entry]]:
    """Read ``<baselines_dir>/<method>_<split>.json`` and return a list of
    ``V1Entry`` in original row order. Returns ``None`` if the file is
    missing or doesn't contain a usable ``entries`` array.
    """
    if method == "per_repo_lora":
        return load_per_repo_lora_entries(split)

    p = baselines_dir / f"{method}_{split}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] {method}: failed to parse {p}: {e}")
        return None
    raw = d.get("entries")
    if not isinstance(raw, list) or not raw:
        return None
    out: List[V1Entry] = []
    for rec in raw:
        out.append(V1Entry(
            got=str(rec.get("got") or ""),
            got_raw=(str(rec.get("got_raw")) if "got_raw" in rec else None),
            expected=str(rec.get("expected") or ""),
            em=bool(rec.get("exact_match")),
            edit=float(rec.get("edit_similarity") or 0.0),
            cb=float(rec.get("code_bleu") or 0.0),
            extras={k: v for k, v in rec.items()
                    if k not in {"repo", "expected", "got", "got_raw",
                                 "exact_match", "edit_similarity",
                                 "code_bleu"}},
        ))
    return out


def load_per_repo_lora_entries(split: str
                               ) -> Optional[List[V1Entry]]:
    """Concatenate per-repo ``lora_results.json`` entries in
    ``sorted(repos)`` order, following the same skip-on-leading-comma
    filter as the V1 evaluator. Returns the global-aligned list.
    """
    base = Path("/scratch/lhotsko/TRAINING_CHECKPOINTS/PER_REPO_LORA")
    if not base.is_dir():
        return None
    # PER_REPO_LORA/<author>/<repo>_results/<split>/lora_results.json
    by_repo: Dict[str, List[V1Entry]] = {}
    for author_dir in base.iterdir():
        if not author_dir.is_dir():
            continue
        for repo_dir in author_dir.iterdir():
            if not repo_dir.name.endswith("_results"):
                continue
            repo_name = repo_dir.name.removesuffix("_results")
            repo_id = f"{author_dir.name}/{repo_name}"
            p = repo_dir / split / "lora_results.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            ents: List[V1Entry] = []
            for rec in d.get("entries", []):
                ents.append(V1Entry(
                    got=str(rec.get("got") or ""),
                    got_raw=(str(rec.get("got_raw"))
                             if "got_raw" in rec else None),
                    expected=str(rec.get("expected") or ""),
                    em=bool(rec.get("exact_match")),
                    edit=float(rec.get("edit_similarity") or 0.0),
                    cb=float(rec.get("code_bleu") or 0.0),
                    extras=None,
                ))
            if ents:
                by_repo[repo_id] = ents
    if not by_repo:
        return None
    # NOTE: per-repo LoRA entries are emitted in the order the repo's
    # test_lora.py iterates `qna_pairs`, which is identical to the
    # global V1 split's per-repo order. So concatenation in sorted-repo
    # order yields the correct global alignment.
    out: List[V1Entry] = []
    for repo_id in sorted(by_repo.keys()):
        out.extend(by_repo[repo_id])
    return out


# ---------------------------------------------------------------------------
# Context lookups
# ---------------------------------------------------------------------------

class _RepoJsonCache:
    """Tiny LRU around ``json.load`` keyed by absolute path."""
    def __init__(self) -> None:
        self._cache: Dict[Path, Any] = {}

    def get(self, path: Path) -> Any:
        if path not in self._cache:
            try:
                self._cache[path] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._cache[path] = None
        return self._cache[path]


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def lookup_drc_context(repo_id: str, test_file: str, lineno: int,
                       cache_dir: Path, cache: _RepoJsonCache,
                       ) -> Optional[Dict[str, Any]]:
    p = cache_dir / f"{_slug(repo_id)}.json"
    d = cache.get(p)
    if not d:
        return None
    contexts = d.get("contexts", {})
    if not isinstance(contexts, dict):
        return None
    key = f"{test_file}::{lineno}"
    rec = contexts.get(key)
    return rec if isinstance(rec, dict) else None


def lookup_rag_context(repo_id: str, global_idx: int, row_within_repo: int,
                       rag_ctx_dir: Optional[Path],
                       cache: _RepoJsonCache,
                       ) -> Optional[Dict[str, Any]]:
    if rag_ctx_dir is None:
        return None
    p = rag_ctx_dir / f"{_slug(repo_id)}.json"
    d = cache.get(p)
    if not d:
        return None
    # Accept either:
    #   (a) ``retrievals`` as a list of records each with ``row_within_repo``
    #       and/or ``global_row_idx`` fields (dump_v1_rag_context.py format);
    #   (b) ``retrievals`` as a dict keyed by index string.
    retr = d.get("retrievals", d)
    if isinstance(retr, list):
        # Match on explicit row id fields, not positional index, because the
        # dump may be sparse (e.g. only a sample of QnAs were dumped).
        for rec in retr:
            if not isinstance(rec, dict):
                continue
            if rec.get("global_row_idx") == global_idx \
                    or rec.get("row_within_repo") == row_within_repo:
                return rec
        # As a final fallback (dense dumps), try positional index.
        if 0 <= row_within_repo < len(retr) \
                and isinstance(retr[row_within_repo], dict):
            return retr[row_within_repo]
        return None
    if isinstance(retr, dict):
        for k in (str(global_idx), str(row_within_repo),
                  f"row{global_idx}", f"row{row_within_repo}"):
            if k in retr:
                return retr[k]
        return None
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

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


def fmt_score(e: Optional[V1Entry]) -> str:
    if e is None:
        return ("<span class='score-cell missing' title='no entry for this QnA'>"
                "&mdash;</span>")
    em_cls = "ok" if e.em else "ko"
    icon = "&#10003;" if e.em else "&#10007;"
    return (
        f"<span class='score-cell {em_cls}' "
        f"title='EM={int(e.em)}  EditSim={e.edit:.3f}  CodeBLEU={e.cb:.3f}'>"
        f"{icon} <span class='score-sub'>{e.edit:.2f}/{e.cb:.2f}</span>"
        f"</span>"
    )


def _format_drc_context(rec: Dict[str, Any]) -> str:
    """Pretty-print the oracle/DRC context record."""
    code = rec.get("extracted_code") or rec.get("extracted_code_full") or ""
    imports = rec.get("resolved_imports") or []
    used = rec.get("used_names") or []
    enc_fn = rec.get("enclosing_function") or ""
    n_chars = rec.get("n_chars_extracted")

    summary_bits = []
    if enc_fn:
        summary_bits.append(f"enclosing_function=<code>{esc(enc_fn)}</code>")
    if imports:
        summary_bits.append(f"resolved_imports=[{esc(', '.join(imports))}]")
    if used:
        summary_bits.append(f"used_names=[{esc(', '.join(used[:20]))}]")
    if n_chars is not None:
        summary_bits.append(f"n_chars={n_chars}")
    summary_html = "<div class='ctx-meta'>" + " &middot; ".join(summary_bits) + "</div>" if summary_bits else ""

    if not code:
        return summary_html + "<p class='muted'>(no extracted_code in cache)</p>"
    return (
        "<details class='ctx-block' open><summary>DRC oracle context "
        f"({code.count(chr(10)) + 1} lines, {len(code):,} chars)</summary>"
        f"{summary_html}<pre><code>{esc(code)}</code></pre></details>"
    )


def _format_rag_context(rec: Dict[str, Any]) -> str:
    chunks = rec.get("top_k_chunks") or rec.get("chunks") or []
    parallel_scores = rec.get("scores") or []
    k = rec.get("top_k") or len(chunks)
    if not chunks:
        return "<p class='muted'>(no chunks in dump)</p>"
    parts = []
    for i, ch in enumerate(chunks):
        if isinstance(ch, dict):
            text = ch.get("text") or ""
            score = ch.get("score")
        else:
            text = str(ch)
            score = parallel_scores[i] if i < len(parallel_scores) else None
        score_html = (f" <span class='muted'>score={score:.4f}</span>"
                      if isinstance(score, (int, float)) else "")
        parts.append(
            f"<details class='ctx-block' open><summary>chunk {i+1}/{k}"
            f" ({text.count(chr(10)) + 1} lines, {len(text):,} chars)"
            f"{score_html}</summary>"
            f"<pre><code>{esc(text)}</code></pre></details>"
        )
    return "\n".join(parts)


def _label_for_pred(e: V1Entry, target: str) -> Tuple[str, str, str, str]:
    """Return (raw_text, pp_text, target_clean, status_label)."""
    raw = e.got_raw if e.got_raw is not None else e.got
    pp = e.got
    # If got_raw is missing, derive an equivalent post-processed string.
    if e.got_raw is None and raw is not None:
        pp = postprocess_prediction(raw, target)
    target_clean = strip_comments(target).strip()
    status = "EM" if e.em else "no-EM"
    return raw, pp, target_clean, status


def render_pred_block(method: str, ctx_kind: str,
                      e: V1Entry, target: str,
                      drc_ctx: Optional[Dict[str, Any]],
                      rag_ctx: Optional[Dict[str, Any]]) -> str:
    raw, pp, target_clean, status = _label_for_pred(e, target)

    em_cls = "ok" if e.em else "ko"
    pp_changed = (raw != pp)
    pp_matches = pp.strip() == target_clean.strip()
    pp_match_cls = "ok" if pp_matches else "ko"
    pp_match_lbl = "matches target" if pp_matches else "differs from target"

    pp_flag = (" <span class='pp-flag changed' title='postprocessing modified the raw output'>postprocessed</span>"
               if pp_changed else "")

    # Extras such as n_retrieved / had_oracle_context.
    extras_html = ""
    if e.extras:
        bits = []
        for k, v in e.extras.items():
            bits.append(f"<span class='extra-chip'>{esc(k)}={esc(v)}</span>")
        if bits:
            extras_html = "<div class='extras-row'>" + " ".join(bits) + "</div>"

    ctx_html = ""
    if ctx_kind == "drc" and drc_ctx is not None:
        ctx_html = _format_drc_context(drc_ctx)
    elif ctx_kind == "drc":
        ctx_html = "<p class='muted'>(no DRC context found in cache for this QnA)</p>"
    elif ctx_kind == "rag" and rag_ctx is not None:
        ctx_html = _format_rag_context(rag_ctx)
    elif ctx_kind == "rag":
        ctx_html = ("<p class='muted'>(RAG retrieval not pre-computed; "
                    "pass --rag-context-dir to load top-k chunks)</p>")

    return (
        f"<div class='pred-block'>"
        f"<div class='pred-header'>"
        f"<span class='pred-method'>{esc(method)}</span>"
        f"<span class='pred-status {em_cls}'>{status}</span>"
        f"<span class='pred-status {pp_match_cls}' "
        f"title='compares post-processed pred vs strip_comments(target)'>"
        f"{pp_match_lbl}</span>"
        f"<span class='ctx-kind {ctx_kind}'>{ctx_kind}</span>"
        f"{pp_flag}"
        f"</div>"
        f"{extras_html}"
        f"{ctx_html}"
        f"<div class='pred-pair'>"
        f"  <div class='pred-col'>"
        f"    <div class='pred-col-label'>Raw model output</div>"
        f"    <pre class='pred-text'><code>"
        f"{esc(raw) if raw else '<em>(empty)</em>'}"
        f"</code></pre>"
        f"  </div>"
        f"  <div class='pred-col'>"
        f"    <div class='pred-col-label'>Post-processed (scored)</div>"
        f"    <pre class='pred-text pp'><code>"
        f"{esc(pp) if pp else '<em>(empty)</em>'}"
        f"</code></pre>"
        f"  </div>"
        f"</div>"
        f"</div>"
    )


def render_card(idx: int, global_row: int, item: Dict[str, Any],
                method_entries: Dict[str, Optional[V1Entry]],
                drc_ctxs: Dict[str, Optional[Dict[str, Any]]],
                rag_ctxs: Dict[str, Optional[Dict[str, Any]]],
                method_ctx_kind: Dict[str, str],
                max_prefix_lines: int) -> str:
    prefix = str(item.get("prefix") or "")
    target = str(item.get("target") or "")
    repo = str(item.get("repo") or "")
    meta = item.get("metadata") or {}
    test_file = str(meta.get("file") or "")
    lineno = meta.get("lineno") or 0
    function = str(meta.get("test_function") or "")
    difficulty = str(item.get("difficulty") or "")
    assertion_type = str(item.get("assertion_type") or "")

    prefix_view = compact_code(prefix, max_prefix_lines)
    target_clean = strip_comments(target).strip()

    method_html_parts: List[str] = []
    pred_block_parts: List[str] = []
    n_correct = 0
    n_total = 0
    for m, e in method_entries.items():
        if e is not None:
            n_total += 1
            if e.em:
                n_correct += 1
        method_html_parts.append(
            f"<div class='m-row'>"
            f"<span class='m-name'>{esc(m)}</span>"
            f"<span class='m-ctx ctx-kind {method_ctx_kind.get(m, 'none')}'>"
            f"{method_ctx_kind.get(m, 'none')}</span>"
            f"{fmt_score(e)}</div>"
        )
        if e is not None:
            pred_block_parts.append(render_pred_block(
                m, method_ctx_kind.get(m, "none"), e, target,
                drc_ctxs.get(m), rag_ctxs.get(m),
            ))
    method_html = "\n".join(method_html_parts)
    preds_html = "\n".join(pred_block_parts) if pred_block_parts else ""

    return f"""
<div class="card" data-repo="{esc(repo)}"
     data-assertion-type="{esc(assertion_type)}"
     data-difficulty="{esc(difficulty)}"
     data-correct="{n_correct}" data-scored="{n_total}">
  <div class="card-header" onclick="toggleCard(this)">
    <span class="card-num">#{idx}</span>
    <span class="card-repo">{esc(repo)}</span>
    <span class="badge">{esc(assertion_type)}</span>
    {('<span class="badge difficulty">' + esc(difficulty) + '</span>') if difficulty else ''}
    <span class="card-file">{esc(test_file)}</span>
    <span class="card-cut">@ line {esc(lineno)}</span>
    <span class="score-summary">{n_correct}/{n_total} methods &#10003;</span>
    <span class="card-toggle">&#9660;</span>
  </div>
  <div class="card-body" style="display:none;">
    <div class="meta-info">
      <span><b>Row idx:</b> {global_row}</span>
      <span><b>Function:</b> {esc(function or 'N/A')}</span>
      <span><b>Has imports:</b> {esc(meta.get('has_imports'))}</span>
      <span><b>Was multiline:</b> {esc(meta.get('was_multiline'))}</span>
    </div>
    <div class="pair-container">
      <div class="pair-section prefix-section">
        <div class="section-label">PREFIX (last {max_prefix_lines} lines)</div>
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
          &#10003; = exact match; sub-numbers show EditSim/CodeBLEU; pill = context kind.
        </p>
      </div>
    </div>
    {('<div class="preds-grid"><div class="section-label">PER-METHOD PREDICTIONS, CONTEXTS &amp; POSTPROCESSING</div>' + preds_html + '</div>') if preds_html else ''}
  </div>
</div>
"""


def generate_html(records: List[Tuple[int, int, Dict[str, Any],
                                       Dict[str, Optional[V1Entry]],
                                       Dict[str, Optional[Dict[str, Any]]],
                                       Dict[str, Optional[Dict[str, Any]]]]],
                  *,
                  global_stats: Dict[str, Any],
                  method_summaries: Dict[str, Dict[str, float]],
                  method_ctx_kind: Dict[str, str],
                  split: str,
                  max_prefix_lines: int) -> str:
    cards_html = "\n".join(
        render_card(i, g, it, m, drcs, rags, method_ctx_kind, max_prefix_lines)
        for i, g, it, m, drcs, rags in records
    )
    methods = list(method_summaries.keys())
    method_summary_html = "".join(
        f"<div class='stats-card'><h3>{esc(m)} "
        f"<span class='m-ctx ctx-kind {method_ctx_kind.get(m, 'none')}'>"
        f"{method_ctx_kind.get(m, 'none')}</span></h3>"
        f"<p><b>Scored:</b> {method_summaries[m]['n']:.0f}</p>"
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
<title>V1 main results &middot; {split} &middot; per-method predictions + contexts</title>
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
.stats-card h3 {{ color:var(--accent); margin:0 0 10px; font-size:1rem; display:flex; align-items:center; gap:6px; }}
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
.badge.difficulty {{ background:#8957e530; color:#d2a8ff; }}
.score-summary {{ font-weight:600; color:var(--muted); margin-left:8px; }}
.card-body {{ padding:0 14px 14px; }}
.meta-info {{ display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:.84rem; margin:4px 0 12px; }}
.pair-container {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(220px,.8fr) minmax(290px,1fr); gap:12px; align-items:start; }}
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
.m-row {{ display:flex; align-items:center; gap:8px; background:#1c2128; border:1px solid var(--border); border-radius:5px; padding:6px 9px; }}
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
.ctx-kind {{ font-family:'Fira Code',monospace; padding:1px 7px; border-radius:4px; font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; }}
.ctx-kind.none {{ background:#30363d80; color:var(--muted); }}
.ctx-kind.drc  {{ background:#8957e540; color:#d2a8ff; }}
.ctx-kind.rag  {{ background:#1f6feb40; color:#58a6ff; }}
.m-ctx {{ font-size:.65rem; padding:1px 6px; }}
.extras-row {{ display:flex; flex-wrap:wrap; gap:6px; margin:2px 0 4px; }}
.extra-chip {{ font-family:'Fira Code',monospace; font-size:.7rem; padding:1px 6px; border-radius:3px; background:#30363d80; color:var(--muted); }}
.ctx-meta {{ color:var(--muted); font-size:.78rem; margin:4px 0 6px; }}
.ctx-block {{ margin-bottom:6px; }}
.pred-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.pred-col {{ min-width:0; }}
.pred-col-label {{ font-size:.7rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin-bottom:2px; }}
.pred-text {{ margin:2px 0 0; max-height:220px; }}
@media (max-width:1100px) {{ .pair-container {{ grid-template-columns:1fr; }} .pred-pair {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>V1 main results &middot; per-method predictions + contexts on <code>{esc(split)}</code></h1>
<p class="subtitle">
  Showing {len(records):,} sampled QnAs out of {global_stats['matched_rows']:,} V1 rows in <code>{esc(split)}.json</code>
  across {global_stats['matched_repos']:,} repos.
  Predictions are loaded from <code>$SCRATCH/BASELINES/&lt;method&gt;_{esc(split)}.json</code>
  (raw <code>got_raw</code> &amp; post-processed <code>got</code>).
  DRC oracle context comes from the oracle cache; RAG retrieved chunks come from
  a pre-computed dump (see <code>--rag-context-dir</code>).
</p>

<h2>Per-method headline scores</h2>
<div class="stats-grid">{method_summary_html}</div>

<h2>Dataset stats</h2>
<div class="stats-grid">
  <div class="stats-card"><h3>Totals</h3>
    <p><b>QnAs:</b> {global_stats['matched_rows']:,}</p>
    <p><b>Repos:</b> {global_stats['matched_repos']:,}</p>
  </div>
  <div class="stats-card"><h3>Assertion types</h3>{make_bar(global_stats['assertion_types'])}</div>
  <div class="stats-card"><h3>Difficulty bins</h3>{make_bar(global_stats['difficulties'])}</div>
  <div class="stats-card"><h3>Top repositories</h3>{make_bar(global_stats['top_repos'])}</div>
  <div class="stats-card"><h3>QnAs / repo</h3>{stat_table(global_stats['qna_per_repo'])}</div>
</div>

<h2>Sample QnAs ({len(records):,})</h2>
<div class="filter-bar">
  <label>Repo</label><select id="repoFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Assertion</label><select id="assertionFilter" onchange="filterCards()"><option value="">All</option></select>
  <label>Difficulty</label><select id="difficultyFilter" onchange="filterCards()"><option value="">All</option></select>
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
fillSelect('assertionFilter', 'assertionType');
fillSelect('difficultyFilter', 'difficulty');
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
  const at     = document.getElementById('assertionFilter').value.toLowerCase();
  const diff   = document.getElementById('difficultyFilter').value.toLowerCase();
  const search = document.getElementById('searchBox').value.toLowerCase();
  return (!repo  || c.dataset.repo.toLowerCase() === repo)
      && (!at    || (c.dataset.assertionType || '').toLowerCase() === at)
      && (!diff  || (c.dataset.difficulty || '').toLowerCase() === diff)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="ir_test",
                   choices=["ir_test", "cr_test", "ood_test"],
                   help="Which V1 split to load (default: ir_test).")
    p.add_argument("--splits-dir", type=Path,
                   default=Path("/scratch/lhotsko/REPO_DATASET"),
                   help="Dir containing the V1 split JSONs.")
    p.add_argument("--baselines-dir", type=Path,
                   default=Path("/scratch/lhotsko/BASELINES"),
                   help="Dir containing <method>_<split>.json prediction files.")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated method list. Default: a curated set "
                        "matching the main results table.")
    p.add_argument("--oracle-cache-dir", type=Path,
                   default=Path(DEFAULT_ORACLE_CACHE),
                   help="DRC / oracle cache dir, "
                        f"default {DEFAULT_ORACLE_CACHE}.")
    p.add_argument("--rag-context-dir", type=Path, default=None,
                   help="Optional dir with per-repo retrieval dumps "
                        "(<repo_slug>.json) produced by "
                        "scripts/dump_v1_rag_context.py.")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--max-prefix-lines", type=int, default=60)
    p.add_argument("--output", "-o", type=Path,
                   default=None,
                   help="Where to write the HTML report. "
                        "Default: report_v1_<split>_qnas.html.")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    out_path = args.output or Path(f"report_v1_{args.split}_qnas.html")

    methods = [m.strip() for m in args.methods.split(",")] if args.methods \
        else list(DEFAULT_METHODS)

    method_ctx_kind: Dict[str, str] = {}
    for m in methods:
        ml = m.lower()
        is_rag = m in RAG_METHODS or ml.startswith("rag") or "_rag" in ml
        # ``_no_oracle`` flags methods trained / evaluated WITHOUT DRC context
        # (e.g. ``fft_no_oracle``, ``hypernet_no_oracle``); only treat
        # ``_oracle`` / ``oracle_context*`` / ``_drc`` as positive DRC users.
        is_drc = (
            (m in DRC_METHODS)
            or ml.startswith("oracle_context")
            or "_drc" in ml
            or ("_oracle" in ml and "_no_oracle" not in ml)
        )
        if is_rag and is_drc:
            # mixed-context method (e.g. slora_rag256_k5 doesn't actually
            # use DRC; the heuristic above shouldn't flag it). Prefer rag
            # when the substring evidence is symmetric.
            method_ctx_kind[m] = "rag"
        elif is_drc:
            method_ctx_kind[m] = "drc"
        elif is_rag:
            method_ctx_kind[m] = "rag"
        else:
            method_ctx_kind[m] = "none"

    print(f"Pass 1: loading V1 split {args.split} ...", flush=True)
    items = load_v1_items(args.splits_dir, args.split)
    n_rows = len(items)
    print(f"  {n_rows:,} items across {len({i['repo'] for i in items}):,} repos",
          flush=True)

    print(f"Pass 2: loading prediction files ...", flush=True)
    per_method: Dict[str, Optional[List[V1Entry]]] = {}
    for m in methods:
        ents = load_method_entries(m, args.split, args.baselines_dir)
        per_method[m] = ents
        if ents is None:
            print(f"  [skip] {m}: no entries found")
            continue
        ok = sum(1 for e in ents if e.em)
        if len(ents) != n_rows:
            print(f"  [warn] {m}: {len(ents):,} entries (expected {n_rows:,}). "
                  f"Will align by row index up to min length.")
        print(f"  {m:30s}  EM={100*ok/max(len(ents),1):6.2f}%  "
              f"(n={len(ents):,})")

    available = [m for m in methods if per_method[m] is not None]
    if not available:
        print("[error] no method prediction files were loaded.", flush=True)
        return 1

    method_summaries: Dict[str, Dict[str, float]] = {}
    for m in available:
        es = per_method[m]
        if not es:
            method_summaries[m] = {"n": 0, "em": 0, "edit": 0, "cb": 0}
            continue
        n = len(es)
        em = sum(1 for e in es if e.em) / n
        ed = sum(e.edit for e in es) / n
        cb = sum(e.cb for e in es) / n
        method_summaries[m] = {"n": n, "em": em, "edit": ed, "cb": cb}

    # Global stats from items only (deterministic).
    assertion_types: Counter = Counter()
    difficulties: Counter = Counter()
    repo_counts: Counter = Counter()
    for it in items:
        assertion_types[it.get("assertion_type") or "unknown"] += 1
        difficulties[it.get("difficulty") or "unknown"] += 1
        repo_counts[it["repo"]] += 1
    qna_per_repo = numeric_summary([float(v) for v in repo_counts.values()])
    global_stats = {
        "matched_rows": n_rows,
        "matched_repos": len(repo_counts),
        "assertion_types": assertion_types.most_common(),
        "difficulties": difficulties.most_common(),
        "top_repos": repo_counts.most_common(20),
        "qna_per_repo": qna_per_repo,
    }

    print(f"Pass 3: sampling {args.max_samples} cards (seed={args.seed}) ...",
          flush=True)
    import random as _r
    rng = _r.Random(args.seed)
    idxs = list(range(n_rows))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:args.max_samples])

    cache = _RepoJsonCache()
    rag_cache = _RepoJsonCache()
    rag_ctx_dir = args.rag_context_dir if args.rag_context_dir else None

    # Build per-(repo) in-repo row indices for RAG lookups.
    in_repo_idx: List[int] = []
    seen_per_repo: Dict[str, int] = defaultdict(int)
    for it in items:
        in_repo_idx.append(seen_per_repo[it["repo"]])
        seen_per_repo[it["repo"]] += 1

    records: List[Tuple[int, int, Dict[str, Any],
                        Dict[str, Optional[V1Entry]],
                        Dict[str, Optional[Dict[str, Any]]],
                        Dict[str, Optional[Dict[str, Any]]]]] = []

    for card_no, gidx in enumerate(chosen, start=1):
        it = items[gidx]
        meta = it.get("metadata") or {}
        test_file = str(meta.get("file") or "")
        lineno = int(meta.get("lineno") or 0)
        repo_id = it["repo"]
        m_entries: Dict[str, Optional[V1Entry]] = {}
        drc_ctxs: Dict[str, Optional[Dict[str, Any]]] = {}
        rag_ctxs: Dict[str, Optional[Dict[str, Any]]] = {}
        for m in available:
            ents = per_method[m]
            e = ents[gidx] if (ents and gidx < len(ents)) else None
            m_entries[m] = e
            if method_ctx_kind.get(m) == "drc":
                drc_ctxs[m] = lookup_drc_context(
                    repo_id, test_file, lineno,
                    args.oracle_cache_dir, cache,
                )
            else:
                drc_ctxs[m] = None
            if method_ctx_kind.get(m) == "rag":
                rag_ctxs[m] = lookup_rag_context(
                    repo_id, gidx, in_repo_idx[gidx],
                    rag_ctx_dir, rag_cache,
                )
            else:
                rag_ctxs[m] = None
        records.append((card_no, gidx, it, m_entries, drc_ctxs, rag_ctxs))

    print(f"Rendering {len(records)} cards -> {out_path}", flush=True)
    html_doc = generate_html(
        records,
        global_stats=global_stats,
        method_summaries={m: method_summaries[m] for m in available},
        method_ctx_kind={m: method_ctx_kind[m] for m in available},
        split=args.split,
        max_prefix_lines=args.max_prefix_lines,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Report saved to: {out_path.resolve()}", flush=True)

    print("\nHeadline:")
    for m in available:
        s = method_summaries[m]
        print(f"  {m:30s}  EM={100*s['em']:6.2f}%  EditSim={s['edit']:.3f}  "
              f"CodeBLEU={s['cb']:.3f}  (n={s['n']:.0f})  "
              f"ctx={method_ctx_kind.get(m,'none')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
