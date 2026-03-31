#!/usr/bin/env python3
"""
Visualize QnA pairs with all context sources:
  - Original pair (prefix + target)
  - Oracle context (import-resolved definitions)
  - RAG retrieved chunks (k=3, 5, 10)
  - ICL few-shot examples (3-shot, 5-shot)
  - Text2LoRA text-conditioned descriptions
  - Text2LoRA code-conditioned embeddings

Produces an interactive HTML report for inspecting what each baseline actually sees.

Usage:
    # GPU needed for RAG/ICL embedding retrieval
    python visualize_pairs_context.py --split cr_test --limit 50 -o context_report.html
    python visualize_pairs_context.py --split ir_test --limit 20 --no-embed -o context_report.html
"""

import argparse
import html as html_mod
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.data_utils import get_default_splits_dir


def load_split_items(splits_dir: Path, split_name: str, limit_repos=None, limit=None):
    """Load QnA pairs from a split JSON, preserving metadata and repo embedding."""
    path = splits_dir / f"{split_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    repo_names = sorted(repos.keys())
    if limit_repos:
        repo_names = repo_names[:limit_repos]
    items = []
    for rn in repo_names:
        r = repos[rn]
        emb = r.get("embedding")
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if not prefix or not target:
                continue
            if target.lstrip().startswith(","):
                continue
            items.append({
                "prefix": prefix,
                "target": target,
                "repo": rn,
                "metadata": p.get("metadata", {}),
                "embedding": emb,
            })
    if limit:
        items = items[:limit]
    return items, repos


def get_oracle_context(oracle_cache_dir, repo_name, metadata):
    """Look up oracle context for a pair."""
    safe = repo_name.replace("/", "__")
    cache_path = oracle_cache_dir / f"{safe}.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    contexts = data.get("contexts", {})
    f = metadata.get("file", "")
    lineno = metadata.get("lineno", "")
    key = f"{f}::{lineno}"
    entry = contexts.get(key)
    if entry:
        return entry.get("extracted_code", "")
    return None


def get_text2lora_text_descriptions(text2lora_dir, repo_name):
    """Load text descriptions for a repo from text2lora tasks."""
    slug = repo_name.replace("/", "__")
    meta_path = text2lora_dir / "tasks" / slug / "metadata.yaml"
    if not meta_path.exists():
        return None
    try:
        import yaml
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        return meta.get("descriptions", [])
    except Exception:
        return None


def get_text2lora_code_embedding_info(embedding):
    """Summarize the code embedding vector (used by text2lora code-conditioned)."""
    if embedding is None or len(embedding) == 0:
        return None
    dim = len(embedding)
    norm = math.sqrt(sum(x * x for x in embedding))
    top_indices = sorted(range(dim), key=lambda i: abs(embedding[i]), reverse=True)[:10]
    top_vals = [(i, embedding[i]) for i in top_indices]
    return {
        "dim": dim,
        "norm": norm,
        "min": min(embedding),
        "max": max(embedding),
        "mean": sum(embedding) / dim,
        "top_magnitude": top_vals,
    }


def get_rag_chunks(rag_cache_dir, repo_name, prefix, embed_model, embed_tokenizer, device, top_k_list=(3, 5, 10)):
    """Retrieve RAG chunks at various k values. Returns {k: [chunk_strings]}."""
    import torch
    import torch.nn.functional as F

    safe = repo_name.replace("/", "__")
    cache_path = rag_cache_dir / f"{safe}.pt"
    if not cache_path.exists():
        return {k: [] for k in top_k_list}
    index = torch.load(cache_path, map_location="cpu", weights_only=False)
    chunks = index["chunks"]
    embs = index["embeddings"]
    if not chunks or embs is None:
        return {k: [] for k in top_k_list}
    embs = embs.float()

    query_text = prefix[-2000:]
    enc = embed_tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = embed_model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    query_emb = F.normalize(mean.cpu().float(), p=2, dim=-1)

    sims = (query_emb @ embs.T).squeeze(0)
    max_k = max(top_k_list)
    actual_k = min(max_k, len(chunks))
    _, top_indices = sims.topk(actual_k)
    top_idx = top_indices.tolist()

    result = {}
    for k in top_k_list:
        result[k] = [chunks[i] for i in top_idx[:min(k, len(top_idx))]]
    return result


def get_icl_examples(splits_dir, split_name, repo_name, repo_embedding, prefix,
                     train_repos, embed_model, embed_tokenizer, device, n_shots_list=(3, 5)):
    """Get ICL examples at various shot counts. Returns {n: [examples]}."""
    import torch
    import torch.nn.functional as F

    is_cr = split_name.startswith("cr_")

    if is_cr:
        if not repo_embedding:
            return {n: [] for n in n_shots_list}
        test_emb = torch.tensor(repo_embedding, dtype=torch.float32)
        test_emb = F.normalize(test_emb.unsqueeze(0), p=2, dim=-1)
        sims = []
        for rn, r in train_repos.items():
            emb = r.get("embedding")
            if emb is None:
                continue
            train_emb = torch.tensor(emb, dtype=torch.float32)
            train_emb = F.normalize(train_emb.unsqueeze(0), p=2, dim=-1)
            sim = (test_emb @ train_emb.T).item()
            sims.append((sim, rn))
        sims.sort(key=lambda x: -x[0])
        top_repos = [rn for _, rn in sims[:5]]
        candidates = []
        for rn in top_repos:
            for p in train_repos[rn].get("qna_pairs", []):
                pf = p.get("prefix", "")
                tg = p.get("target", "")
                if pf and tg and not tg.lstrip().startswith(","):
                    candidates.append({"prefix": pf, "target": tg, "repo": rn})
    else:
        rdata = train_repos.get(repo_name, {})
        candidates = []
        for p in rdata.get("qna_pairs", []):
            pf = p.get("prefix", "")
            tg = p.get("target", "")
            if pf and tg and not tg.lstrip().startswith(",") and pf != prefix:
                candidates.append({"prefix": pf, "target": tg, "repo": repo_name})

    if not candidates or embed_model is None:
        max_n = max(n_shots_list)
        selected = candidates[:max_n]
        return {n: selected[:n] for n in n_shots_list}

    query_text = prefix[-2000:]
    enc = embed_tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = embed_model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    query_emb = F.normalize(mean.cpu().float(), p=2, dim=-1)

    scored = []
    batch_size = 32
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        texts = [ex["prefix"][-2000:] for ex in batch]
        enc = embed_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = embed_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        means = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs = F.normalize(means.cpu().float(), p=2, dim=-1)
        s = (query_emb @ embs.T).squeeze(0)
        for j, ex in enumerate(batch):
            scored.append((s[j].item() if s.dim() > 0 else s.item(), ex))
    scored.sort(key=lambda x: -x[0])

    max_n = max(n_shots_list)
    top = [(sim, ex) for sim, ex in scored[:max_n]]
    result = {}
    for n in n_shots_list:
        result[n] = [(sim, ex) for sim, ex in top[:n]]
    return result


def esc(s):
    return html_mod.escape(str(s))


def truncate_code(text, max_lines=30):
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text, False
    return "\n".join(lines[:max_lines]), True


def generate_html(items_with_context, split_name, stats):
    cards = []
    for i, item in enumerate(items_with_context):
        meta = item["metadata"]
        prefix = item["prefix"]
        target = item["target"]
        repo = item["repo"]
        oracle_code = item.get("oracle_context") or ""
        rag_chunks = item.get("rag_chunks", {})
        icl_examples = item.get("icl_examples", {})
        t2l_text_descs = item.get("t2l_text_descriptions") or []
        t2l_code_info = item.get("t2l_code_embedding_info")
        file_name = meta.get("file", "")
        lineno = meta.get("lineno", "?")

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
            + '<span class="target-hl">' + esc(target) + '</span>'
        )
        cut_html = "\n".join(cut_parts)

        oracle_section = ""
        if oracle_code:
            oc_trunc, was_trunc = truncate_code(oracle_code)
            trunc_note = " (truncated)" if was_trunc else ""
            oracle_section = f"""
            <div class="context-block oracle-block">
              <div class="ctx-label oracle-label">ORACLE CONTEXT{trunc_note} ({len(oracle_code)} chars, {oracle_code.count(chr(10))+1} lines)</div>
              <pre><code>{esc(oc_trunc)}</code></pre>
              {f'<details><summary>Full oracle ({oracle_code.count(chr(10))+1} lines)</summary><pre><code>{esc(oracle_code)}</code></pre></details>' if was_trunc else ''}
            </div>"""
        else:
            oracle_section = '<div class="context-block oracle-block"><div class="ctx-label oracle-label">ORACLE CONTEXT</div><p class="no-data">No oracle context available</p></div>'

        rag_section = ""
        for k in sorted(rag_chunks.keys()):
            chunks = rag_chunks[k]
            if chunks:
                chunk_html = ""
                for ci, chunk in enumerate(chunks):
                    ct, was_t = truncate_code(chunk, max_lines=15)
                    chunk_html += f'<div class="chunk"><div class="chunk-num">Chunk {ci+1}/{len(chunks)} ({len(chunk)} chars)</div><pre><code>{esc(ct)}</code></pre>'
                    if was_t:
                        chunk_html += f'<details><summary>Full chunk</summary><pre><code>{esc(chunk)}</code></pre></details>'
                    chunk_html += '</div>'
                rag_section += f"""
                <div class="context-block rag-block">
                  <div class="ctx-label rag-label">RAG k={k} ({len(chunks)} chunks, {sum(len(c) for c in chunks)} chars total)</div>
                  {chunk_html}
                </div>"""
            else:
                rag_section += f'<div class="context-block rag-block"><div class="ctx-label rag-label">RAG k={k}</div><p class="no-data">No chunks found</p></div>'

        icl_section = ""
        for n in sorted(icl_examples.keys()):
            examples = icl_examples[n]
            if examples:
                ex_html = ""
                for ei, ex_item in enumerate(examples):
                    if isinstance(ex_item, tuple):
                        sim, ex = ex_item
                        sim_str = f" (sim={sim:.3f})"
                    else:
                        ex = ex_item
                        sim_str = ""
                    ep = ex.get("prefix", "")
                    et = ex.get("target", "")
                    er = ex.get("repo", "")
                    ep_trunc = ep[-800:] if len(ep) > 800 else ep
                    was_t = len(ep) > 800
                    ex_html += f"""<div class="chunk">
                      <div class="chunk-num">Example {ei+1}/{len(examples)} from <b>{esc(er)}</b>{sim_str}</div>
                      <div class="icl-pair">
                        <div class="icl-prefix"><div class="mini-label">Prefix{' (last 800 chars)' if was_t else ''}</div><pre><code>{esc(ep_trunc)}</code></pre></div>
                        <div class="icl-target"><div class="mini-label">Target</div><pre><code>{esc(et)}</code></pre></div>
                      </div>
                    </div>"""
                icl_section += f"""
                <div class="context-block icl-block">
                  <div class="ctx-label icl-label">ICL {n}-shot</div>
                  {ex_html}
                </div>"""
            else:
                icl_section += f'<div class="context-block icl-block"><div class="ctx-label icl-label">ICL {n}-shot</div><p class="no-data">No examples found</p></div>'

        t2l_text_section = ""
        if t2l_text_descs:
            desc_html = ""
            for di, desc in enumerate(t2l_text_descs):
                desc_html += f'<div class="chunk"><div class="chunk-num">Variant {di+1}/{len(t2l_text_descs)}</div><pre><code>{esc(desc)}</code></pre></div>'
            t2l_text_section = f"""
            <div class="context-block t2l-text-block">
              <div class="ctx-label t2l-text-label">TEXT2LORA TEXT DESCRIPTIONS ({len(t2l_text_descs)} variants)</div>
              <p style="color: var(--text-dim); font-size: 0.83em; margin-bottom: 8px;">
                Text2LoRA text-conditioned: the hypernetwork receives one of these descriptions
                (embedded via GTE-large) to predict LoRA weights for this repo.
              </p>
              {desc_html}
            </div>"""
        else:
            t2l_text_section = '<div class="context-block t2l-text-block"><div class="ctx-label t2l-text-label">TEXT2LORA TEXT DESCRIPTIONS</div><p class="no-data">No descriptions available for this repo</p></div>'

        t2l_code_section = ""
        if t2l_code_info:
            stats_html = f"""<table class="emb-stats">
              <tr><td>Dimension</td><td>{t2l_code_info['dim']}</td></tr>
              <tr><td>L2 Norm</td><td>{t2l_code_info['norm']:.4f}</td></tr>
              <tr><td>Min / Max</td><td>{t2l_code_info['min']:.6f} / {t2l_code_info['max']:.6f}</td></tr>
              <tr><td>Mean</td><td>{t2l_code_info['mean']:.6f}</td></tr>
            </table>"""
            top_html = "<div class='chunk'><div class='chunk-num'>Top-10 dimensions by magnitude</div><pre><code>"
            for idx, val in t2l_code_info['top_magnitude']:
                top_html += f"dim[{idx:4d}] = {val:+.6f}\n"
            top_html += "</code></pre></div>"
            t2l_code_section = f"""
            <div class="context-block t2l-code-block">
              <div class="ctx-label t2l-code-label">TEXT2LORA CODE EMBEDDING (Qwen3-Embedding, {t2l_code_info['dim']}-dim)</div>
              <p style="color: var(--text-dim); font-size: 0.83em; margin-bottom: 8px;">
                Text2LoRA code-conditioned: the hypernetwork receives this pre-computed code embedding
                (mean-pooled over repo code chunks) to predict LoRA weights.
              </p>
              {stats_html}
              {top_html}
            </div>"""
        else:
            t2l_code_section = '<div class="context-block t2l-code-block"><div class="ctx-label t2l-code-label">TEXT2LORA CODE EMBEDDING</div><p class="no-data">No code embedding available for this repo</p></div>'

        card = f"""
        <div class="card" data-repo="{esc(repo)}" data-idx="{i}">
          <div class="card-header" onclick="toggleCard(this)">
            <span class="card-num">#{i+1}</span>
            <span class="card-repo">{esc(repo)}</span>
            <span class="card-file">{esc(file_name)}</span>
            <span class="card-line">line {lineno}</span>
            <span class="badge">{esc(meta.get('prefix_type',''))}</span>
            <span class="card-toggle">&#9660;</span>
          </div>
          <div class="card-body" style="display:none;">
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
            </div>

            <h3 class="context-heading">Context Sources</h3>
            <div class="tab-bar">
              <button class="tab active" onclick="showTab(this,'oracle-{i}')">Oracle</button>
              <button class="tab" onclick="showTab(this,'rag-{i}')">RAG</button>
              <button class="tab" onclick="showTab(this,'icl-{i}')">ICL</button>
              <button class="tab" onclick="showTab(this,'t2l-text-{i}')">T2L Text</button>
              <button class="tab" onclick="showTab(this,'t2l-code-{i}')">T2L Code</button>
            </div>
            <div class="tab-content" id="oracle-{i}">{oracle_section}</div>
            <div class="tab-content" id="rag-{i}" style="display:none;">{rag_section}</div>
            <div class="tab-content" id="icl-{i}" style="display:none;">{icl_section}</div>
            <div class="tab-content" id="t2l-text-{i}" style="display:none;">{t2l_text_section}</div>
            <div class="tab-content" id="t2l-code-{i}" style="display:none;">{t2l_code_section}</div>
          </div>
        </div>"""
        cards.append(card)

    cards_html = "\n".join(cards)
    n_with_oracle = stats.get("n_with_oracle", 0)
    n_total = stats.get("n_total", len(items_with_context))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QnA + Context Visualization — {split_name}</title>
<style>
  :root {{
    --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --oracle-color: #d2a8ff; --rag-color: #f0883e; --icl-color: #3fb950;
    --prefix-bg: #1c2128; --target-bg: #0f1a0f;
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
  .card-file {{ color: var(--text-dim); }}
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

  pre {{ padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 0.8em; line-height: 1.4;
         max-height: 350px; overflow-y: auto; }}
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
  .oracle-label {{ color: var(--oracle-color); }}
  .rag-label {{ color: var(--rag-color); }}
  .icl-label {{ color: var(--icl-color); }}
  .oracle-block pre {{ border-left: 3px solid var(--oracle-color); }}
  .rag-block pre {{ border-left: 3px solid var(--rag-color); }}
  .icl-block pre {{ border-left: 3px solid var(--icl-color); }}
  .t2l-text-block pre {{ border-left: 3px solid #79c0ff; }}
  .t2l-code-block pre {{ border-left: 3px solid #ffa657; }}
  .t2l-text-label {{ color: #79c0ff; }}
  .t2l-code-label {{ color: #ffa657; }}
  .emb-stats {{ border-collapse: collapse; margin: 8px 0; font-size: 0.85em; }}
  .emb-stats td {{ padding: 3px 12px 3px 0; color: var(--text); }}
  .emb-stats td:first-child {{ color: var(--text-dim); }}

  .chunk {{ margin-bottom: 10px; }}
  .chunk-num {{ font-size: 0.78em; color: var(--text-dim); margin-bottom: 3px; }}
  .icl-pair {{ display: flex; gap: 10px; }}
  .icl-prefix {{ flex: 3; min-width: 0; }}
  .icl-target {{ flex: 1; min-width: 0; }}
  .mini-label {{ font-size: 0.72em; color: var(--text-dim); text-transform: uppercase; margin-bottom: 3px; }}

  @media (max-width: 900px) {{
    .pair-row {{ flex-direction: column; }}
    .arrow {{ transform: rotate(90deg); padding: 0; text-align: center; }}
    .icl-pair {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
<h1>QnA Pairs + Context Visualization</h1>
<p class="subtitle">Split: <b>{split_name}</b> &mdash; {n_total} pairs shown &mdash; Oracle: {n_with_oracle}/{n_total} have context</p>

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


def main():
    ap = argparse.ArgumentParser(description="Visualize QnA pairs with all context sources")
    default_splits = get_default_splits_dir()
    default_oracle = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "ORACLE_CONTEXT_CACHE",
    )
    default_rag = os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~/scratch")),
        "RAG_CHUNK_CACHE",
    )

    ap.add_argument("--splits-dir", type=str, default=default_splits)
    ap.add_argument("--split", type=str, default="cr_test")
    ap.add_argument("--oracle-cache-dir", type=str, default=default_oracle)
    ap.add_argument("--rag-cache-dir", type=str, default=default_rag)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--text2lora-dir", type=str, default="text2lora",
                    help="Path to text2lora/ repo root (for task descriptions)")
    ap.add_argument("--no-embed", action="store_true", help="Skip RAG/ICL retrieval (no GPU needed)")
    ap.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("-o", "--output", type=str, default=None)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    oracle_cache_dir = Path(args.oracle_cache_dir).expanduser().resolve()
    rag_cache_dir = Path(args.rag_cache_dir).expanduser().resolve()
    text2lora_dir = Path(args.text2lora_dir).expanduser().resolve()

    print(f"Loading split: {args.split} from {splits_dir}")
    items, _ = load_split_items(splits_dir, args.split, limit_repos=args.limit_repos, limit=args.limit)
    print(f"  {len(items)} pairs loaded")

    train_path = splits_dir / "train.json"
    train_repos = {}
    if train_path.exists():
        print("Loading train split for ICL candidates...")
        td = json.loads(train_path.read_text(encoding="utf-8"))
        train_repos = td.get("repositories", {})
        print(f"  {len(train_repos)} training repos")

    embed_model = embed_tokenizer = device = None
    if not args.no_embed:
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {args.embed_model} on {device}")
        embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
        embed_model = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True).to(device).eval()
        print("  Embedding model loaded")

    n_with_oracle = 0
    for idx, item in enumerate(items):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing {idx+1}/{len(items)}...", flush=True)

        oc = get_oracle_context(oracle_cache_dir, item["repo"], item["metadata"])
        item["oracle_context"] = oc
        if oc:
            n_with_oracle += 1

        item["t2l_text_descriptions"] = get_text2lora_text_descriptions(
            text2lora_dir, item["repo"])
        item["t2l_code_embedding_info"] = get_text2lora_code_embedding_info(
            item.get("embedding"))

        if not args.no_embed and rag_cache_dir.exists():
            item["rag_chunks"] = get_rag_chunks(
                rag_cache_dir, item["repo"], item["prefix"],
                embed_model, embed_tokenizer, device,
            )
        else:
            item["rag_chunks"] = {}

        if not args.no_embed:
            item["icl_examples"] = get_icl_examples(
                splits_dir, args.split, item["repo"], item["embedding"],
                item["prefix"], train_repos, embed_model, embed_tokenizer, device,
            )
        else:
            item["icl_examples"] = {}

    stats = {"n_total": len(items), "n_with_oracle": n_with_oracle}
    print(f"\nOracle coverage: {n_with_oracle}/{len(items)} ({100*n_with_oracle/max(1,len(items)):.1f}%)")
    print("Generating HTML...")

    report = generate_html(items, args.split, stats)

    out_path = args.output or f"context_report_{args.split}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {out_path}")
    print(f"Open: firefox {out_path}")


if __name__ == "__main__":
    main()
