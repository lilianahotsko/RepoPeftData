#!/usr/bin/env python3
"""Score dumped predictions with task-appropriate metrics, aggregate per repo,
and produce a **demo-selection report**: which repos (and which individual
QnAs) the multi-task Code2LoRA-GRU model answers best -- separately for the
``qa`` task and the ``assert_rhs`` task, plus a combined ranking of repos that
are strong on **both** (ideal candidates for a demo video).

Inputs are the per-QnA prediction JSONL files written by
``evaluation/run_code2lora_gru_v2_eval.py --predictions-out`` (one record per
QnA with ``prefix`` / ``target`` / ``prediction`` and generation-time
code metrics).

QA answers are free-form natural language, so alongside the code-oriented
metrics (exact_match / edit_similarity / code_bleu) we compute the standard
**generative-QA** metrics:

  * SQuAD-style **token-F1** and normalized **EM** (article/punct-insensitive)
  * **ROUGE-1 / ROUGE-2 / ROUGE-L** F-measure
  * **BLEU** (sentence BLEU with smoothing)

The headline ``qa_quality`` score is the mean of ROUGE-L(F) and token-F1 --
both in [0, 1] and robust to paraphrase length. For assertions the headline is
the canonical ``exact_match``.

Usage::

    python evaluation/qna_demo_selection.py \
        --qa-dir     $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/c2l_evo_mt_qa_pred \
        --assert-dir $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/c2l_evo_mt_assert_pred \
        --out-dir    $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/c2l_evo_mt_demo_selection \
        --top-repos 15 --examples-per-repo 5
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_SMOOTH = SmoothingFunction().method1
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# QA metrics (SQuAD / ROUGE / BLEU)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common: Dict[str, int] = defaultdict(int)
    rc: Dict[str, int] = defaultdict(int)
    for t in r:
        rc[t] += 1
    n_same = 0
    seen: Dict[str, int] = defaultdict(int)
    for t in p:
        seen[t] += 1
        if seen[t] <= rc[t]:
            n_same += 1
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(r)
    return 2 * prec * rec / (prec + rec)


def _bleu(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0
    return float(sentence_bleu([r], p, smoothing_function=_SMOOTH))


def qa_metrics(pred: str, ref: str) -> Dict[str, float]:
    rs = _ROUGE.score(ref or "", pred or "")
    f1 = _token_f1(pred, ref)
    rougeL = rs["rougeL"].fmeasure
    return {
        "token_f1": f1,
        "em_norm": float(_normalize(pred) == _normalize(ref)),
        "rouge1": rs["rouge1"].fmeasure,
        "rouge2": rs["rouge2"].fmeasure,
        "rougeL": rougeL,
        "bleu": _bleu(pred, ref),
        "qa_quality": 0.5 * (rougeL + f1),
    }


# ---------------------------------------------------------------------------
# Loading + aggregation
# ---------------------------------------------------------------------------

def _load_preds(pred_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(str(pred_dir / "*.jsonl")))
    recs: List[Dict[str, Any]] = []
    seen: set = set()
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("repo_id"), r.get("commit_sha"),
                       r.get("qna_pos"), r.get("prefix"))
                if key in seen:
                    continue
                seen.add(key)
                recs.append(r)
    return recs


def score_qa(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in recs:
        m = qa_metrics(r.get("prediction", ""), r.get("target", ""))
        r.update(m)
        # keep gen-time code metrics too
        r["edit_similarity"] = float(r.get("edit_similarity", 0.0))
        r["code_bleu"] = float(r.get("code_bleu", 0.0))
        r["exact_match"] = float(r.get("exact_match", 0.0))
        r["_headline"] = m["qa_quality"]
    return recs


def score_assert(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for r in recs:
        r["exact_match"] = float(r.get("exact_match", 0.0))
        r["edit_similarity"] = float(r.get("edit_similarity", 0.0))
        r["code_bleu"] = float(r.get("code_bleu", 0.0))
        # blended headline so ties on EM break by closeness
        r["_headline"] = r["exact_match"] + 0.001 * r["edit_similarity"]
    return recs


def per_repo(recs: List[Dict[str, Any]], metric_keys: List[str]
             ) -> Dict[str, Dict[str, Any]]:
    by_repo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_repo[r.get("repo_id", "?")].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for repo, rs in by_repo.items():
        agg = {"repo_id": repo, "n": len(rs),
               "n_commits": len({x.get("commit_sha") for x in rs})}
        for k in metric_keys:
            agg[k] = mean(float(x.get(k, 0.0)) for x in rs)
        out[repo] = agg
    return out


def top_examples(recs: List[Dict[str, Any]], repo: str, k: int
                 ) -> List[Dict[str, Any]]:
    rs = [r for r in recs if r.get("repo_id") == repo]
    rs.sort(key=lambda r: r.get("_headline", 0.0), reverse=True)
    return rs[:k]


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            vals = []
            for c in cols:
                v = row.get(c, "")
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            fh.write(",".join(vals) + "\n")


def _fmt_examples_md(recs: List[Dict[str, Any]], task: str) -> str:
    lines = []
    for i, r in enumerate(recs, 1):
        lines.append(f"- **[{i}]** commit `{str(r.get('commit_sha',''))[:10]}`  "
                     f"(headline={r.get('_headline',0):.3f})")
        if task == "qa":
            lines.append(f"    - metrics: rougeL={r.get('rougeL',0):.3f} "
                         f"tokenF1={r.get('token_f1',0):.3f} "
                         f"bleu={r.get('bleu',0):.3f} "
                         f"em_norm={r.get('em_norm',0):.0f}")
        else:
            lines.append(f"    - metrics: EM={r.get('exact_match',0):.0f} "
                         f"edit={r.get('edit_similarity',0):.3f} "
                         f"codebleu={r.get('code_bleu',0):.3f}")
        lines.append(f"    - Q: {r.get('prefix','').strip()}")
        lines.append(f"    - REF : {r.get('target','').strip()}")
        lines.append(f"    - PRED: {r.get('prediction','').strip()}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--assert-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-repos", type=int, default=15)
    ap.add_argument("--examples-per-repo", type=int, default=5)
    ap.add_argument("--min-qnas", type=int, default=8,
                    help="Ignore repos with fewer than this many scored QnAs "
                         "when ranking (avoids tiny-sample winners).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qa_metric_keys = ["qa_quality", "rougeL", "rouge1", "rouge2", "token_f1",
                      "bleu", "em_norm", "edit_similarity", "code_bleu"]
    as_metric_keys = ["exact_match", "edit_similarity", "code_bleu"]

    # ---- QA ----
    print("[qa] loading predictions ...", flush=True)
    qa_recs = score_qa(_load_preds(Path(args.qa_dir)))
    print(f"[qa] {len(qa_recs)} scored QnAs", flush=True)
    qa_repo = per_repo(qa_recs, qa_metric_keys)
    qa_rows = sorted(qa_repo.values(),
                     key=lambda r: (r["qa_quality"] if r["n"] >= args.min_qnas
                                    else -1),
                     reverse=True)
    _write_csv(out / "per_repo_qa.csv", qa_rows,
               ["repo_id", "n", "n_commits"] + qa_metric_keys)
    with open(out / "records_qa_scored.jsonl", "w") as fh:
        for r in qa_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Assertions ----
    as_repo: Dict[str, Dict[str, Any]] = {}
    as_recs: List[Dict[str, Any]] = []
    if args.assert_dir:
        print("[assert] loading predictions ...", flush=True)
        as_recs = score_assert(_load_preds(Path(args.assert_dir)))
        print(f"[assert] {len(as_recs)} scored QnAs", flush=True)
        as_repo = per_repo(as_recs, as_metric_keys)
        as_rows = sorted(as_repo.values(),
                         key=lambda r: (r["exact_match"] if r["n"] >= args.min_qnas
                                        else -1),
                         reverse=True)
        _write_csv(out / "per_repo_assert.csv", as_rows,
                   ["repo_id", "n", "n_commits"] + as_metric_keys)
        with open(out / "records_assert_scored.jsonl", "w") as fh:
            for r in as_recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Combined demo ranking (repos strong on BOTH) ----
    combined = []
    for repo, q in qa_repo.items():
        if q["n"] < args.min_qnas:
            continue
        a = as_repo.get(repo)
        row = {
            "repo_id": repo,
            "qa_quality": q["qa_quality"],
            "qa_rougeL": q["rougeL"],
            "qa_n": q["n"],
            "assert_em": (a["exact_match"] if a else float("nan")),
            "assert_edit": (a["edit_similarity"] if a else float("nan")),
            "assert_n": (a["n"] if a else 0),
        }
        if a and a["n"] >= args.min_qnas:
            # normalized combined score: mean of qa_quality and assert EM
            row["demo_score"] = 0.5 * (q["qa_quality"] + a["exact_match"])
        else:
            row["demo_score"] = float("nan")
        combined.append(row)
    combined_valid = [r for r in combined if r["demo_score"] == r["demo_score"]]
    combined_valid.sort(key=lambda r: r["demo_score"], reverse=True)
    _write_csv(out / "demo_candidates.csv", combined_valid,
               ["repo_id", "demo_score", "qa_quality", "qa_rougeL", "qa_n",
                "assert_em", "assert_edit", "assert_n"])

    # ---- Markdown summary + best examples ----
    md: List[str] = ["# Code2LoRA-GRU multi-task demo selection\n"]
    md.append(f"QA QnAs scored: {len(qa_recs):,}  |  "
              f"Assertion QnAs scored: {len(as_recs):,}\n")
    md.append("Headline metrics: **qa_quality** = mean(ROUGE-L F, token-F1); "
              "assertions = **exact_match**. demo_score = mean(qa_quality, "
              "assert_EM).\n")

    md.append("\n## Top repos for the QA task\n")
    md.append("| repo | n | qa_quality | rougeL | token_f1 | bleu | em_norm |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in qa_rows[: args.top_repos]:
        if r["n"] < args.min_qnas:
            continue
        md.append(f"| {r['repo_id']} | {r['n']} | {r['qa_quality']:.3f} | "
                  f"{r['rougeL']:.3f} | {r['token_f1']:.3f} | {r['bleu']:.3f} | "
                  f"{r['em_norm']:.3f} |")

    if as_repo:
        md.append("\n## Top repos for the assertion task\n")
        md.append("| repo | n | exact_match | edit_similarity | code_bleu |")
        md.append("|---|--:|--:|--:|--:|")
        for r in as_rows[: args.top_repos]:
            if r["n"] < args.min_qnas:
                continue
            md.append(f"| {r['repo_id']} | {r['n']} | {r['exact_match']:.3f} | "
                      f"{r['edit_similarity']:.3f} | {r['code_bleu']:.3f} |")

        md.append("\n## Best demo candidates (strong on BOTH tasks)\n")
        md.append("| repo | demo_score | qa_quality | qa_n | assert_EM | assert_n |")
        md.append("|---|--:|--:|--:|--:|--:|")
        for r in combined_valid[: args.top_repos]:
            md.append(f"| {r['repo_id']} | {r['demo_score']:.3f} | "
                      f"{r['qa_quality']:.3f} | {r['qa_n']} | "
                      f"{r['assert_em']:.3f} | {r['assert_n']} |")

    # Best examples for the top demo candidates (or top QA repos as fallback).
    lead_repos = ([r["repo_id"] for r in combined_valid[: args.top_repos]]
                  if combined_valid else
                  [r["repo_id"] for r in qa_rows[: args.top_repos]
                   if r["n"] >= args.min_qnas])
    md.append("\n## Best example QnAs for the leading demo repos\n")
    for repo in lead_repos:
        md.append(f"\n### {repo}\n")
        md.append("**QA:**\n")
        md.append(_fmt_examples_md(
            top_examples(qa_recs, repo, args.examples_per_repo), "qa"))
        if as_recs:
            md.append("\n\n**Assertions:**\n")
            md.append(_fmt_examples_md(
                top_examples(as_recs, repo, args.examples_per_repo), "assert"))

    (out / "demo_selection.md").write_text("\n".join(md))
    print(f"\nWrote:\n  {out/'per_repo_qa.csv'}\n  "
          f"{out/'per_repo_assert.csv'}\n  {out/'demo_candidates.csv'}\n  "
          f"{out/'demo_selection.md'}\n  {out/'records_qa_scored.jsonl'}\n  "
          f"{out/'records_assert_scored.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
