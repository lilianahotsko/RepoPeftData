#!/usr/bin/env python3
"""GGUF (Q4_K_M, llama.cpp) evaluation + CPU latency for the demo repo.

Serves the shipped multi-task checkpoint the way the live demo does: a
quantized base GGUF (Q4_K_M) with a per-commit, per-task LoRA GGUF applied at
runtime via ``llama-completion -no-cnv`` (raw completion, no chat template).

For the demo repo (0xricksanchez/like-dbg) at both demo commits it runs every
assertion (task=assert_rhs, 16 new tokens) and every QA pair (task=qa, 64 new
tokens), then reports:

  * quality  -- assertion exact-match / edit-similarity; QA ROUGE-L / token-F1
  * latency  -- llama.cpp per-query timings (prompt prefill t/s, decode t/s,
                ms/token) which exclude model load, plus cold end-to-end wall.

Everything runs on CPU (no GPU needed); point ``--threads`` at the CPU node's
core count.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

# ---- text metrics ---------------------------------------------------------

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip()).strip()


def _tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())


def exact_match(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


def edit_sim(pred: str, gold: str) -> float:
    return difflib.SequenceMatcher(None, _norm(pred), _norm(gold)).ratio()


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common: Dict[str, int] = {}
    gp: Dict[str, int] = {}
    for t in p:
        gp[t] = gp.get(t, 0) + 1
    gg: Dict[str, int] = {}
    for t in g:
        gg[t] = gg.get(t, 0) + 1
    overlap = sum(min(gp[t], gg.get(t, 0)) for t in gp)
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    # LCS length
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        pi = p[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            row[j] = prev[j - 1] + 1 if pi == g[j - 1] else max(prev[j], row[j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


# ---- llama.cpp runner -----------------------------------------------------

_PROMPT_TPS = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second")
_EVAL_TPS = re.compile(r"[^t]eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second")


def run_completion(cli: str, base: str, lora: str, prefix: str, n_predict: int,
                   threads: int, stop_newline: bool) -> Dict:
    cmd = [cli, "-m", base, "--lora", lora, "-p", prefix,
           "-n", str(int(n_predict)), "--no-display-prompt", "--temp", "0",
           "-no-cnv", "-t", str(int(threads))]
    if stop_newline:
        cmd += ["-r", "\n"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=600)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"llama-completion failed (rc={res.returncode}):\n"
                           f"{res.stderr[-1500:]}")
    err = res.stderr
    out = res.stdout.strip()

    def _grab(rx) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        m = rx.search(err)
        if not m:
            return None, None, None
        return float(m.group(1)), int(m.group(2)), float(m.group(3))

    p_ms, p_tok, p_tps = _grab(_PROMPT_TPS)
    e_ms, e_tok, e_tps = _grab(_EVAL_TPS)
    return {
        "pred": out,
        "wall_s": wall,
        "prompt_tokens": p_tok, "prompt_ms": p_ms, "prompt_tps": p_tps,
        "gen_tokens": e_tok, "gen_ms": e_ms, "gen_tps": e_tps,
        "ms_per_gen_token": (e_ms / e_tok) if (e_ms and e_tok) else None,
    }


# ---- driver ---------------------------------------------------------------

def eval_commit(cli, base, pairs_path: Path, adapters_dir: Path, threads: int,
                assert_tokens: int, qa_tokens: int) -> Dict:
    data = json.loads(pairs_path.read_text())
    commit = data["commit"]
    out = {"repo": data["repo"], "commit": commit,
           "assertions": [], "qa": []}

    lora_assert = str(adapters_dir / "assert" / "adapter.gguf")
    lora_qa = str(adapters_dir / "qa" / "adapter.gguf")

    for i, a in enumerate(data.get("assertions", [])):
        r = run_completion(cli, base, lora_assert, a["prefix"], assert_tokens,
                           threads, stop_newline=True)
        r.update({
            "idx": i, "category": a.get("category"),
            "gold": a["target"],
            "exact_match": exact_match(r["pred"], a["target"]),
            "edit_sim": edit_sim(r["pred"], a["target"]),
        })
        out["assertions"].append(r)
        print(f"[{commit[:10]}] A{i+1}/{len(data['assertions'])} "
              f"EM={r['exact_match']:.0f} pred={r['pred'][:40]!r} gold={a['target'][:30]!r} "
              f"gen={r['gen_tps']}t/s", flush=True)

    for i, q in enumerate(data.get("qa", [])):
        r = run_completion(cli, base, lora_qa, q["prefix"], qa_tokens,
                           threads, stop_newline=False)
        r.update({
            "idx": i, "qa_kind": q.get("qa_kind"), "category": q.get("category"),
            "gold": q["target"],
            "rouge_l": rouge_l(r["pred"], q["target"]),
            "token_f1": token_f1(r["pred"], q["target"]),
        })
        out["qa"].append(r)
        print(f"[{commit[:10]}] Q{i+1}/{len(data['qa'])} "
              f"RgL={r['rouge_l']:.3f} F1={r['token_f1']:.3f} "
              f"gen={r['gen_tps']}t/s", flush=True)
    return out


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": mean(vals), "median": median(vals), "min": min(vals),
            "max": max(vals), "n": len(vals)}


def summarize(commits: List[Dict]) -> Dict:
    all_assert = [a for c in commits for a in c["assertions"]]
    all_qa = [q for c in commits for q in c["qa"]]
    lat_prompt = [x["prompt_tps"] for x in all_assert + all_qa]
    lat_gen = [x["gen_tps"] for x in all_assert + all_qa]
    lat_mstok = [x["ms_per_gen_token"] for x in all_assert + all_qa]
    lat_wall = [x["wall_s"] for x in all_assert + all_qa]
    return {
        "n_assertions": len(all_assert),
        "n_qa": len(all_qa),
        "assert_exact_match": mean(a["exact_match"] for a in all_assert) if all_assert else None,
        "assert_edit_sim": mean(a["edit_sim"] for a in all_assert) if all_assert else None,
        "qa_rouge_l": mean(q["rouge_l"] for q in all_qa) if all_qa else None,
        "qa_token_f1": mean(q["token_f1"] for q in all_qa) if all_qa else None,
        "latency": {
            "prompt_prefill_tps": _agg(lat_prompt),
            "decode_tps": _agg(lat_gen),
            "ms_per_gen_token": _agg(lat_mstok),
            "cold_end_to_end_wall_s": _agg(lat_wall),
            "assert_decode_tps": _agg([a["gen_tps"] for a in all_assert]),
            "qa_decode_tps": _agg([q["gen_tps"] for q in all_qa]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-completion",
                    default="/scratch/lhotsko/llama.cpp/build/bin/llama-completion")
    ap.add_argument("--base-gguf",
                    default="/scratch/lhotsko/models/qwen-coder-gguf/qwen2.5-coder-1.5b-q4_k_m.gguf")
    ap.add_argument("--demo-dir", default="/scratch/lhotsko/demo_repos")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--assert-tokens", type=int, default=16)
    ap.add_argument("--qa-tokens", type=int, default=64)
    ap.add_argument("--out", default="/scratch/lhotsko/demo_repos/gguf_eval_likedbg.json")
    args = ap.parse_args()

    demo = Path(args.demo_dir)
    commits = [
        ("early", demo / "pairs_522c24530a.json", demo / "adapters_522c24530a"),
        ("later", demo / "pairs_6432fa835f.json", demo / "adapters_6432fa835f"),
    ]
    results = []
    for tag, pairs, adir in commits:
        print(f"\n===== {tag.upper()} COMMIT  ({pairs.name}) =====", flush=True)
        results.append(eval_commit(args.llama_completion, args.base_gguf, pairs,
                                    adir, args.threads,
                                    args.assert_tokens, args.qa_tokens))

    summary = summarize(results)
    payload = {
        "config": {
            "backend": "gguf (Q4_K_M, llama.cpp runtime LoRA)",
            "base_gguf": args.base_gguf,
            "threads": args.threads,
            "assert_tokens": args.assert_tokens,
            "qa_tokens": args.qa_tokens,
        },
        "summary": summary,
        "commits": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("\n================ SUMMARY (GGUF Q4_K_M, CPU) ================")
    print(f"assertions n={summary['n_assertions']}  "
          f"EM={summary['assert_exact_match']:.3f}  "
          f"editsim={summary['assert_edit_sim']:.3f}")
    print(f"qa         n={summary['n_qa']}  "
          f"ROUGE-L={summary['qa_rouge_l']:.3f}  "
          f"token-F1={summary['qa_token_f1']:.3f}")
    lat = summary["latency"]
    def _p(d):
        return f"mean={d['mean']:.1f} median={d['median']:.1f} (n={d['n']})" if d else "n/a"
    print("-- latency (llama.cpp perf, load excluded) --")
    print(f"  prompt prefill  t/s : {_p(lat['prompt_prefill_tps'])}")
    print(f"  decode          t/s : {_p(lat['decode_tps'])}")
    print(f"    assert decode t/s : {_p(lat['assert_decode_tps'])}")
    print(f"    qa     decode t/s : {_p(lat['qa_decode_tps'])}")
    print(f"  ms / generated tok  : {_p(lat['ms_per_gen_token'])}")
    print(f"  cold end-to-end (s) : {_p(lat['cold_end_to_end_wall_s'])}  [incl. model+LoRA load per call]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
