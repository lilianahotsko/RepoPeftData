#!/usr/bin/env python3
"""Latency benchmark for Code2LoRA across serving backends and workflow stages.

Measures, on the demo repo (0xricksanchez/like-dbg), the wall-clock cost of the
three workflow stages a live demo exercises:

  * adapt      -- generate the per-repo LoRA from the commit history
                  (full walk / incremental one-commit / cached no-op)
  * assertion  -- assertion-completion inference (16 new tokens)
  * qa         -- repository-QA inference (64 new tokens)

for a chosen serving backend (``hf`` fp16 / ``8bit`` / ``4bit`` on GPU;
``gguf`` Q4_K_M on CPU via llama.cpp).

Inference numbers are reported *steady state* (model + adapter already loaded);
the one-time model-load cost is reported separately. Adapt numbers use the
shipped ``AdapterGenerator`` and the on-disk walk-state cache so the
full/incremental/cached distinction is exactly what the terminal does.

Run one process per (backend, device); results are written as JSON and merged
by ``latency_report.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _agg(vals: List[float]) -> Optional[Dict[str, float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": mean(vals), "median": median(vals),
            "min": min(vals), "max": max(vals), "n": len(vals)}


def _load_pairs(path: Path, n_assert: int, n_qa: int):
    data = json.loads(path.read_text())
    asserts = data.get("assertions", [])[:n_assert]
    qa = data.get("qa", [])[:n_qa]
    return asserts, qa


# --------------------------------------------------------------------------- #
# inference timing
# --------------------------------------------------------------------------- #

def bench_infer(backend: str, device: str, adapters_dir: Path,
                asserts: List[dict], qa: List[dict],
                base_gguf: Optional[str], threads: Optional[int]) -> Dict:
    """Time assertion (16 tok) and qa (64 tok) generation, load excluded."""
    import c2l.config as ccfg
    from c2l.infer import make_backend

    cfg = ccfg.load_config()
    # pin device for the transformers backends
    if backend in ("hf", "4bit", "8bit"):
        cfg.device = device

    def _engine(task: str):
        lora_dir = adapters_dir / task
        if backend == "gguf":
            return make_backend("gguf", base_gguf=base_gguf,
                                lora_gguf=str(lora_dir / "adapter.gguf"),
                                config=cfg)
        eng = make_backend(backend, adapter_dir=str(lora_dir), config=cfg)
        return eng

    out: Dict[str, object] = {"backend": backend, "device": device}

    # ---- assertion (16 tok) ----
    eng_a = _engine("assert")
    t0 = time.perf_counter()
    if hasattr(eng_a, "load"):
        eng_a.load()
    load_a = time.perf_counter() - t0
    # warm-up (kernels / caches)
    if asserts:
        eng_a.generate(asserts[0]["prefix"], max_new_tokens=16)
    a_walls = []
    for a in asserts:
        t = time.perf_counter()
        eng_a.generate(a["prefix"], max_new_tokens=16)
        a_walls.append(time.perf_counter() - t)

    # ---- qa (64 tok) ----
    eng_q = _engine("qa")
    t0 = time.perf_counter()
    if hasattr(eng_q, "load"):
        eng_q.load()
    load_q = time.perf_counter() - t0
    if qa:
        eng_q.generate(qa[0]["prefix"], max_new_tokens=64)
    q_walls = []
    for q in qa:
        t = time.perf_counter()
        eng_q.generate(q["prefix"], max_new_tokens=64)
        q_walls.append(time.perf_counter() - t)

    out["model_load_s"] = {"assert_engine": load_a, "qa_engine": load_q}
    out["assertion_16tok_s"] = _agg(a_walls)
    out["qa_64tok_s"] = _agg(q_walls)
    if a_walls:
        out["assertion_tok_per_s"] = _agg([16.0 / w for w in a_walls])
    if q_walls:
        out["qa_tok_per_s"] = _agg([64.0 / w for w in q_walls])
    return out


# --------------------------------------------------------------------------- #
# adapt timing
# --------------------------------------------------------------------------- #

def bench_adapt(device: str, repo_dir: Path, repo_id: str,
                task: str = "assert_rhs") -> Dict:
    """Time full walk / one-commit incremental / cached re-adapt."""
    import c2l.config as ccfg
    from c2l.pipeline import AdapterGenerator, _walk_state_path

    cfg = ccfg.load_config()
    cfg.device = device
    gen = AdapterGenerator(config=cfg)

    t0 = time.perf_counter()
    gen.load()
    load_s = time.perf_counter() - t0

    wpath = _walk_state_path(cfg, repo_id, task)

    def _clear_state():
        try:
            Path(wpath).unlink()
        except FileNotFoundError:
            pass

    # ---- FULL walk (fresh) ----
    _clear_state()
    t0 = time.perf_counter()
    gen.generate(repo_dir, repo_id, task=task, walk_fraction=1.0,
                 incremental=False)
    full_s = time.perf_counter() - t0
    n_kept = int(getattr(gen.last_result, "kept_commits", 0) or 0)

    # ---- CACHED (state already at endpoint -> 0 new commits) ----
    t0 = time.perf_counter()
    gen.generate(repo_dir, repo_id, task=task, walk_fraction=1.0,
                 incremental=True)
    cached_s = time.perf_counter() - t0

    # ---- INCREMENTAL (exactly one new commit) ----
    # Prime a fresh full walk that stops one commit early, then resume +1.
    incr_s = None
    if n_kept >= 2:
        wf_prev = (n_kept - 1) / n_kept   # endpoint = n_kept-2
        _clear_state()
        gen.generate(repo_dir, repo_id, task=task, walk_fraction=wf_prev,
                     incremental=False)
        t0 = time.perf_counter()
        gen.generate(repo_dir, repo_id, task=task, walk_fraction=1.0,
                     incremental=True)   # walks the final 1 commit
        incr_s = time.perf_counter() - t0
    _clear_state()

    return {
        "device": device,
        "n_kept_commits": n_kept,
        "generator_load_s": load_s,
        "adapt_full_walk_s": full_s,
        "adapt_incremental_1commit_s": incr_s,
        "adapt_cached_s": cached_s,
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf",
                    choices=["hf", "4bit", "8bit", "gguf"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mode", default="both", choices=["infer", "adapt", "both"])
    ap.add_argument("--repo-dir", default="/scratch/lhotsko/demo_repos/like-dbg")
    ap.add_argument("--repo-id", default="0xricksanchez/like-dbg")
    ap.add_argument("--pairs",
                    default="/scratch/lhotsko/demo_repos/pairs_6432fa835f.json")
    ap.add_argument("--adapters",
                    default="/scratch/lhotsko/demo_repos/adapters_6432fa835f")
    ap.add_argument("--n-assert", type=int, default=16)
    ap.add_argument("--n-qa", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--base-gguf",
                    default="/scratch/lhotsko/models/qwen-coder-gguf/"
                            "qwen2.5-coder-1.5b-q4_k_m.gguf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.backend == "gguf":
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    payload: Dict[str, object] = {
        "backend": args.backend, "device": args.device,
        "repo_id": args.repo_id, "pairs": args.pairs,
        "n_assert": args.n_assert, "n_qa": args.n_qa,
        "threads": args.threads if args.backend == "gguf" else None,
    }

    asserts, qa = _load_pairs(Path(args.pairs), args.n_assert, args.n_qa)

    if args.mode in ("infer", "both"):
        print(f"[infer] backend={args.backend} device={args.device} "
              f"n_assert={len(asserts)} n_qa={len(qa)}", flush=True)
        payload["infer"] = bench_infer(
            args.backend, args.device, Path(args.adapters),
            asserts, qa, args.base_gguf, args.threads)
        i = payload["infer"]
        print(f"  load(assert/qa) = {i['model_load_s']['assert_engine']:.1f}"
              f"/{i['model_load_s']['qa_engine']:.1f}s", flush=True)
        if i.get("assertion_16tok_s"):
            print(f"  assertion 16tok : {i['assertion_16tok_s']['mean']:.3f}s "
                  f"(median {i['assertion_16tok_s']['median']:.3f})", flush=True)
        if i.get("qa_64tok_s"):
            print(f"  qa        64tok : {i['qa_64tok_s']['mean']:.3f}s "
                  f"(median {i['qa_64tok_s']['median']:.3f})", flush=True)

    if args.mode in ("adapt", "both"):
        print(f"[adapt] device={args.device}", flush=True)
        payload["adapt"] = bench_adapt(args.device, Path(args.repo_dir),
                                       args.repo_id)
        a = payload["adapt"]
        print(f"  n_kept={a['n_kept_commits']}  full={a['adapt_full_walk_s']:.2f}s "
              f"incr={a['adapt_incremental_1commit_s']}s "
              f"cached={a['adapt_cached_s']:.2f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
