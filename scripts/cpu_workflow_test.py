"""End-to-end CPU smoke test for the c2l quantized workflow.

Runs the full user path on a CPU-only node:

  1. generate a repo-conditioned adapter (encoder + GRU + head; no base LLM),
  2. export it to a standard PEFT adapter,
  3. verify export fidelity vs the native C2L injection (fp32),
  4. run base-vs-adapted inference on the CPU (hf / fp32),
  5. attempt the bitsandbytes 4-bit backend on CPU and report the outcome.

Designed to run fully offline (HF cache pre-populated on the login node).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local git repo path")
    ap.add_argument("--task", default="assert_rhs")
    ap.add_argument("--out", default="/tmp/c2l_cpu_adapter")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--try-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(torch.get_num_threads())
    log(f"torch {torch.__version__} | cuda available={torch.cuda.is_available()} "
        f"| threads={torch.get_num_threads()}")

    from c2l.config import load_config
    from c2l.pipeline import AdapterGenerator
    from c2l.export import export_peft_adapter

    cfg = load_config(device="cpu")
    log(f"base_model={cfg.base_model}  encoder={cfg.encoder_model}")

    # ---- 1. generate adapter (the cheap, CPU-feasible half) ----
    t0 = time.time()
    gen = AdapterGenerator(cfg)
    adapter = gen.generate(Path(args.repo), repo_id=Path(args.repo).name,
                           task=args.task, progress=lambda m, f: log(f"  gen: {m}"))
    log(f"generated adapter in {time.time() - t0:.1f}s | "
        f"fingerprint={adapter.fingerprint()} | walked={adapter.n_commits_walked} "
        f"commits | task_conditioned={adapter.task_conditioned}")

    # ---- 2. export to a standard PEFT adapter ----
    out = export_peft_adapter(adapter, args.out)
    files = sorted(p.name for p in Path(out).iterdir())
    log(f"exported PEFT adapter -> {out}  ({files})")

    # ---- 3. fidelity check (native injection vs exported PEFT, fp32) ----
    result = gen.last_result
    qna = None
    for c in result.commits:
        if c.in_repo_split in ("val", "test") and c.qnas:
            qna = c.qnas[0]
            break
    if qna is None and result.commits:
        for c in result.commits:
            if c.qnas:
                qna = c.qnas[0]
                break
    if qna is None:
        log("no QnA available to score; stopping after export.")
        return 0
    sample_prefix = qna.prefix

    from c2l.infer import verify_export_fidelity
    log("running export-fidelity check (fp32)…")
    fid = verify_export_fidelity(adapter, sample_prefix=sample_prefix[-512:], config=cfg)
    log(f"fidelity: max_abs_logit_diff={fid['max_abs_logit_diff']:.3e} "
        f"argmax_match={fid['argmax_match']}")

    # ---- 4. base vs adapted on CPU (hf / fp32) ----
    from c2l.infer import HFInference
    from c2l.metrics import compute_metrics
    log("loading base model on CPU for hf inference…")
    eng = HFInference(adapter_dir=str(out), quantize=None, config=cfg)
    t0 = time.time()
    base = eng.generate(qna.prefix, max_new_tokens=args.max_new_tokens, use_adapter=False)
    adapted = eng.generate(qna.prefix, max_new_tokens=args.max_new_tokens, use_adapter=True)
    log(f"hf inference done in {time.time() - t0:.1f}s")
    bm = compute_metrics(base, qna.target)
    am = compute_metrics(adapted, qna.target)
    print("\n================ PREDICTION (CPU, fp32) ================", flush=True)
    print(f"  test_file : {qna.test_file}  L{qna.lineno}", flush=True)
    print(f"  target    : {qna.target!r}", flush=True)
    print(f"  base      : {bm['pred_clean']!r}  (EM={bm['exact_match']})", flush=True)
    print(f"  adapted   : {am['pred_clean']!r}  (EM={am['exact_match']})", flush=True)
    print("========================================================\n", flush=True)

    # ---- 5. optional: bitsandbytes 4-bit on CPU ----
    if args.try_4bit:
        log("attempting bitsandbytes 4-bit backend on CPU…")
        try:
            eng4 = HFInference(adapter_dir=str(out), quantize="4bit", config=cfg)
            t0 = time.time()
            pred4 = eng4.generate(qna.prefix, max_new_tokens=args.max_new_tokens,
                                  use_adapter=True)
            m4 = compute_metrics(pred4, qna.target)
            log(f"4-bit CPU inference OK in {time.time() - t0:.1f}s | "
                f"pred={m4['pred_clean']!r} EM={m4['exact_match']}")
        except Exception as e:
            log(f"4-bit on CPU NOT supported here: {type(e).__name__}: {e}")
            log("  -> use the GGUF backend (llama.cpp) for true CPU quantization.")

    log("CPU workflow test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
