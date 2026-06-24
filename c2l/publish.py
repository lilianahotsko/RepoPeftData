"""Gate a trained checkpoint on the eval harness, then publish it to the Hub.

This is the "easy to manage" release control: a multi-task checkpoint is only
published if it clears per-suite (and optionally per-task) exact-match
thresholds, and it is published as a *new versioned revision* so existing
deployments keep pinning the checkpoint they were validated against.

It shells out to the existing evaluator
(``evaluation/run_code2lora_gru_v2_eval.py``) so scoring is identical to the
paper pipeline, reads the summary JSON it writes, checks thresholds, and only
then uploads via ``huggingface_hub``.

Example::

    python -m c2l.publish \
        --checkpoint runs/gru_multitask/gru_head.best.pt \
        --commits-dir $SCRATCH/REPO_DATASET/commit_parquet_hf_v2 \
        --qnas-dir    $SCRATCH/REPO_DATASET/commit_parquet_hf_v2 \
        --suites cr_test ir_test --tasks assert_rhs code_gen \
        --output-dir runs/gate --min-exact-match 0.30 \
        --repo code2lora/code2lora-gru --revision v2-multitask --publish
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "evaluation" / "run_code2lora_gru_v2_eval.py"


def run_eval_suite(checkpoint: str, commits_dir: str, qnas_dir: str,
                   suite: str, tasks: List[str], output_dir: str,
                   base_model: Optional[str] = None,
                   extra: Optional[List[str]] = None) -> Path:
    """Run the evaluator for one suite; return the summary JSON path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--checkpoint", checkpoint,
        "--commits-dir", commits_dir,
        "--qnas-dir", qnas_dir,
        "--suite", suite,
        "--tasks", *tasks,
        "--output-dir", str(out_dir),
    ]
    if base_model:
        cmd += ["--base-model", base_model]
    if extra:
        cmd += extra
    print("[gate] running:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"eval failed for suite {suite!r} (rc={res.returncode})")
    # The evaluator writes <suite>*.json into output-dir; pick the newest.
    candidates = sorted(out_dir.glob(f"*{suite}*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(out_dir.glob("*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no eval JSON written to {out_dir}")
    return candidates[0]


def read_exact_match(summary_json: Path) -> float:
    data = json.loads(Path(summary_json).read_text())
    summary = data.get("summary", data)
    return float(summary.get("exact_match", 0.0))


def gate(checkpoint: str, commits_dir: str, qnas_dir: str, suites: List[str],
         tasks: List[str], output_dir: str, min_exact_match: float,
         base_model: Optional[str] = None,
         extra: Optional[List[str]] = None) -> Dict[str, float]:
    """Evaluate every suite and return {suite: exact_match}; raise if any fails."""
    scores: Dict[str, float] = {}
    failures: List[str] = []
    for suite in suites:
        jp = run_eval_suite(checkpoint, commits_dir, qnas_dir, suite, tasks,
                            output_dir, base_model=base_model, extra=extra)
        em = read_exact_match(jp)
        scores[suite] = em
        status = "PASS" if em >= min_exact_match else "FAIL"
        print(f"[gate] {suite}: exact_match={em:.4f} "
              f"(threshold {min_exact_match:.4f}) -> {status}", flush=True)
        if em < min_exact_match:
            failures.append(suite)
    if failures:
        raise RuntimeError(
            f"gate failed on suites {failures}; not publishing. Scores={scores}")
    return scores


def publish(checkpoint: str, repo: str, revision: str,
            scores: Optional[Dict[str, float]] = None,
            ckpt_filename: str = "code2lora_gru.pt") -> str:
    """Upload the checkpoint to the Hub as a new versioned revision (branch)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    try:
        api.create_branch(repo_id=repo, branch=revision, exist_ok=True)
    except Exception as e:
        print(f"[publish] create_branch note: {e}", flush=True)
    msg = f"C2L checkpoint {revision}"
    if scores:
        msg += " | " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo=ckpt_filename,
        repo_id=repo,
        repo_type="model",
        revision=revision,
        commit_message=msg,
    )
    url = f"https://huggingface.co/{repo}/tree/{revision}"
    print(f"[publish] uploaded to {url}", flush=True)
    return url


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--commits-dir", required=True)
    ap.add_argument("--qnas-dir", required=True)
    ap.add_argument("--suites", nargs="+", default=["cr_test"])
    ap.add_argument("--tasks", nargs="+", default=["assert_rhs"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--min-exact-match", type=float, default=0.0)
    ap.add_argument("--repo", help="HF model repo to publish to")
    ap.add_argument("--revision", default="v-next",
                    help="branch/revision name for the published checkpoint")
    ap.add_argument("--ckpt-filename", default="code2lora_gru.pt")
    ap.add_argument("--publish", action="store_true",
                    help="actually upload (otherwise dry-run after gating)")
    ap.add_argument("--eval-arg", action="append", default=[],
                    help="extra arg(s) forwarded to the evaluator, repeatable")
    args = ap.parse_args(argv)

    scores = gate(args.checkpoint, args.commits_dir, args.qnas_dir, args.suites,
                  args.tasks, args.output_dir, args.min_exact_match,
                  base_model=args.base_model, extra=args.eval_arg)
    print(f"[gate] all suites passed: {scores}", flush=True)

    if args.publish:
        if not args.repo:
            print("error: --publish requires --repo", file=sys.stderr)
            return 2
        publish(args.checkpoint, args.repo, args.revision, scores=scores,
                ckpt_filename=args.ckpt_filename)
    else:
        print("[publish] dry-run (pass --publish --repo ... to upload).",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
