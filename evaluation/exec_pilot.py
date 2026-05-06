#!/usr/bin/env python3
"""
Execution-based pilot evaluation for RepoPeftBench.

Given an eval results JSON (any of the unified driver / per-baseline JSONs that
contain ``entries`` or ``per_repo`` with predictions), this script:

  1. Picks a small, controlled slice of CR-test repos for which we already have
     a clone at ``$SCRATCH/REPO_DATASET/repositories/<owner>/<repo>``.
  2. For each prediction, patches the predicted RHS of the assertion back into
     the original test source line (using the QnA-pair metadata: ``file``,
     ``lineno``, ``target_col_offset``).
  3. Runs ``pytest -k <test_function>`` in an ephemeral ``venv`` with a strict
     wall-clock timeout, capturing pass/fail/error.

Output: a JSONL with one row per QnA pair containing
``{repo, file, lineno, exec_pass, exec_error, expected, predicted}`` plus a
short summary on stdout.

We label this as a *pilot* in the paper (not a primary metric) for two reasons:
  - The harness depends on each repository's ``pytest`` configuration; some
    repos require non-trivial fixtures, network access, or external services.
  - We only run on a hand-picked, runnable slice; a full execution evaluation
    over all 6,414 CR-test pairs is out of scope for this submission.

Usage::

    python evaluation/exec_pilot.py \\
        --results $SCRATCH/BASELINES/code2lora_cr_test.json \\
        --repos-root $SCRATCH/REPO_DATASET/repositories \\
        --slice-json scripts/exec_pilot_slice.json \\
        --output exec_pilot_results.jsonl

The slice JSON is a list of ``{owner/repo: [test_function, ...]}`` so the
pilot is reproducible from a fixed manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def patch_assertion(
    src_path: Path,
    lineno: int,
    target_col_offset: int,
    predicted_target: str,
) -> str:
    """Return new file contents with ``predicted_target`` spliced into the
    original target's column on ``lineno``.

    We replace from ``target_col_offset`` to the end of that line. This works
    for the typical single-line assertion case extracted by RepoPeftBench. For
    the (rare) ``was_multiline`` cases we fall back to replacing the entire
    line after ``target_col_offset``; this is a known limitation of the pilot.
    """
    text = src_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if lineno <= 0 or lineno > len(lines):
        raise ValueError(f"lineno {lineno} out of range for {src_path}")
    line = lines[lineno - 1]
    # split the line at target_col_offset; preserve trailing newline.
    nl = "\n" if line.endswith("\n") else ""
    head = line[:target_col_offset]
    new_line = head + predicted_target.rstrip("\n") + nl
    lines[lineno - 1] = new_line
    return "".join(lines)


def run_pytest(
    repo_dir: Path,
    rel_test_file: str,
    test_function: str,
    timeout_s: int,
    venv_python: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Run ``pytest -k <test_function> <test_file>`` from ``repo_dir`` and
    return ``(passed, message)``.

    ``passed`` is True iff pytest exit code is 0 and at least one test was
    collected and passed.
    """
    py = str(venv_python) if venv_python else sys.executable
    cmd = [
        py, "-m", "pytest", "-q", "--tb=line",
        "-k", test_function, rel_test_file,
        "--no-header",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        passed = (proc.returncode == 0) and (" passed" in out)
        return passed, out[-2000:]  # tail
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"EXEC ERROR: {e!r}"


def iter_pred_pairs(results: Dict[str, Any]):
    """Yield (repo, qna_pair, prediction_string) tuples from any of our eval
    JSON layouts. Supports the legacy ``entries`` layout used by the per-baseline
    scripts AND the unified-driver ``per_repo`` layout.
    """
    if "entries" in results and isinstance(results["entries"], list):
        for e in results["entries"]:
            yield (
                e.get("repo"),
                {  # synthesize a QnA-shaped dict for downstream code
                    "target": e.get("expected"),
                    "metadata": e.get("metadata") or {},
                    "predicted": e.get("got"),
                },
                e.get("got") or "",
            )
        return
    pr = results.get("per_repo") or {}
    for rid, info in pr.items():
        for e in info.get("entries", []) or []:
            yield (rid, e, e.get("predicted") or e.get("got") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="Eval results JSON containing predictions.")
    ap.add_argument(
        "--repos-root", type=Path,
        default=Path(os.environ.get("SCRATCH", ".")) / "REPO_DATASET" / "repositories",
        help="Root directory of cloned repositories; expected layout "
             "<repos-root>/<owner>/<repo>",
    )
    ap.add_argument(
        "--slice-json", type=Path, required=True,
        help="JSON manifest of repos to include: {repo_id: [test_function, ...]}.",
    )
    ap.add_argument(
        "--venv-python", type=Path, default=None,
        help="Optional path to a python in a venv with pytest installed and "
             "all repo-level requirements present. If unset, uses the current "
             "interpreter.",
    )
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-pairs-per-repo", type=int, default=20)
    args = ap.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    slice_manifest = json.loads(args.slice_json.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")

    n_total = 0
    n_pass = 0
    n_skipped = 0
    n_per_repo: Dict[str, int] = {}
    t0 = time.time()

    for repo, entry, pred in iter_pred_pairs(results):
        if not repo or repo not in slice_manifest:
            continue
        md = entry.get("metadata") or {}
        tfn = md.get("test_function")
        if not tfn or tfn not in slice_manifest[repo]:
            continue
        if n_per_repo.get(repo, 0) >= args.max_pairs_per_repo:
            continue

        rel_file = md.get("file")
        lineno = md.get("lineno")
        tco = md.get("target_col_offset")
        if not (rel_file and lineno and tco is not None):
            n_skipped += 1
            continue

        owner, name = repo.split("/", 1)
        repo_dir = (args.repos_root / owner / name).resolve()
        src_path = repo_dir / rel_file
        if not src_path.exists():
            n_skipped += 1
            continue

        # Patch the file in a tmp copy of the repo (avoid corrupting the
        # cached clone). We use a per-pair tmp dir; this is slow but safe.
        with tempfile.TemporaryDirectory(prefix="execpilot_") as td:
            scratch_repo = Path(td) / name
            shutil.copytree(repo_dir, scratch_repo, symlinks=True,
                            ignore=shutil.ignore_patterns(".git"))
            try:
                new_text = patch_assertion(
                    scratch_repo / rel_file, int(lineno), int(tco), pred,
                )
                (scratch_repo / rel_file).write_text(new_text, encoding="utf-8")
            except Exception as e:
                row = {
                    "repo": repo, "file": rel_file, "lineno": lineno,
                    "exec_pass": False, "exec_error": f"PATCH: {e!r}",
                    "expected": entry.get("target") or entry.get("expected"),
                    "predicted": pred, "test_function": tfn,
                }
                out_f.write(json.dumps(row) + "\n")
                n_skipped += 1
                continue

            passed, msg = run_pytest(
                scratch_repo, rel_file, tfn,
                timeout_s=args.timeout_s,
                venv_python=args.venv_python,
            )
            row = {
                "repo": repo, "file": rel_file, "lineno": lineno,
                "exec_pass": bool(passed),
                "exec_error": None if passed else msg,
                "expected": entry.get("target") or entry.get("expected"),
                "predicted": pred,
                "test_function": tfn,
            }
            out_f.write(json.dumps(row) + "\n")

            n_total += 1
            if passed:
                n_pass += 1
            n_per_repo[repo] = n_per_repo.get(repo, 0) + 1

    out_f.close()
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("Execution-based pilot summary")
    print("=" * 60)
    print(f"  pairs evaluated: {n_total}")
    print(f"  pairs passed:    {n_pass}")
    print(f"  pairs skipped:   {n_skipped}")
    if n_total:
        print(f"  pilot pass-rate: {100.0 * n_pass / n_total:.2f}%")
    print(f"  elapsed:         {elapsed:.0f}s")
    print(f"  written -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
