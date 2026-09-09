#!/usr/bin/env python3
"""LLM-as-judge evaluation of repository-QA answers.

Fills the human-eval annotation sheet (see
Code2LoRADemo/human_eval/qa_human_eval_sample.csv) with ratings from an
OpenAI judge model, using the same rubric a human annotator would:

  correctness: 2 correct / 1 partially correct / 0 incorrect
  grounding:   1 repo-specific / 0 generic or hallucinated

Outputs a filled CSV compatible with evaluation/score_human_eval.py (so two
judge runs, e.g. different models, can be compared with Cohen's kappa) plus a
JSONL with the raw judgments including rationales.

Usage:
  python evaluation/llm_judge_qa.py \
      --sheet Code2LoRADemo/human_eval/qa_human_eval_sample.csv \
      --out   Code2LoRADemo/human_eval/qa_llm_judge_gpt54.csv \
      --model gpt-5.4

The OpenAI key is read from OPENAI_API_KEY (or repo-root .env).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from create_dataset.generate_qna_batch import get_client  # noqa: E402

SYSTEM_PROMPT = """\
You are an expert software engineer evaluating answers produced by a coding
assistant. Each item has a QUESTION about a specific commit in a software
repository, a REFERENCE answer (written by a strong model that saw the actual
commit diff), and a MODEL ANSWER produced by the system under evaluation
(which saw only a learned representation of the repository history).

Rate the MODEL ANSWER on two axes:

correctness (compare against the REFERENCE):
  2 = correct: factually consistent with the reference; wording and level of
      detail may differ.
  1 = partially correct: captures the main point but has a factual error,
      an important omission, or unwarranted invented specifics.
  0 = incorrect: contradicts the reference or answers a different question.

grounding (is the answer about *this* repository?):
  1 = repo-specific: names files, APIs, identifiers, or behaviors plausibly
      belonging to this repository and consistent with the reference.
  0 = generic or hallucinated: could apply to almost any project, or invents
      entities unsupported by the reference.

Judge only against the given reference; do not reward verbosity. Respond with
a JSON object: {"correctness": 0|1|2, "grounding": 0|1, "rationale": "<one
sentence>"}."""


def build_user_prompt(row: dict) -> str:
    return (
        f"REPOSITORY: {row['repo']}\n\n"
        f"QUESTION:\n{row['question']}\n\n"
        f"REFERENCE:\n{row['reference_answer']}\n\n"
        f"MODEL ANSWER:\n{row['model_answer']}"
    )


def judge_one(client, model: str, reasoning_effort: str, row: dict,
              retries: int = 4) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 2048,
    }
    if reasoning_effort and reasoning_effort.lower() != "none":
        body["reasoning_effort"] = reasoning_effort
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**body)
            out = json.loads(resp.choices[0].message.content)
            c, g = int(out["correctness"]), int(out["grounding"])
            if c not in (0, 1, 2) or g not in (0, 1):
                raise ValueError(f"out-of-range judgment: {out}")
            return {
                "item_id": row["item_id"],
                "correctness": c,
                "grounding": g,
                "rationale": str(out.get("rationale", "")),
            }
        except Exception as e:  # noqa: BLE001 - retry API/parse errors alike
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"{row['item_id']}: giving up after {retries} tries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="blind annotation CSV")
    ap.add_argument("--out", required=True, help="filled CSV to write")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    with open(args.sheet) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    corr_col = next(c for c in fieldnames if c.startswith("correctness"))
    grnd_col = next(c for c in fieldnames if c.startswith("grounding"))
    print(f"{len(rows)} items -> judge {args.model} "
          f"(effort={args.reasoning_effort}, {args.concurrency} workers)")

    client = get_client()
    judgments: dict[str, dict] = {}
    lock = threading.Lock()

    def work(row: dict) -> None:
        j = judge_one(client, args.model, args.reasoning_effort, row)
        with lock:
            judgments[j["item_id"]] = j
            if len(judgments) % 20 == 0:
                print(f"  {len(judgments)}/{len(rows)} judged")

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, rows))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            j = judgments[row["item_id"]]
            row[corr_col] = str(j["correctness"])
            row[grnd_col] = str(j["grounding"])
            row["notes"] = j["rationale"]
            w.writerow(row)

    raw_path = out_path.with_suffix(".jsonl")
    with open(raw_path, "w") as f:
        for row in rows:
            f.write(json.dumps({**judgments[row["item_id"]],
                                "judge_model": args.model}) + "\n")

    corr = [judgments[r["item_id"]]["correctness"] for r in rows]
    grnd = [judgments[r["item_id"]]["grounding"] for r in rows]
    n = len(corr)
    print(f"\nwrote {out_path} and {raw_path}")
    print(f"correctness: 2={corr.count(2)} ({corr.count(2)/n:.0%})  "
          f"1={corr.count(1)} ({corr.count(1)/n:.0%})  "
          f"0={corr.count(0)} ({corr.count(0)/n:.0%})  mean={sum(corr)/n:.2f}")
    print(f"grounding:   repo-specific={sum(grnd)}/{n} ({sum(grnd)/n:.0%})")


if __name__ == "__main__":
    main()
