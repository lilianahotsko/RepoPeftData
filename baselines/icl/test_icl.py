#!/usr/bin/env python3
"""
In-Context Learning (ICL) baseline: provide few-shot assertion completion
examples in the prompt before the test prefix.

For IR evaluation: examples come from the same repo (different QnA pairs).
For CR evaluation: examples come from the most similar training repo
(by embedding cosine similarity).

Usage:
    python baselines/icl/test_icl.py --split cr_test --n-shots 5
    python baselines/icl/test_icl.py --split ir_test --n-shots 3
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import postprocess_prediction, exact_match, edit_similarity, code_bleu_score, strip_comments
from evaluation.data_utils import get_default_splits_dir, get_bos_id


def load_split_by_repo(splits_dir: Path, split_name: str) -> dict:
    """Load split JSON grouped by repo. Returns {repo: {pairs, embedding}}."""
    path = splits_dir / f"{split_name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repositories", {})
    result = {}
    for repo_name, r in repos.items():
        pairs = []
        for p in r.get("qna_pairs", []):
            prefix = p.get("prefix", "")
            target = p.get("target", "")
            if prefix and target and not target.lstrip().startswith(","):
                pairs.append({"prefix": prefix, "target": target})
        if pairs:
            result[repo_name] = {
                "pairs": pairs,
                "embedding": r.get("embedding"),
            }
    return result


def find_nearest_train_repo(test_embedding: list, train_repos: dict) -> str:
    """Find the training repo with highest cosine similarity."""
    test_emb = torch.tensor(test_embedding, dtype=torch.float32)
    test_emb = F.normalize(test_emb.unsqueeze(0), p=2, dim=-1)

    best_repo = None
    best_sim = -1.0
    for repo_name, r in train_repos.items():
        if r["embedding"] is None:
            continue
        train_emb = torch.tensor(r["embedding"], dtype=torch.float32)
        train_emb = F.normalize(train_emb.unsqueeze(0), p=2, dim=-1)
        sim = (test_emb @ train_emb.T).item()
        if sim > best_sim:
            best_sim = sim
            best_repo = repo_name
    return best_repo


def format_icl_prompt(prefix: str, examples: list[dict], n_shots: int, max_example_chars: int = 1000) -> str:
    """Format few-shot prompt with examples followed by the test prefix."""
    selected = examples[:n_shots]
    parts = []
    for ex in selected:
        ex_prefix = ex["prefix"]
        ex_target = ex["target"]
        # Truncate long prefixes for examples to save context
        if len(ex_prefix) > max_example_chars:
            ex_prefix = ex_prefix[-max_example_chars:]
        parts.append(f"{ex_prefix}{ex_target}")

    if parts:
        examples_str = "\n\n# ---\n\n".join(parts)
        return f"# Examples of assertion completions:\n\n{examples_str}\n\n# ---\n\n# Now complete the following assertion:\n{prefix}"
    return prefix


def main():
    ap = argparse.ArgumentParser(description="ICL baseline evaluation")
    default_dataset = get_default_splits_dir()

    ap.add_argument("--splits-dir", type=str, default=default_dataset)
    ap.add_argument("--split", type=str, default="cr_test_structured")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--n-shots", type=int, default=5, help="Number of few-shot examples")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-repos", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    splits_dir = Path(args.splits_dir).expanduser().resolve()

    # Load test split
    test_repos = load_split_by_repo(splits_dir, args.split)
    if not test_repos:
        raise ValueError(f"No repos in {args.split}.json")

    repo_names = sorted(test_repos.keys())
    if args.limit_repos:
        repo_names = repo_names[:args.limit_repos]

    # Load training data for finding similar repos / drawing examples
    train_repos = load_split_by_repo(splits_dir, "train")

    # For IR splits, examples come from the same repo's training data
    is_ir = args.split.startswith("ir_")

    # Build flat item list
    items = []
    for repo in repo_names:
        r = test_repos[repo]
        for p in r["pairs"]:
            items.append({
                "repo": repo,
                "prefix": p["prefix"],
                "target": p["target"],
                "embedding": r["embedding"],
            })
    if args.limit is not None and args.limit > 0:
        items = items[:args.limit]

    if not items:
        raise ValueError(f"No items to evaluate")

    print(f"Loading model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": args.device},
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    bos_id = get_bos_id(tok)

    # Pre-compute nearest train repos for CR evaluation
    nearest_cache = {}
    if not is_ir:
        print("Finding nearest training repos for CR evaluation...")
        for repo in repo_names:
            emb = test_repos[repo]["embedding"]
            if emb:
                nearest = find_nearest_train_repo(emb, train_repos)
                nearest_cache[repo] = nearest
                print(f"  {repo} -> {nearest}")

    em_count = 0
    bleu_sum = 0.0
    edit_sum = 0.0
    entries = []
    n = len(items)

    print(f"\nEvaluating {n} examples (n_shots={args.n_shots})...")
    for i, it in enumerate(items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{n}...", flush=True)

        prefix = it["prefix"]
        target = it["target"]
        repo = it["repo"]

        # Select few-shot examples
        if is_ir and repo in train_repos:
            example_pool = train_repos[repo]["pairs"]
        elif repo in nearest_cache and nearest_cache[repo] in train_repos:
            example_pool = train_repos[nearest_cache[repo]]["pairs"]
        else:
            example_pool = []

        # Randomly sample examples (avoid using the exact same prefix)
        candidates = [ex for ex in example_pool if ex["prefix"] != prefix]
        if len(candidates) > args.n_shots:
            examples = random.sample(candidates, args.n_shots)
        else:
            examples = candidates

        icl_prompt = format_icl_prompt(prefix, examples, args.n_shots)

        prefix_ids = tok.encode(icl_prompt, add_special_tokens=False)
        input_ids = [bos_id] + prefix_ids
        if len(input_ids) > args.max_input_tokens:
            input_ids = input_ids[-args.max_input_tokens:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model.generate(
                input_t, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][len(input_ids):].tolist()
        pred = tok.decode(gen_ids, skip_special_tokens=True)

        pred_clean = postprocess_prediction(pred, target)
        target_clean = strip_comments(target)

        em = exact_match(pred_clean, target_clean)
        bleu = code_bleu_score(pred_clean, target_clean)
        edit_sim = edit_similarity(pred_clean, target_clean)
        em_count += 1 if em else 0
        bleu_sum += bleu
        edit_sum += edit_sim

        entries.append({
            "repo": repo, "expected": target_clean, "got": pred_clean,
            "exact_match": em, "code_bleu": bleu, "edit_similarity": edit_sim,
            "n_shots_used": len(examples),
        })

    exact_match_pct = 100.0 * em_count / n
    code_bleu_avg = bleu_sum / n
    edit_sim_avg = edit_sum / n

    results = {
        "method": f"icl_{args.n_shots}shot",
        "split": args.split,
        "exact_match_pct": exact_match_pct,
        "exact_match_count": em_count,
        "n": n,
        "code_bleu": code_bleu_avg,
        "edit_similarity": edit_sim_avg,
        "config": {"n_shots": args.n_shots, "model_name": args.model_name},
        "entries": entries,
    }

    if args.output:
        results_path = Path(args.output).expanduser().resolve()
    else:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
        results_path = Path(scratch) / "BASELINES" / f"icl_{args.n_shots}shot_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"ICL Baseline ({args.n_shots}-shot) on {args.split}")
    print("=" * 60)
    print(f"  Exact Match:     {exact_match_pct:.2f}% ({em_count}/{n})")
    print(f"  Code BLEU:       {code_bleu_avg:.4f}")
    print(f"  Edit Similarity: {edit_sim_avg:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
