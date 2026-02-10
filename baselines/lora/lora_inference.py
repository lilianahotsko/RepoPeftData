"""
Inference script for comparing the plain base model vs LoRA-adapted model.

Usage:
    python lora_inference.py                          # Interactive mode
    python lora_inference.py --eval                    # Evaluate on test set
    python lora_inference.py --prompt "Write a function that adds two numbers"
"""

import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
ADAPTER_PATH = "./qwen-coder-lora-adapter"  # Path to saved LoRA adapter
MAX_NEW_TOKENS = 256


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_base_model(model_name, quantize=True):
    """Load the plain base model (no adapter)."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def load_lora_model(model_name, adapter_path, quantize=True):
    """Load base model + LoRA adapter on top."""
    print(f"Loading LoRA model: {model_name} + {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def build_prompt(instruction, input_text=None):
    """Build the same prompt format used during training."""
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response (code only, no explanation):\n"
    return prompt


def trim_response(text: str) -> str:
    """
    Trim the model output to just the code/answer, removing trailing
    explanations, notes, and commentary that the model tends to generate.
    """
    # 1. Cut at section markers (model repeating prompt structure)
    for marker in ["### Instruction:", "### Input:", "### Response:"]:
        if marker in text:
            text = text[: text.index(marker)]

    # 2. Cut at common explanation markers (case-insensitive line start)
    lines = text.split("\n")
    cut_markers = [
        "explanation:", "note:", "output:", "example:", "result:",
        "// explanation", "// note", "// output", "# explanation",
        "# note", "# output", "\"\"\"", "'''",
    ]
    kept_lines = []
    for line in lines:
        lower = line.strip().lower()
        # Stop if we hit a line that is purely a commentary marker
        if any(lower.startswith(m) for m in cut_markers):
            break
        kept_lines.append(line)

    # 3. Remove trailing empty lines and comment-only tails
    while kept_lines and kept_lines[-1].strip() == "":
        kept_lines.pop()

    return "\n".join(kept_lines).strip()


def generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, trim=True):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][prompt_len:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if trim:
        prediction = trim_response(prediction)

    return prediction


# ── Evaluation ────────────────────────────────────────────────────────────────
def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip() == target.strip()


def normalize_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_match(prediction: str, target: str) -> bool:
    return normalize_code(prediction) == normalize_code(target)


def prefix_match(prediction: str, target: str) -> bool:
    return prediction.strip().startswith(target.strip())


def evaluate_model(model, tokenizer, dataset, label, max_examples=200, max_new_tokens=MAX_NEW_TOKENS, print_mismatches=3):
    """Run evaluation on a dataset and print results."""
    num_examples = min(max_examples, len(dataset))
    exact_matches = 0
    normalized_matches = 0
    prefix_matches = 0
    mismatches_printed = 0

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Examples: {num_examples}")
    print(f"{'='*60}")

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(range(num_examples), desc=f"[{label}]"):
            example = dataset[i]

            prompt = build_prompt(example["instruction"], example.get("input"))
            prediction = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            ground_truth = example["output"].strip()

            is_exact = exact_match(prediction, ground_truth)
            is_normalized = normalized_match(prediction, ground_truth)
            is_prefix = prefix_match(prediction, ground_truth)

            if is_exact:
                exact_matches += 1
            if is_normalized:
                normalized_matches += 1
            if is_prefix:
                prefix_matches += 1

            if not is_exact and mismatches_printed < print_mismatches:
                mismatches_printed += 1
                print(f"\n--- Mismatch {mismatches_printed} ---")
                print(f"  INSTRUCTION: {example['instruction'][:120]}")
                print(f"  EXPECTED:    {ground_truth[:200]}")
                print(f"  PREDICTED:   {prediction[:200]}")
                print(f"  norm_match={is_normalized}, prefix_match={is_prefix}")

    em = exact_matches / num_examples * 100
    nm = normalized_matches / num_examples * 100
    pm = prefix_matches / num_examples * 100

    print(f"\n--- Results: {label} ---")
    print(f"  Exact Match:      {em:.2f}% ({exact_matches}/{num_examples})")
    print(f"  Normalized Match: {nm:.2f}% ({normalized_matches}/{num_examples})")
    print(f"  Prefix Match:     {pm:.2f}% ({prefix_matches}/{num_examples})")

    return {"exact_match": em, "normalized_match": nm, "prefix_match": pm}


# ── Interactive mode ──────────────────────────────────────────────────────────
def interactive_mode(base_model, base_tokenizer, lora_model, lora_tokenizer):
    """Run interactive inference comparing both models side by side."""
    print("\n" + "=" * 60)
    print("Interactive mode  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        print("\nEnter an instruction (or 'quit'):")
        instruction = input("> ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break

        input_text = input("Enter input (optional, press Enter to skip): ").strip() or None

        prompt = build_prompt(instruction, input_text)
        print(f"\nPrompt:\n{prompt}")

        print("\n--- Base Model ---")
        base_response = generate(base_model, base_tokenizer, prompt)
        print(base_response)

        print("\n--- LoRA Model ---")
        lora_response = generate(lora_model, lora_tokenizer, prompt)
        print(lora_response)


# ── Single prompt mode ────────────────────────────────────────────────────────
def single_prompt_mode(base_model, base_tokenizer, lora_model, lora_tokenizer, instruction, input_text=None):
    """Generate a single response from both models."""
    prompt = build_prompt(instruction, input_text)
    print(f"\nPrompt:\n{prompt}")

    print("\n--- Base Model ---")
    base_response = generate(base_model, base_tokenizer, prompt)
    print(base_response)

    print("\n--- LoRA Model ---")
    lora_response = generate(lora_model, lora_tokenizer, prompt)
    print(lora_response)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA-adapted model")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Base model name")
    parser.add_argument("--adapter", type=str, default=ADAPTER_PATH, help="Path to LoRA adapter")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--eval-train", action="store_true", help="Run evaluation on a subset of train set")
    parser.add_argument("--max-examples", type=int, default=200, help="Max examples for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Single instruction to run")
    parser.add_argument("--input", type=str, default=None, help="Optional input for the instruction")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    quantize = not args.no_quantize

    # Load both models
    base_model, base_tokenizer = load_base_model(args.model, quantize=quantize)
    lora_model, lora_tokenizer = load_lora_model(args.model, args.adapter, quantize=quantize)

    if args.eval or args.eval_train:
        # Load the same splits used in training
        full_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        dataset_splits = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset_splits["train"]
        temp_dataset = dataset_splits["test"].train_test_split(test_size=0.5, seed=42)
        test_dataset = temp_dataset["test"]

        if args.eval_train:
            eval_subset = train_dataset.select(range(min(args.max_examples, len(train_dataset))))
            split_label = "Train"
        else:
            eval_subset = test_dataset.select(range(min(args.max_examples, len(test_dataset))))
            split_label = "Test"

        print(f"\nEvaluating on {split_label} split ({len(eval_subset)} examples)")

        base_results = evaluate_model(
            base_model, base_tokenizer, eval_subset,
            label=f"Base Model ({split_label})",
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
        )
        lora_results = evaluate_model(
            lora_model, lora_tokenizer, eval_subset,
            label=f"LoRA Model ({split_label})",
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
        )

        # Print comparison table
        print("\n" + "=" * 60)
        print(f"COMPARISON SUMMARY ({split_label} split)")
        print("=" * 60)
        print(f"{'Metric':<25} {'Base Model':>15} {'LoRA Model':>15}")
        print("-" * 55)
        for metric in ["exact_match", "normalized_match", "prefix_match"]:
            print(f"{metric:<25} {base_results[metric]:>14.2f}% {lora_results[metric]:>14.2f}%")

    elif args.prompt:
        single_prompt_mode(base_model, base_tokenizer, lora_model, lora_tokenizer, args.prompt, args.input)

    else:
        interactive_mode(base_model, base_tokenizer, lora_model, lora_tokenizer)


if __name__ == "__main__":
    main()
