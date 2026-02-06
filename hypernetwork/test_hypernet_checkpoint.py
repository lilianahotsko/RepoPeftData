import os
import sys
import argparse
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

# Ensure we import from the same directory as this script (RepoPeftData/)
# This prevents importing other hypernetwork.py files from elsewhere
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import from the local hypernetwork module
from hypernetwork import (
    Hypernetwork,
    LoRA,
    get_module_specs,
    replace_with_lora,
    inject_lora_weights,
    HypernetDataCollator,
    set_seed
)


def load_checkpoint(checkpoint_path):
    """Load hypernetwork checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    hypernet_state_dict = checkpoint["hypernet_state_dict"]
    module_specs = checkpoint["module_specs"]
    hypernet_config = checkpoint["hypernet_config"]
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Input dim: {hypernet_config['input_dim']}")
    print(f"  Hidden dim: {hypernet_config['hidden_dim']}")
    print(f"  Rank: {hypernet_config['rank']}")
    print(f"  Module types: {hypernet_config['types']}")
    print(f"  Number of module specs: {len(module_specs)}")
    
    return hypernet_state_dict, module_specs, hypernet_config


def setup_model_and_hypernet(model_name, hypernet_config, module_specs, hypernet_state_dict, rank=16, alpha=32):
    """Set up the base model and hypernetwork."""
    print(f"\nLoading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )
    
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    print("✓ Base model loaded")
    
    # Replace modules with LoRA wrappers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    replace_with_lora(model, module_specs, r=rank, alpha=alpha)
    print(f"✓ Replaced {len(module_specs)} modules with LoRA wrappers")
    
    # Reconstruct hypernetwork
    print("\nReconstructing hypernetwork...")
    hypernet = Hypernetwork(
        input_dim=hypernet_config["input_dim"],
        module_specs=module_specs,
        hidden_dim=hypernet_config["hidden_dim"],
        rank=hypernet_config["rank"],
        num_layers=max((li for _, li, *_ in module_specs if li >= 0), default=-1) + 1 or 32
    ).cuda()
    
    # Load checkpoint state dict
    hypernet.load_state_dict(hypernet_state_dict)
    hypernet.eval()
    
    print("✓ Hypernetwork reconstructed and loaded")
    print(f"  Hypernetwork parameters: {sum(p.numel() for p in hypernet.parameters()):,}")
    
    return model, hypernet, tokenizer


def extract_ground_truth_from_labels(labels, tokenizer):
    """Extract ground truth text from labels (removing -100 tokens)."""
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()
    
    # Remove -100 tokens (ignored tokens)
    valid_labels = [token for token in labels if token != -100]
    
    if not valid_labels:
        return ""
    
    # Decode the tokens to get the ground truth text
    ground_truth = tokenizer.decode(valid_labels, skip_special_tokens=True)
    return ground_truth.strip()


def exact_match(prediction: str, target: str) -> bool:
    """Compute exact match score (case-sensitive comparison after stripping)."""
    return prediction.strip() == target.strip()


def evaluate_on_validation_set(model, hypernet, tokenizer, val_ds, module_specs, max_examples=None, 
                                max_new_tokens=128, temperature=0.0, compute_exact_match=True):
    """Evaluate the hypernetwork on the validation set."""
    print(f"\nEvaluating on validation set...")
    print(f"  Total examples: {len(val_ds)}")
    if max_examples:
        print(f"  Evaluating on first {max_examples} examples")
    
    model.eval()
    hypernet.eval()
    
    collator = HypernetDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=8192
    )
    
    total_loss = 0.0
    total_tokens = 0
    num_examples = 0
    
    # For exact match evaluation
    exact_matches = 0
    predictions_list = []
    ground_truths_list = []
    
    eval_examples = val_ds[:max_examples] if max_examples else val_ds
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(eval_examples, desc="Evaluating")):
            # Prepare batch using collator
            batch = collator([example])
            
            ctx = batch["ctx"].to(model.device).float()  # [1, dim]
            
            # Generate LoRA weights from hypernetwork
            hyper_out = hypernet(ctx)
            
            # Inject LoRA weights into model
            inject_lora_weights(model, module_specs, hyper_out, batch_index=0)
            
            # Forward pass for loss calculation
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Count valid tokens (non-ignore labels)
            labels = batch["labels"].to(model.device)
            valid_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            
            # Extract ground truth from labels
            ground_truth = extract_ground_truth_from_labels(labels, tokenizer)
            ground_truths_list.append(ground_truth)
            
            # Generate prediction for exact match if requested
            if compute_exact_match:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                
                # Find where the prompt ends (first position where labels are not -100)
                labels_cpu = labels.cpu().tolist()[0] if len(labels.shape) > 1 else labels.cpu().tolist()
                prompt_end_idx = None
                for i, label in enumerate(labels_cpu):
                    if label != -100:
                        prompt_end_idx = i
                        break
                
                # If we found where target starts, use only prompt for generation
                # Otherwise, generate from the full input (fallback)
                if prompt_end_idx is not None:
                    prompt_input_ids = input_ids[:, :prompt_end_idx]
                    prompt_attention_mask = attention_mask[:, :prompt_end_idx]
                else:
                    # If all labels are -100 (shouldn't happen), use full input
                    prompt_input_ids = input_ids
                    prompt_attention_mask = attention_mask
                
                # Generate continuation from prompt
                generated = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Extract only the generated part (after prompt)
                prompt_len = prompt_input_ids.shape[1]
                generated_tokens = generated[0, prompt_len:]
                prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                predictions_list.append(prediction)
                
                # Check exact match
                if exact_match(prediction, ground_truth):
                    exact_matches += 1
            
            num_examples += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    # Calculate exact match accuracy
    em_accuracy = (exact_matches / num_examples * 100) if num_examples > 0 and compute_exact_match else None
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Examples evaluated: {num_examples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    if compute_exact_match:
        print(f"Exact Match Accuracy: {em_accuracy:.2f}% ({exact_matches}/{num_examples})")
    print(f"{'='*70}")
    
    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_examples": num_examples,
        "total_tokens": total_tokens
    }
    
    if compute_exact_match:
        results["exact_match_accuracy"] = em_accuracy
        results["exact_matches"] = exact_matches
        results["predictions"] = predictions_list
        results["ground_truths"] = ground_truths_list
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test hypernetwork checkpoint on validation set")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="RepoPeftData/model_checkpoints/hypernet_gte_data/checkpoint-8000",
        help="Path to hypernetwork checkpoint file"
    )
    parser.add_argument(
        "--val_ds_path",
        type=str,
        default="RepoPeftData/gte/arrow/val_ds",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="Base model name"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate for exact match evaluation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--no_exact_match",
        action="store_true",
        help="Skip exact match evaluation (only compute loss/perplexity)"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load checkpoint
    hypernet_state_dict, module_specs, hypernet_config = load_checkpoint(args.checkpoint_path)
    
    # Set up model and hypernetwork
    model, hypernet, tokenizer = setup_model_and_hypernet(
        args.model_name,
        hypernet_config,
        module_specs,
        hypernet_state_dict,
        rank=hypernet_config["rank"],
        alpha=32  # Default alpha from training script
    )
    
    # Load validation dataset
    print(f"\nLoading validation dataset from {args.val_ds_path}...")
    val_ds = load_from_disk(args.val_ds_path)
    print(f"✓ Validation dataset loaded: {len(val_ds)} examples")
    
    # Verify dataset structure
    sample = val_ds[0]
    required_fields = ["repo_embedding", "tokens", "labels"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing required field in dataset: {field}")
    print(f"✓ Dataset structure verified")
    
    # Evaluate
    results = evaluate_on_validation_set(
        model,
        hypernet,
        tokenizer,
        val_ds,
        module_specs,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        compute_exact_match=not args.no_exact_match
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

