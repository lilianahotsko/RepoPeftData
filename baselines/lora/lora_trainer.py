import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from typing import Dict, List
from tqdm import tqdm

# Model
model_name = "Qwen/Qwen2.5-Coder-1.5B"

# Quantization (optional, saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dataset - replace with your own
# dataset = load_dataset("json", data_files="your_data.jsonl", split="train")

# # Formatting function - adjust to your data format
# def formatting_func(example):
#     return example["text"]  # or build your prompt template here

# Initialize wandb
wandb.init(
    project="lora-baseline-REPOPEFTDATA",
    name="qwen-coder-lora-20k",
    config={
        "model_name": model_name,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "max_seq_length": 2048,
        "label_masking": True,  # Using custom collator to mask instruction/input
    }
)

# Training
training_args = TrainingArguments(
    output_dir="/home/lhotsko/scratch/baselines/lora/qwen-coder-lora-20k",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="wandb",  # Enable wandb logging
    remove_unused_columns=False,  # Keep all columns for custom data collator
)

from datasets import load_dataset

# Load dataset (only has "train" split)
full_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

# Create train/validation/test splits
# Use 80% train, 10% validation, 10% test
dataset_splits = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits["train"]
temp_dataset = dataset_splits["test"].train_test_split(test_size=0.5, seed=42)
val_dataset = temp_dataset["train"]
test_dataset = temp_dataset["test"]

# Use subset for faster training/testing (use .select() instead of slicing)
# Remove .select() calls to use full dataset
dataset = train_dataset.select(range(min(1000, len(train_dataset))))
val_dataset_subset = val_dataset.select(range(min(100, len(val_dataset))))
test_dataset_subset = test_dataset.select(range(min(100, len(test_dataset))))

def formatting_func(example):
    """
    Formats the example into a single text string.
    The response_start_marker helps identify where to start computing loss.
    """
    instruction_part = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input"):
        instruction_part += f"### Input:\n{example['input']}\n"
    response_part = f"### Response:\n{example['output']}"
    text = instruction_part + response_part
    return text


def exact_match(prediction: str, target: str) -> bool:
    """Compute exact match score (case-sensitive comparison after stripping)."""
    return prediction.strip() == target.strip()


def prefix_match(prediction: str, target: str) -> bool:
    """Check if the prediction starts with the target (model generates correct answer + extras)."""
    return prediction.strip().startswith(target.strip())


def normalize_code(text: str) -> str:
    """Normalize code for comparison: collapse whitespace, strip."""
    import re
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # collapse all whitespace to single space
    return text


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
        if any(lower.startswith(m) for m in cut_markers):
            break
        kept_lines.append(line)

    # 3. Remove trailing empty lines
    while kept_lines and kept_lines[-1].strip() == "":
        kept_lines.pop()

    return "\n".join(kept_lines).strip()


def normalized_match(prediction: str, target: str) -> bool:
    """Check if prediction matches target after normalizing whitespace."""
    return normalize_code(prediction) == normalize_code(target)


def compute_exact_match_accuracy(model, tokenizer, dataset, max_examples=None, max_new_tokens=256, print_mismatches=5):
    """
    Compute exact match accuracy by generating predictions and comparing with ground truth.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: The dataset to evaluate on
        max_examples: Maximum number of examples to evaluate (None for all)
        max_new_tokens: Maximum number of tokens to generate
        print_mismatches: Number of mismatched examples to print for debugging
    
    Returns:
        Dictionary with exact_match_accuracy, normalized_match_accuracy, prefix_match_accuracy
    """
    model.eval()
    
    num_examples = min(max_examples, len(dataset)) if max_examples else len(dataset)
    exact_matches = 0
    normalized_matches = 0
    prefix_matches = 0
    mismatches_printed = 0
    
    print(f"Computing exact match on {num_examples} examples...")
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in tqdm(range(num_examples), desc="Evaluating"):
            example = dataset[i]
            
            # Build the prompt (instruction + input, without response)
            prompt = f"### Instruction:\n{example['instruction']}\n"
            if example.get("input"):
                prompt += f"### Input:\n{example['input']}\n"
            prompt += "### Response (code only, no explanation):\n"
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs['input_ids'].shape[1]
            
            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the generated tokens (after the prompt)
            generated_tokens = outputs[0][prompt_len:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Trim trailing explanations, notes, and commentary
            prediction = trim_response(prediction)
            
            # Get ground truth
            ground_truth = example['output'].strip()
            
            # Check metrics
            is_exact = exact_match(prediction, ground_truth)
            is_normalized = normalized_match(prediction, ground_truth)
            is_prefix = prefix_match(prediction, ground_truth)
            
            if is_exact:
                exact_matches += 1
            if is_normalized:
                normalized_matches += 1
            if is_prefix:
                prefix_matches += 1
            
            # Print a few mismatches for debugging
            if not is_exact and mismatches_printed < print_mismatches:
                mismatches_printed += 1
                print(f"\n--- Mismatch {mismatches_printed} ---")
                print(f"  INSTRUCTION: {example['instruction'][:100]}...")
                print(f"  EXPECTED:    {ground_truth[:200]}")
                print(f"  PREDICTED:   {prediction[:200]}")
                print(f"  norm_match={is_normalized}, prefix_match={is_prefix}")
    
    em_accuracy = (exact_matches / num_examples * 100) if num_examples > 0 else 0.0
    nm_accuracy = (normalized_matches / num_examples * 100) if num_examples > 0 else 0.0
    pm_accuracy = (prefix_matches / num_examples * 100) if num_examples > 0 else 0.0
    
    print(f"\n  Exact Match:      {em_accuracy:.2f}% ({exact_matches}/{num_examples})")
    print(f"  Normalized Match: {nm_accuracy:.2f}% ({normalized_matches}/{num_examples})")
    print(f"  Prefix Match:     {pm_accuracy:.2f}% ({prefix_matches}/{num_examples})")
    
    return {
        "exact_match_accuracy": em_accuracy,
        "exact_matches": exact_matches,
        "normalized_match_accuracy": nm_accuracy,
        "normalized_matches": normalized_matches,
        "prefix_match_accuracy": pm_accuracy,
        "prefix_matches": prefix_matches,
        "total_examples": num_examples,
    }


class InstructionDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator that masks labels for instruction/input parts.
    Only computes loss on the response part (after "### Response:").
    
    This ensures the model only learns to predict the response, not the instruction/input.
    """
    def __init__(self, tokenizer, response_marker="### Response:", **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.response_marker = response_marker
        # Tokenize the marker to find it in tokenized sequences
        # Use the same tokenization settings as SFTTrainer
        self.response_marker_tokens = tokenizer.encode(
            response_marker, add_special_tokens=False
        )
        # Also try with newline if tokenizer adds it
        self.response_marker_with_newline = tokenizer.encode(
            response_marker + "\n", add_special_tokens=False
        )
    
    def __call__(self, examples):
        # SFTTrainer may pass raw dicts or formatted strings
        # Format dicts to strings if needed
        if isinstance(examples[0], dict):
            # Format the examples using the same logic as formatting_func
            formatted_texts = []
            for example in examples:
                instruction_part = f"### Instruction:\n{example['instruction']}\n"
                if example.get("input"):
                    instruction_part += f"### Input:\n{example['input']}\n"
                response_part = f"### Response:\n{example['output']}"
                text = instruction_part + response_part
                formatted_texts.append(text)
            examples = formatted_texts
        
        # Now examples should be strings - tokenize them
        if isinstance(examples[0], str):
            # Tokenize the text strings
            batch = self.tokenizer(
                examples,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            # Labels = copy of input_ids (the model shifts internally)
            labels = batch["input_ids"].clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        else:
            # Already tokenized, use parent class
            batch = super().__call__(examples)
        
        # Now mask labels for instruction/input parts
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        
        for i in range(len(labels)):
            token_ids = input_ids[i].tolist()
            # Find where response starts
            response_start_idx = self._find_response_start(token_ids)
            
            if response_start_idx > 0:
                # Mask everything before response (set to -100)
                labels[i, :response_start_idx] = -100
        
        return batch
    
    def _find_response_start(self, token_ids: List[int]) -> int:
        """Find the start position of the response marker in token_ids."""
        # Try both marker variants
        for marker_tokens in [self.response_marker_tokens, self.response_marker_with_newline]:
            if len(marker_tokens) == 0:
                continue
            
            # Search for the marker tokens in the sequence
            for i in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[i:i+len(marker_tokens)] == marker_tokens:
                    # Return position after the marker (where response content starts)
                    return i + len(marker_tokens)
        
        # If marker not found, return 0 (compute loss on everything as fallback)
        # This shouldn't happen if formatting is correct
        return 0



# Create custom data collator that masks instruction/input labels
data_collator = InstructionDataCollator(
    tokenizer=tokenizer,
    response_marker="### Response:",
    mlm=False,  # We're doing causal LM, not masked LM
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=val_dataset_subset,
    formatting_func=formatting_func,
    data_collator=data_collator,  # Use custom collator for label masking
    args=training_args,
)

# Eval before training
print("\nInitial eval...")
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
print(f"Initial eval loss: {eval_loss}")

# Compute exact match before training
em_results = compute_exact_match_accuracy(
    model=model,
    tokenizer=tokenizer,
    dataset=val_dataset_subset,
    max_examples=len(val_dataset_subset),
    max_new_tokens=256,
)
wandb.log({
    "init_eval_loss": eval_loss,
    "init_exact_match_accuracy": em_results['exact_match_accuracy'],
    "init_normalized_match_accuracy": em_results['normalized_match_accuracy'],
    "init_prefix_match_accuracy": em_results['prefix_match_accuracy'],
})

# Training
print("\nTraining...")
trainer.train()

# Eval after training
print("\nFinal eval...")
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
print(f"Final eval loss: {eval_loss}")

# Compute exact match after training
em_results = compute_exact_match_accuracy(
    model=model,
    tokenizer=tokenizer,
    dataset=val_dataset_subset,
    max_examples=len(val_dataset_subset),
    max_new_tokens=256,
)
wandb.log({
    "final_eval_loss": eval_loss,
    "final_exact_match_accuracy": em_results['exact_match_accuracy'],
    "final_normalized_match_accuracy": em_results['normalized_match_accuracy'],
    "final_prefix_match_accuracy": em_results['prefix_match_accuracy'],
})

# Test on test set
print("\nTest eval...")
test_results = trainer.evaluate(test_dataset_subset)
test_loss = test_results["eval_loss"]
print(f"Test loss: {test_loss}")

# Compute exact match on test set
em_results = compute_exact_match_accuracy(
    model=model,
    tokenizer=tokenizer,
    dataset=test_dataset_subset,
    max_examples=len(test_dataset_subset),
    max_new_tokens=256,
)
wandb.log({
    "test_loss": test_loss,
    "test_exact_match_accuracy": em_results['exact_match_accuracy'],
    "test_normalized_match_accuracy": em_results['normalized_match_accuracy'],
    "test_prefix_match_accuracy": em_results['prefix_match_accuracy'],
})

# Save model
trainer.model.save_pretrained("./qwen-coder-lora-adapter")
tokenizer.save_pretrained("./qwen-coder-lora-adapter")

# Finish wandb run
wandb.finish()