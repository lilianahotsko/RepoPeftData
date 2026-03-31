# Training Memory Analysis — Qwen2.5-Coder-1.5B

This document provides a detailed breakdown of GPU memory usage during FFT
(full fine-tuning) and LoRA training, including the evaluation stage that
causes OOM on 80 GB H100 GPUs at long context lengths.

---

## 1. Model Architecture

| Parameter | Value |
|---|---|
| `vocab_size` (V) | 151,936 |
| `hidden_size` (H) | 1,536 |
| `num_hidden_layers` (L) | 28 |
| `num_attention_heads` (A) | 12 |
| `num_key_value_heads` (KV) | 2 |
| `intermediate_size` (I) | 8,960 |
| `head_dim` (HD) | 128 |
| `kv_dim` (KVD) | 256 |
| `tie_word_embeddings` | True (lm_head shares embedding weights) |

### 1.1 Parameter Counts

| Component | Parameters | Notes |
|---|---:|---|
| Embedding (`embed_tokens`) | 233,373,696 | V × H, shared with `lm_head` |
| **Per transformer layer** | **46,797,824** | × 28 layers |
| &ensp; `q_proj` (weight + bias) | 2,360,832 | H × H + H |
| &ensp; `k_proj` (weight + bias) | 393,472 | KVD × H + KVD |
| &ensp; `v_proj` (weight + bias) | 393,472 | KVD × H + KVD |
| &ensp; `o_proj` (weight) | 2,359,296 | H × H |
| &ensp; `gate_proj` (weight) | 13,762,560 | I × H |
| &ensp; `up_proj` (weight) | 13,762,560 | I × H |
| &ensp; `down_proj` (weight) | 13,762,560 | H × I |
| &ensp; Layernorms (×2) | 3,072 | 2 × H |
| All 28 layers | 1,310,339,072 | |
| Final layernorm | 1,536 | H |
| **Total** | **1,543,714,304** | ~1.54B |

### 1.2 LoRA Parameters (rank=16, 7 target modules)

| Module | lora_A | lora_B | Per layer | × 28 |
|---|---:|---:|---:|---:|
| `q_proj` | H×R = 24,576 | R×H = 24,576 | 49,152 | 1,376,256 |
| `k_proj` | H×R = 24,576 | R×KVD = 4,096 | 28,672 | 802,816 |
| `v_proj` | H×R = 24,576 | R×KVD = 4,096 | 28,672 | 802,816 |
| `o_proj` | H×R = 24,576 | R×H = 24,576 | 49,152 | 1,376,256 |
| `gate_proj` | H×R = 24,576 | R×I = 143,360 | 167,936 | 4,702,208 |
| `up_proj` | H×R = 24,576 | R×I = 143,360 | 167,936 | 4,702,208 |
| `down_proj` | I×R = 143,360 | R×H = 24,576 | 167,936 | 4,702,208 |
| **Total** | | | **659,456** | **18,464,768** |

LoRA trainable = 18.46M parameters (1.20% of base).

---

## 2. Memory Components

GPU memory during training is consumed by four categories:

### 2.1 Static Memory (persists throughout training)

**FFT (full fine-tuning):**

| Component | Formula | Size |
|---|---|---:|
| Model weights (bf16) | P × 2 | 2.88 GiB |
| Gradients (bf16) | P × 2 | 2.88 GiB |
| AdamW — fp32 master copy | P × 4 | 5.75 GiB |
| AdamW — momentum (fp32) | P × 4 | 5.75 GiB |
| AdamW — variance (fp32) | P × 4 | 5.75 GiB |
| **FFT static total** | **P × 16** | **23.00 GiB** |

**LoRA (rank=16):**

| Component | Formula | Size |
|---|---|---:|
| Frozen base weights (bf16) | P × 2 | 2.88 GiB |
| LoRA adapter weights (bf16) | T × 2 | 0.03 GiB |
| LoRA gradients (bf16) | T × 2 | 0.03 GiB |
| AdamW — fp32 master copy | T × 4 | 0.07 GiB |
| AdamW — momentum (fp32) | T × 4 | 0.07 GiB |
| AdamW — variance (fp32) | T × 4 | 0.07 GiB |
| **LoRA static total** | **P × 2 + T × 14** | **3.15 GiB** |

> P = total parameters (1.54B), T = trainable parameters (18.46M for LoRA r=16).
> AdamW stores three fp32 tensors per trainable parameter. For FFT, all
> parameters are trainable, so the optimizer alone takes 17.25 GiB. For LoRA,
> only the adapter parameters are optimized, so the optimizer is negligible.

### 2.2 Activation Memory (during forward/backward)

With **gradient checkpointing** enabled, the forward pass does not store
intermediate activations for every layer. Instead it stores only the input
tensor at each layer boundary (the "checkpoint") and recomputes the rest
during the backward pass one layer at a time.

With **Flash Attention 2**, the attention computation is memory-efficient:
O(S) instead of O(S²), so no materialized attention score matrix.

**Checkpoint storage** (persists during backward):

```
L × B × S × H × 2 bytes
= 28 × B × S × 1536 × 2
```

**Peak single-layer recompute** (temporary, one layer at a time):

The MLP intermediates dominate (I = 8960 >> H = 1536):

```
B × S × (3 × I + 2 × H) × 2 bytes
= B × S × (3 × 8960 + 2 × 1536) × 2
= B × S × 29,952 × 2
```

The three MLP tensors are: `gate_proj` output, `up_proj` output, and their
element-wise product after SiLU activation — all shape (B, S, 8960).

**Total activation peak** = checkpoints + one layer recompute:

| seq_len (S) | Checkpoints | Layer recompute | **Total** |
|---:|---:|---:|---:|
| 512 | 0.04 GiB | 0.03 GiB | **0.07 GiB** |
| 1,024 | 0.08 GiB | 0.06 GiB | **0.14 GiB** |
| 2,048 | 0.16 GiB | 0.11 GiB | **0.28 GiB** |
| 4,096 | 0.33 GiB | 0.23 GiB | **0.56 GiB** |
| 8,192 | 0.66 GiB | 0.46 GiB | **1.11 GiB** |

> Activations scale linearly with sequence length (thanks to Flash Attention
> and gradient checkpointing). Even at 8K, activations are only ~1 GiB.

### 2.3 Eval Logits — The OOM Source

During evaluation, the Trainer calls `model.forward()` which returns logits
of shape `(B_eval, S, V)`, then computes cross-entropy loss. This is where
the memory explodes.

**The two-step allocation inside the loss function:**

1. **Model output logits** (bf16): `B_eval × S × V × 2` bytes
2. **cross_entropy upcast** (fp32): `B_eval × S × V × 4` bytes — PyTorch's
   `F.cross_entropy` casts inputs to fp32 internally for numerical stability
3. **Softmax scratch** (fp32): another `B_eval × S × V × 4` bytes for the
   log-softmax intermediate

**Combined eval logits memory:**

```
B_eval × S × V × 8  bytes    (logits fp32 + softmax scratch)
```

> **CRITICAL**: HuggingFace `TrainingArguments` defaults
> `per_device_eval_batch_size = 8`, not 1. This is independent of the
> training batch size. Most users set `per_device_train_batch_size=1` but
> forget that eval uses 8× larger batches.

| B_eval | S = 2,048 | S = 4,096 | S = 8,192 |
|---:|---:|---:|---:|
| 1 | 2.32 GiB | 4.64 GiB | 9.27 GiB |
| 2 | 4.64 GiB | 9.27 GiB | 18.55 GiB |
| 4 | 9.27 GiB | 18.55 GiB | 37.09 GiB |
| **8 (default)** | **18.55 GiB** | **37.09 GiB** | **74.18 GiB** |

At S=8192 with the default B_eval=8, the eval logits alone need 74.18 GiB —
nearly the entire 80 GB GPU. Combined with static memory, this guarantees OOM.

**Additional hazard — `eval_accumulation_steps`:**

When `eval_accumulation_steps=None` (default), the Trainer accumulates *all*
prediction logits on GPU across the entire eval dataset before computing
metrics. For a val set of N examples, this stores `N × S × V × 4` bytes on
GPU. With N=2000 and S=8192, that would be **9,273 GiB** — obviously
impossible. In practice the OOM occurs on the first few batches when the
single-batch logits alone exceed available memory.

---

## 3. Total Peak Memory

### 3.1 Training Peak (no eval)

```
M_train = M_static + M_activations
```

| Config | Static | Activations | **Peak** |
|---|---:|---:|---:|
| FFT, S=2048 | 23.00 | 0.28 | **23.28 GiB** |
| FFT, S=4096 | 23.00 | 0.56 | **23.56 GiB** |
| FFT, S=8192 | 23.00 | 1.11 | **24.12 GiB** |
| LoRA r=16, S=2048 | 3.15 | 0.28 | **3.43 GiB** |
| LoRA r=16, S=4096 | 3.15 | 0.56 | **3.71 GiB** |
| LoRA r=16, S=8192 | 3.15 | 1.11 | **4.26 GiB** |

> Training alone fits comfortably on an 80 GB H100 for any sequence length.
> FFT uses ~24 GiB (30% of GPU), LoRA uses ~4 GiB (5%).

### 3.2 Eval Peak

```
M_eval = M_static + B_eval × S × V × 8
```

| Config | Static | Eval logits (B_e=8) | **Peak** | Fits 80 GB? |
|---|---:|---:|---:|:---:|
| FFT, S=2048 | 23.00 | 18.55 | **41.55 GiB** | Yes |
| FFT, S=4096 | 23.00 | 37.09 | **60.10 GiB** | Yes |
| FFT, S=8192 | 23.00 | 74.18 | **97.19 GiB** | **NO** |
| LoRA r=16, S=2048 | 3.15 | 18.55 | **21.70 GiB** | Yes |
| LoRA r=16, S=4096 | 3.15 | 37.09 | **40.24 GiB** | Yes |
| LoRA r=16, S=8192 | 3.15 | 74.18 | **77.34 GiB** | Borderline |

With `per_device_eval_batch_size=1`:

| Config | Static | Eval logits (B_e=1) | **Peak** | Fits 80 GB? |
|---|---:|---:|---:|:---:|
| FFT, S=8192 | 23.00 | 9.27 | **32.28 GiB** | Yes |
| LoRA r=16, S=8192 | 3.15 | 9.27 | **12.42 GiB** | Yes |

---

## 4. General Formulas

### Notation

| Symbol | Meaning | This model |
|---|---|---|
| P | Total parameters | 1,543,714,304 |
| T | Trainable parameters | P (FFT) or 18,464,768 (LoRA r=16) |
| B | Training batch size | 1 |
| B_e | Eval batch size | 8 (default!) |
| S | Sequence length | 2048–8192 |
| V | Vocab size | 151,936 |
| H | Hidden size | 1,536 |
| I | Intermediate (MLP) size | 8,960 |
| L | Number of layers | 28 |
| R | LoRA rank | 16 |

### 4.1 Static Memory

```
FFT:    M_static = P × 16 bytes
LoRA:   M_static = P × 2 + T × 14 bytes
```

Breakdown: weights (×2 bf16) + gradients (×2 bf16) + AdamW master/momentum/variance (×4+4+4 fp32).
For LoRA, only T parameters have gradients and optimizer states.

### 4.2 Activation Memory (gradient checkpointing + Flash Attention 2)

```
M_act = B × S × (L × H × 2  +  (3I + 2H) × 2)  bytes
      = B × S × (L × H + 3I + 2H) × 2            bytes
```

Plugging in this model's values:

```
M_act = B × S × (28 × 1536 + 3 × 8960 + 2 × 1536) × 2
      = B × S × 72,888 × 2
      = B × S × 145,776  bytes
      ≈ B × S × 0.139 MiB
```

### 4.3 Eval Logits Memory

```
M_eval_logits = B_e × S × V × 8 bytes
```

The factor 8 = 4 (fp32 logits) + 4 (softmax scratch in cross_entropy).

Plugging in V = 151,936:

```
M_eval_logits = B_e × S × 1,215,488  bytes
              ≈ B_e × S × 1.159 MiB
```

### 4.4 Total Peak

```
Training (forward+backward):
    M_peak_train = M_static + M_act

Eval (forward + loss):
    M_peak_eval = M_static + M_eval_logits

Overall peak = max(M_peak_train, M_peak_eval)
```

### 4.5 Quick Calculator

To check if a configuration fits in GPU_MEM GiB:

```python
def fits_in_memory(
    gpu_gib: float,           # e.g. 80
    method: str,              # "fft" or "lora"
    seq_len: int,             # e.g. 8192
    eval_batch: int = 1,      # set per_device_eval_batch_size
    train_batch: int = 1,     # per_device_train_batch_size
    lora_rank: int = 16,
    # Qwen2.5-Coder-1.5B constants
    P: int = 1_543_714_304,
    V: int = 151_936,
    H: int = 1536,
    I: int = 8960,
    L: int = 28,
):
    if method == "fft":
        T = P
    else:
        # LoRA trainable params (7 target modules)
        KVD = 256
        T = L * (
            H * lora_rank + lora_rank * H      # q
            + H * lora_rank + lora_rank * KVD   # k
            + H * lora_rank + lora_rank * KVD   # v
            + H * lora_rank + lora_rank * H     # o
            + H * lora_rank + lora_rank * I     # gate
            + H * lora_rank + lora_rank * I     # up
            + I * lora_rank + lora_rank * H     # down
        )

    static = P * 2 + T * 14  # works for both FFT (T=P) and LoRA
    act = train_batch * seq_len * (L * H + 3 * I + 2 * H) * 2
    eval_logits = eval_batch * seq_len * V * 8
    overhead = 1.5 * 2**30  # ~1.5 GiB framework overhead

    train_peak = static + act + overhead
    eval_peak = static + eval_logits + overhead

    peak = max(train_peak, eval_peak)
    peak_gib = peak / 2**30

    return peak_gib <= gpu_gib, peak_gib
```

---

## 5. Why Eval Causes OOM at 8K — Root Cause

The training loop (`trainer.train()`) never materializes the full
`(B, S, V)` logits tensor. The causal LM loss is computed inside
`model.forward()` using a fused kernel that processes the loss
incrementally per position. Peak memory during training is dominated by
the static memory (weights + optimizer) plus modest activation memory.

The **evaluation loop** (`trainer.evaluate()`) is different:

1. `model.forward(**inputs)` returns the full logits tensor `(B_e, S, V)`
   in bf16.
2. The loss function `ForCausalLMLoss` shifts logits by 1, calls
   `F.cross_entropy` which upcasts to fp32 and computes log-softmax,
   allocating scratch buffers.
3. By default, `per_device_eval_batch_size=8` — 8× the training batch.

At S=8192 and B_e=8:
- Step 1: logits bf16 = 8 × 8192 × 151,936 × 2 = **18.55 GiB**
- Step 2: fp32 for loss = 8 × 8191 × 151,936 × 4 ≈ **37 GiB**
- Combined with model weights + optimizer: exceeds 80 GB

### 5.1 Fixes (in order of preference)

| Fix | How | Trade-off |
|---|---|---|
| `--no-eval` | Skip eval entirely | No val loss monitoring — use WandB train loss only |
| `per_device_eval_batch_size=1` | Reduce eval batch | Slower eval, but fits. Logits: 4.64 GiB at 8K |
| `eval_accumulation_steps=1` | Move logits to CPU after each step | Slight CPU overhead, prevents accumulation OOM |
| Both batch=1 + accum=1 | Belt and suspenders | Safest option when eval is needed |

For HPO with Optuna (needs eval_loss), set both:

```python
TrainingArguments(
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    ...
)
```

This reduces eval peak from **97 GiB** (impossible) to **32 GiB** (comfortable).

---

## 6. Scaling to Other Models

The formulas generalize to any causal LM. The key insight is that eval
memory scales with `V × S × B_e`, while training memory scales with `P`
(static) and `S × H` (activations). Models with large vocabularies (Qwen:
152K, Llama: 128K) are particularly vulnerable to eval OOM.

**Rule of thumb for eval feasibility:**

```
B_e × S × V × 8 bytes  <  GPU_MEM - M_static

Rearranging for max eval batch:
B_e_max = floor((GPU_MEM - M_static) / (S × V × 8))
```

For FFT on 80 GB H100 at S=8192:

```
B_e_max = floor((80 GiB - 23 GiB) / (8192 × 151936 × 8 bytes))
        = floor(57 GiB / 9.27 GiB)
        = 6
```

For LoRA on 80 GB H100 at S=8192:

```
B_e_max = floor((80 GiB - 3.15 GiB) / 9.27 GiB)
        = 8  (just barely)
```

---

## 7. Memory Map Diagram

```
 80 GiB H100 GPU
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  ┌─── FFT Training (S=8192) ── 24.12 GiB ───┐              │
 │  │ Weights bf16         2.88 GiB              │              │
 │  │ Gradients bf16       2.88 GiB              │              │
 │  │ AdamW fp32 master    5.75 GiB              │              │
 │  │ AdamW momentum       5.75 GiB              │              │
 │  │ AdamW variance       5.75 GiB              │              │
 │  │ Activations          1.11 GiB              │              │
 │  └─────────────────────────────────────────────┘              │
 │                                                              │
 │  Remaining: ~56 GiB free                                     │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  ┌─── FFT Eval (S=8192, B_e=8) ── 97.19 GiB ──── OOM! ─┐  │
 │  │ Weights bf16         2.88 GiB                          │  │
 │  │ Gradients bf16       2.88 GiB                          │  │
 │  │ AdamW fp32 master    5.75 GiB                          │  │
 │  │ AdamW momentum       5.75 GiB                          │  │
 │  │ AdamW variance       5.75 GiB                          │  │
 │  │ Logits bf16         18.55 GiB  ← model output         │  │
 │  │ Logits fp32         37.09 GiB  ← cross_entropy        │▓▓│
 │  │ Softmax scratch     37.09 GiB  ← F.cross_entropy      │▓▓│
 │  └───────────────────────────────────  EXCEEDS 80 GiB ───┘  │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘

 Fix: per_device_eval_batch_size=1

 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  ┌─── FFT Eval (S=8192, B_e=1) ── 32.28 GiB ───┐          │
 │  │ Static (weights+opt+grad)  23.00 GiB          │          │
 │  │ Logits fp32 + scratch       9.28 GiB          │          │
 │  └───────────────────────────────────────────────┘          │
 │                                                              │
 │  Remaining: ~48 GiB free                                     │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
```

---

## 8. Practical Configuration Reference

### 8.1 Recommended TrainingArguments for 80 GB H100

**Training only (no eval monitoring):**

```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # effective batch = 32
    gradient_checkpointing=True,
    bf16=True,
    eval_strategy="no",              # skip eval entirely
    max_seq_length=8192,
)
```

Peak memory: ~24 GiB (FFT) or ~4 GiB (LoRA).

**Training with eval monitoring (for HPO):**

```python
TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,    # CRITICAL: override default of 8
    eval_accumulation_steps=1,       # CRITICAL: don't accumulate on GPU
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    bf16=True,
    eval_strategy="epoch",
    max_seq_length=8192,
)
```

Peak memory: ~32 GiB (FFT) or ~12 GiB (LoRA).

### 8.2 Model loading

Always use Flash Attention 2 and explicit device placement:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": "cuda:0"},
)
```

Using `device_map="auto"` with a single GPU can cause unnecessary CPU
offloading overhead. Explicit placement is cleaner.
