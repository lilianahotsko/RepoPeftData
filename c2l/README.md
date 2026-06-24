# Code2LoRA (`c2l`)

Turn any git repository into a **portable LoRA adapter** for a frozen code LLM
(`Qwen/Qwen2.5-Coder-1.5B`) — with **no per-user training**. A trained
hypernetwork (a commit-streaming GRU + a multi-task head) reads the repo's
commit history and *generates* the adapter in a single forward pass, so it runs
on CPU and in air-gapped environments.

## Why this scales to end users

- **Generation is cheap.** It only needs the 0.6B encoder + a tiny GRU/head —
  not the multi-GB base model. So an adapter can be produced server-side in
  seconds, or locally on CPU.
- **Adapters are standard.** `c2l.export` writes a normal PEFT
  `adapter_model.safetensors` + `adapter_config.json`. The few-MB artifact runs
  with `transformers`/PEFT, vLLM/TGI, or (after GGUF conversion) `llama.cpp`.
- **No GPU needed to use it.** Apply the adapter to a 4-bit (bitsandbytes) or
  GGUF-quantized base (~1GB) and run on CPU.

## Delivery modes (one codebase)

| Mode | Generation | Inference | For |
|------|-----------|-----------|-----|
| Fully local | local (GPU or CPU) | local (GPU / 4-bit / GGUF) | secure / air-gapped |
| Hybrid | hosted API (GPU) | local quantized | low-resource clients |
| Hosted SaaS | hosted API (GPU) | hosted endpoint | zero-setup users |

## Install

```bash
pip install -e .            # core
pip install -e .[quant]     # + bitsandbytes (4-/8-bit no-GPU inference)
pip install -e .[app]       # + local Gradio app
pip install -e .[api]       # + FastAPI service
```

## CLI

```bash
# Generate a portable adapter (no base model loaded)
c2l adapt https://github.com/psf/cachecontrol --task assert_rhs -o ./adapter

# Run it: full precision, 4-bit, or GGUF/CPU
c2l run --adapter ./adapter --backend hf   --prefix "assert add(2, 2) == "
c2l run --adapter ./adapter --backend 4bit --prefix "assert add(2, 2) == "

# Convert to a GGUF LoRA for llama.cpp
c2l export --adapter ./adapter --gguf ./adapter/adapter.gguf

# List tasks / show config / check export fidelity
c2l tasks
c2l config
c2l verify https://github.com/psf/cachecontrol --with-4bit
```

Set `C2L_OFFLINE=1` (with models + checkpoint pre-cached) for a fully offline
session. All knobs live in [`c2l.yaml`](c2l.yaml) and can be overridden by
`C2L_*` env vars.

## Hosted service

```bash
uvicorn c2l.api.app:app --host 0.0.0.0 --port 8000
# POST /adapters {repo, task} -> job_id ; GET /adapters/{job_id} -> fingerprint
# GET /adapters/{fp}/download ; POST /predict {prefix, fingerprint, backend}
```

Adapters are content-addressed (`AdapterRegistry`), so the same
repo + commit + task is generated at most once.

## Tasks (pluggable)

A task = `extract_from_source` + `format` + `metric` + a stable `task_index`.
Adapter generation itself is task-agnostic; the task selects the head's task
embedding and the QnAs used for training/eval.

- `assert_rhs` — complete a test assertion's right-hand side (the original task).
- `code_gen` — generate a function/method body from its signature + docstring.

Register more in `c2l/tasks/` and they appear everywhere (CLI, API, trainer,
evaluator).

## Training & release

The research trainer (`hypernetwork/train_code2lora_gru_v2.py`) gained a
`--tasks` flag: more than one task enables the task-conditioned head. Gate and
publish a checkpoint with:

```bash
python -m c2l.publish --checkpoint ckpt.pt --commits-dir ... --qnas-dir ... \
    --suites cr_test ir_test --tasks assert_rhs code_gen \
    --output-dir gate --min-exact-match 0.30 \
    --repo code2lora/code2lora-gru --revision v2-multitask --publish
```
