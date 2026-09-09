# Inference token overhead + amortization (yeRc-W1/W6)

Per-query *added* input tokens (context that must be re-encoded on every query):

| Method | added tokens / query |
|---|---|
| Code2LoRA (merged adapter) | 0 |
| RAG | up to 1536 |
| DRC | up to 4096 |

## Per-query inference latency (H100, Qwen2.5-Coder-1.5B, batch=4, greedy; job 17450724)

All methods run the same frozen backbone; a merged LoRA adds **0** inference cost (equality check: merged-LoRA 197.1 ms vs base 198.3 ms @ in=2048/out=64, Δ=-1.2 ms). Latency is therefore set by total input length; RAG/DRC pay for prepended context on *every* query.

Latency (ms/query) vs input length:

| input tok | out=8 | out=64 |
|---|---|---|
| 512 | 24.9 | 187.3 |
| 1024 | 28.4 | 190.3 |
| 2048 | 36.0 | 198.3 |
| 3072 | 44.5 | 206.6 |
| 4096 | 53.3 | 216.8 |

By method (representative CR prefix ≈2048 tok):

| method | input | out=8 | out=64 |
|---|---|---|---|
| Pretrained / FFT / sLoRA / pLoRA / Code2LoRA-Static / Code2LoRA-Evo | prefix only (≈2048) | 36.0 | 198.3 |
| RAG (k=3) | +≈1.5k (≈3584) | 44.5 (+24%) | 206.6 (+4%) |
| DRC | fills 4096 cap | 53.3 (+48%) | 216.8 (+9%) |

Short (realistic) targets are prefill-bound, so RAG/DRC's per-query penalty is largest there (+24% / +48%); with the 64-token cap decode dominates and the penalty shrinks (+4% / +9%). Code2LoRA matches the pretrained backbone exactly at inference.

## Prefix cost and queries per repo

| Suite | queries | repos | queries/repo | prefix tok (median) | prefix tok (mean) |
|---|---|---|---|---|---|
| cr_test | 476455 | 51 | 9342.3 | 1940 | 2248 |
| ir_test | 535003 | 389 | 1375.3 | 2483 | 3255 |

## One-time adapter generation (paid once per repo-state, amortized over all subsequent completions)

### CPU (Intel Xeon Platinum 8570, 8 threads)

mean full walk: **728.9s**, mean incremental /adapt: **120.2s**.

| repo | full_s | incr_s | cached_s | commits |
|---|---|---|---|---|
| SAML-Toolkits/python-saml | 763.3 | 296.6 | 295.9 | 356 |
| jpsca/jinjax | 344.0 | 30.4 | 22.7 | 188 |
| fastapi/sqlmodel | 1079.3 | 33.7 | 27.0 | 1073 |

### GPU (single NVIDIA H100 80GB; job 17447454)

mean full walk: **111.2s**, mean incremental /adapt: **96.9s** (dominated by python-saml; jinjax/sqlmodel incr ≈ 21–23s). During the walk GPU utilization stayed ≈0% with ≈3.4 GB resident — the cost is per-commit git checkout + AST parsing (CPU-bound), so the GPU mainly removes the CPU matmul cost of the encoder/hypernetwork.

| repo | full_s | incr_s | cached_s | commits |
|---|---|---|---|---|
| SAML-Toolkits/python-saml | 256.8 | 246.7 | 245.1 | 356 |
| jpsca/jinjax | 33.5 | 21.3 | 21.9 | 188 |
| fastapi/sqlmodel | 43.4 | 22.8 | 24.9 | 1073 |

CPU→GPU full-walk speedup: 3.0× (python-saml), 10.3× (jinjax), 24.9× (sqlmodel).
