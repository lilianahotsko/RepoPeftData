# Per-query inference compute budget (yeRc-W2 / Suggestion 2)

Backbone: **Qwen2.5-Coder-1.5B** (shared by every method). N_nonembed ≈ **1.31B**, N_total ≈ 1.54B (28L, hidden 1536, inter 8960, vocab 151936; source: config).

Per-query cost is set by INPUT tokens: the merged adapter adds **0**, RAG appends up to **1536** retrieved tokens, and DRC fills the input up to a **4096**-token cap — on every query.


## cr_test  (median prefix ≈ 1940 tok; 476455 queries / 51 repos; 9342 queries/repo)


### Generation = 8 output tokens

| Method | input tok/query | ≈FLOPs/query | ×vs Code2LoRA |
|---|---:|---:|---:|
| Pretrained | 1940 | 5.43e+12 | 1.00× |
| FFT | 1940 | 5.43e+12 | 1.00× |
| sLoRA | 1940 | 5.43e+12 | 1.00× |
| per-repo LoRA | 1940 | 5.43e+12 | 1.00× |
| Code2LoRA-Static | 1940 | 5.43e+12 | 1.00× |
| Code2LoRA-Evo | 1940 | 5.43e+12 | 1.00× |
| RAG (k=3) | 3476 | 1.02e+13 | 1.87× |
| DRC | 4096 | 1.22e+13 | 2.25× |

### Generation = 64 output tokens

| Method | input tok/query | ≈FLOPs/query | ×vs Code2LoRA |
|---|---:|---:|---:|
| Pretrained | 1940 | 5.63e+12 | 1.00× |
| FFT | 1940 | 5.63e+12 | 1.00× |
| sLoRA | 1940 | 5.63e+12 | 1.00× |
| per-repo LoRA | 1940 | 5.63e+12 | 1.00× |
| Code2LoRA-Static | 1940 | 5.63e+12 | 1.00× |
| Code2LoRA-Evo | 1940 | 5.63e+12 | 1.00× |
| RAG (k=3) | 3476 | 1.04e+13 | 1.85× |
| DRC | 4096 | 1.24e+13 | 2.21× |

## ir_test  (median prefix ≈ 2483 tok; 535003 queries / 389 repos; 1375 queries/repo)


### Generation = 8 output tokens

| Method | input tok/query | ≈FLOPs/query | ×vs Code2LoRA |
|---|---:|---:|---:|
| Pretrained | 2483 | 7.06e+12 | 1.00× |
| FFT | 2483 | 7.06e+12 | 1.00× |
| sLoRA | 2483 | 7.06e+12 | 1.00× |
| per-repo LoRA | 2483 | 7.06e+12 | 1.00× |
| Code2LoRA-Static | 2483 | 7.06e+12 | 1.00× |
| Code2LoRA-Evo | 2483 | 7.06e+12 | 1.00× |
| RAG (k=3) | 4019 | 1.20e+13 | 1.69× |
| DRC | 4096 | 1.22e+13 | 1.73× |

### Generation = 64 output tokens

| Method | input tok/query | ≈FLOPs/query | ×vs Code2LoRA |
|---|---:|---:|---:|
| Pretrained | 2483 | 7.26e+12 | 1.00× |
| FFT | 2483 | 7.26e+12 | 1.00× |
| sLoRA | 2483 | 7.26e+12 | 1.00× |
| per-repo LoRA | 2483 | 7.26e+12 | 1.00× |
| Code2LoRA-Static | 2483 | 7.26e+12 | 1.00× |
| Code2LoRA-Evo | 2483 | 7.26e+12 | 1.00× |
| RAG (k=3) | 4019 | 1.22e+13 | 1.67× |
| DRC | 4096 | 1.24e+13 | 1.71× |

**Takeaway.** Code2LoRA/FFT/sLoRA feed only the prefix, so they use the least per-query compute (identical to the bare backbone). RAG and DRC re-encode 1.5K / 4K extra tokens on *every* query, costing ≈1.5–2× the prefill FLOPs and latency — Code2LoRA's advantage compounds over the thousands of queries a repo serves. Its only extra compute is the one-time adapter generation, amortized away.

