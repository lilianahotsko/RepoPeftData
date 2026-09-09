# Rebuttal reanalysis (static track, stored predictions)


## cr_test  (n=6414)

### Per-method EM, strict-EM, and 95% CIs

| Method | EM % | strict-EM % | 95% CI (instance) | 95% CI (repo-clustered) |
|---|---|---|---|---|
| Pretrained | 45.7 | 44.4 | [44.5, 46.9] | [41.4, 49.8] |
| RAG (k=3) | 39.7 | 38.6 | [38.6, 40.9] | [35.6, 43.6] |
| DRC | 48.2 | 48.2 | [47.0, 49.5] | [43.7, 52.5] |
| FFT | 51.4 | 14.8 | [50.2, 52.7] | [46.8, 55.7] |
| sLoRA | 47.4 | 7.5 | [46.2, 48.6] | [42.8, 52.1] |
| Code2LoRA-Static | 63.8 | 63.8 | [62.6, 65.0] | [59.3, 68.0] |
| Code2LoRA-PAW | 64.1 | 64.0 | [62.9, 65.3] | [59.8, 68.2] |

### Fraction solvable from local prefix: **45.7%** (repo-knowledge-required subset: 3482/6414)

| Method | EM on repo-required % | EM on local-solvable % |
|---|---|---|
| Pretrained | 0.0 | 100.0 |
| RAG (k=3) | 8.0 | 77.4 |
| DRC | 9.1 | 94.7 |
| FFT | 33.7 | 72.5 |
| sLoRA | 32.1 | 65.6 |
| Code2LoRA-Static | 38.1 | 94.3 |
| Code2LoRA-PAW | 37.3 | 95.9 |

### EM by target length (chars)

| bin | n | Pretrained | RAG (k=3) | DRC | FFT | sLoRA | Code2LoRA-Static | Code2LoRA-PAW |
|---|---|---|---|---|---|---|---|---|
| [0,3) | 1489 | 61.9 | 62.4 | 65.5 | 58.7 | 52.0 | 74.8 | 74.4 |
| [3,6) | 1226 | 56.5 | 43.1 | 58.9 | 69.0 | 65.7 | 78.1 | 77.7 |
| [6,10) | 1076 | 47.6 | 38.1 | 49.9 | 58.2 | 56.1 | 72.6 | 72.4 |
| [10,20) | 1543 | 39.4 | 32.5 | 42.4 | 47.2 | 42.6 | 63.1 | 64.4 |
| [20,1000000) | 1080 | 18.3 | 16.6 | 18.8 | 20.7 | 18.4 | 24.7 | 25.8 |

### Headline comparisons (paired)

| Comparison | delta EM (pp) | 95% CI (pp) | P(delta>0) | McNemar b/c | McNemar p |
|---|---|---|---|---|---|
| Code2LoRA-Static vs FFT | +12.4 | [+11.4, +13.4] | 1.000 | 1016/221 | <1e-12 |
| Code2LoRA-Static vs DRC | +15.6 | [+14.4, +16.8] | 1.000 | 1293/293 | <1e-12 |
| Code2LoRA-Static vs sLoRA | +16.4 | [+15.3, +17.5] | 1.000 | 1238/187 | <1e-12 |
| Code2LoRA-Static vs Pretrained | +18.1 | [+17.0, +19.2] | 1.000 | 1328/167 | <1e-12 |
| Code2LoRA-Static vs RAG (k=3) | +24.1 | [+22.8, +25.4] | 1.000 | 1793/247 | <1e-12 |

## ir_test  (n=5222)

### Per-method EM, strict-EM, and 95% CIs

| Method | EM % | strict-EM % | 95% CI (instance) | 95% CI (repo-clustered) |
|---|---|---|---|---|
| Pretrained | 46.8 | 45.2 | [45.4, 48.2] | [44.6, 48.9] |
| RAG (k=3) | 42.1 | 40.7 | [40.8, 43.5] | [40.1, 44.1] |
| DRC | 49.5 | 49.5 | [48.2, 50.9] | [47.3, 51.7] |
| FFT | 55.9 | 18.4 | [54.5, 57.2] | [53.9, 57.8] |
| sLoRA | 50.4 | 9.7 | [49.0, 51.8] | [48.5, 52.2] |
| Code2LoRA-Static | 66.2 | 66.2 | [65.0, 67.5] | [64.2, 68.2] |
| Code2LoRA-PAW | 65.8 | 65.7 | [64.6, 67.1] | [63.9, 67.8] |

### Fraction solvable from local prefix: **46.8%** (repo-knowledge-required subset: 2777/5222)

| Method | EM on repo-required % | EM on local-solvable % |
|---|---|---|
| Pretrained | 0.0 | 100.0 |
| RAG (k=3) | 8.4 | 80.4 |
| DRC | 9.1 | 95.4 |
| FFT | 40.4 | 73.5 |
| sLoRA | 36.5 | 66.2 |
| Code2LoRA-Static | 41.5 | 94.3 |
| Code2LoRA-PAW | 39.2 | 96.1 |

### EM by target length (chars)

| bin | n | Pretrained | RAG (k=3) | DRC | FFT | sLoRA | Code2LoRA-Static | Code2LoRA-PAW |
|---|---|---|---|---|---|---|---|---|
| [0,3) | 1217 | 65.7 | 65.9 | 69.6 | 58.5 | 52.8 | 75.2 | 75.9 |
| [3,6) | 1109 | 54.6 | 44.7 | 57.3 | 72.0 | 67.9 | 78.8 | 78.7 |
| [6,10) | 898 | 49.1 | 40.8 | 51.7 | 61.6 | 55.7 | 75.3 | 75.3 |
| [10,20) | 1218 | 38.5 | 34.0 | 40.7 | 52.5 | 47.4 | 64.2 | 62.8 |
| [20,1000000) | 780 | 16.5 | 15.6 | 18.2 | 27.4 | 20.5 | 27.2 | 25.6 |

### Headline comparisons (paired)

| Comparison | delta EM (pp) | 95% CI (pp) | P(delta>0) | McNemar b/c | McNemar p |
|---|---|---|---|---|---|
| Code2LoRA-Static vs FFT | +10.4 | [+9.2, +11.5] | 1.000 | 790/249 | <1e-12 |
| Code2LoRA-Static vs DRC | +16.7 | [+15.5, +18.0] | 1.000 | 1101/227 | <1e-12 |
| Code2LoRA-Static vs sLoRA | +15.8 | [+14.7, +17.0] | 1.000 | 1008/181 | <1e-12 |
| Code2LoRA-Static vs Pretrained | +19.4 | [+18.2, +20.6] | 1.000 | 1153/139 | <1e-12 |
| Code2LoRA-Static vs RAG (k=3) | +24.1 | [+22.7, +25.5] | 1.000 | 1468/209 | <1e-12 |
