# EM vs. repository size (yeRc-W3)


## cr_test  (n=6414, repos=52, missing size=0)

| repo size (approx tokens) | n inst | n repos | Pretrained | DRC | FFT | sLoRA | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|
| <25K | 534 | 10 | 36.7 | 38.8 | 51.3 | 47.0 | 57.9 |
| 25-75K | 1720 | 16 | 43.5 | 46.2 | 45.8 | 41.2 | 58.5 |
| 75-150K | 2293 | 15 | 47.4 | 49.6 | 52.0 | 47.7 | 65.8 |
| 150-300K | 1383 | 8 | 44.3 | 45.6 | 52.6 | 49.2 | 62.8 |
| >300K | 484 | 3 | 59.7 | 66.9 | 65.1 | 63.4 | 82.9 |

### Correlation of per-repo EM with log10(repo size)

| Method | Pearson r |
|---|---|
| Pretrained | +0.188 |
| DRC | +0.207 |
| FFT | +0.131 |
| sLoRA | +0.144 |
| Code2LoRA-Static | +0.241 |

## ir_test  (n=5222, repos=409, missing size=0)

| repo size (approx tokens) | n inst | n repos | Pretrained | DRC | FFT | sLoRA | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|
| <25K | 524 | 66 | 44.1 | 48.3 | 51.5 | 46.9 | 61.3 |
| 25-75K | 1551 | 137 | 45.6 | 48.7 | 54.2 | 49.4 | 63.4 |
| 75-150K | 1356 | 94 | 46.7 | 49.0 | 55.4 | 49.9 | 66.6 |
| 150-300K | 1021 | 66 | 50.0 | 53.3 | 61.4 | 54.5 | 72.2 |
| >300K | 770 | 46 | 47.1 | 47.9 | 55.8 | 50.4 | 66.9 |

### Correlation of per-repo EM with log10(repo size)

| Method | Pearson r |
|---|---|
| Pretrained | +0.040 |
| DRC | -0.009 |
| FFT | +0.050 |
| sLoRA | +0.031 |
| Code2LoRA-Static | +0.076 |

# Code2LoRA-Evo (GRU) — EM vs. repository size (evolution / per-commit track, yeRc-W3)


Same fixed-size repository embedding, evaluated on the per-commit evolution track. EM is QnA-weighted across all commits whose repository falls in each size bin.


## cr_test  (n=44732 QnAs, repos=51, missing size=0)

| repo size (approx tokens) | n QnAs | n repos | Code2LoRA-Evo |
|---|---|---|---|
| <25K | 4265 | 10 | 64.7 |
| 25-75K | 13639 | 16 | 56.5 |
| 75-150K | 13621 | 15 | 64.1 |
| 150-300K | 11970 | 7 | 57.4 |
| >300K | 1237 | 3 | 73.3 |

Per-repo EM vs. log10(repo size) Pearson r: **+0.105**


## ir_test  (n=42061 QnAs, repos=389, missing size=0)

| repo size (approx tokens) | n QnAs | n repos | Code2LoRA-Evo |
|---|---|---|---|
| <25K | 3518 | 65 | 63.3 |
| 25-75K | 12552 | 135 | 64.7 |
| 75-150K | 10612 | 90 | 64.8 |
| 150-300K | 8903 | 59 | 65.5 |
| >300K | 6476 | 40 | 62.8 |

Per-repo EM vs. log10(repo size) Pearson r: **-0.046**

