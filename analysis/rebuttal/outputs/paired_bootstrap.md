# Paired bootstrap significance (static track)

scipy.stats.bootstrap, paired, one-sided (greater), percentile, n_resamples=100,000, seed=42. Statistic = mean(Code2LoRA-Static − baseline) EM.


## cr_test (n=6414)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) | significant |
|---|---|---|---|---|
| Code2LoRA-Static vs FFT | +12.4 | <1e-5 | +11.5 | yes |
| Code2LoRA-Static vs DRC | +15.6 | <1e-5 | +14.6 | yes |
| Code2LoRA-Static vs sLoRA | +16.4 | <1e-5 | +15.5 | yes |
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +23.0 | yes |
| Code2LoRA-Static vs Pretrained | +18.1 | <1e-5 | +17.2 | yes |

## ir_test (n=5222)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) | significant |
|---|---|---|---|---|
| Code2LoRA-Static vs FFT | +10.4 | <1e-5 | +9.4 | yes |
| Code2LoRA-Static vs DRC | +16.7 | <1e-5 | +15.6 | yes |
| Code2LoRA-Static vs sLoRA | +15.8 | <1e-5 | +14.8 | yes |
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +22.9 | yes |
| Code2LoRA-Static vs Pretrained | +19.4 | <1e-5 | +18.4 | yes |
