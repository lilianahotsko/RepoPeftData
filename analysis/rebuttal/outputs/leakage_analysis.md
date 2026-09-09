# Memorization / leakage analysis (yeRc-W4)

Training-target set built from the static-track train split.


## cr_test  (n=6414)

Novel-target rate (target string never seen in training): **50.5%** (3242/6414)

| Method | EM on novel targets % | EM on seen targets % | of correct preds, % seen in train |
|---|---|---|---|
| Pretrained | 32.8 | 58.9 | 63.7 |
| FFT | 42.2 | 60.9 | 58.6 |
| sLoRA | 39.3 | 55.8 | 58.2 |
| Code2LoRA-Static | 50.5 | 77.4 | 60.0 |

## ir_test  (n=5222)

Novel-target rate (target string never seen in training): **13.9%** (727/5222)

| Method | EM on novel targets % | EM on seen targets % | of correct preds, % seen in train |
|---|---|---|---|
| Pretrained | 31.5 | 49.3 | 90.6 |
| FFT | 42.5 | 58.0 | 89.4 |
| sLoRA | 36.5 | 52.7 | 89.9 |
| Code2LoRA-Static | 48.6 | 69.1 | 89.8 |
