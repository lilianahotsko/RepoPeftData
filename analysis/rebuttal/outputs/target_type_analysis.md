# Base-LLM accuracy by target type and length (static track)

Base LLM = pretrained Qwen2.5-Coder-1.5B, prefix only, no repository context.


## cr_test (n=6414)

### By target type (Table 6 taxonomy)

| type | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| Numeric literal | 1274 | 19.9 | 71.4 | 51.0 | 41.3 | 74.9 | 73.6 | 70.3 |
| String literal | 1167 | 18.2 | 36.8 | 53.3 | 54.2 | 42.9 | 34.1 | 57.2 |
| Bool literal | 131 | 2.0 | 67.2 | 87.0 | 90.8 | 72.5 | 42.0 | 90.8 |
| None literal | 227 | 3.5 | 81.5 | 97.4 | 95.6 | 78.4 | 45.8 | 96.9 |
| Variable | 1405 | 21.9 | 56.3 | 44.8 | 39.2 | 57.2 | 41.7 | 77.3 |
| Collection | 657 | 10.2 | 17.5 | 36.4 | 31.8 | 20.7 | 15.4 | 42.6 |
| Function call | 651 | 10.1 | 20.4 | 34.1 | 31.8 | 19.7 | 16.6 | 32.0 |
| Complex expression | 902 | 14.1 | 31.3 | 66.6 | 64.4 | 33.0 | 28.5 | 68.4 |

### Collapsed groups

| group | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| Literals (num/str/bool/None) | 2799 | 43.6 | 57.6 | 57.4 | 53.4 | 61.7 | 53.4 | 68.0 |
| Variables | 1405 | 21.9 | 56.3 | 44.8 | 39.2 | 57.2 | 41.7 | 77.3 |
| Structured (collection/call/complex) | 2210 | 34.5 | 24.0 | 48.1 | 45.1 | 25.4 | 21.1 | 50.0 |

### By target length

| bin | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| <20 chars | 5334 | 83.2 | 51.3 | 57.6 | 53.3 | 54.2 | 44.4 | 71.7 |
| >=20 chars | 1080 | 16.8 | 18.3 | 20.7 | 18.4 | 18.8 | 16.6 | 24.7 |


## ir_test (n=5222)

### By target type (Table 6 taxonomy)

| type | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| Numeric literal | 1012 | 19.4 | 74.9 | 54.7 | 47.1 | 77.2 | 76.8 | 74.5 |
| String literal | 967 | 18.5 | 37.1 | 59.5 | 56.4 | 43.1 | 34.2 | 58.1 |
| Bool literal | 110 | 2.1 | 54.5 | 85.5 | 83.6 | 60.0 | 39.1 | 84.5 |
| None literal | 155 | 3.0 | 81.9 | 98.7 | 97.4 | 80.6 | 51.6 | 98.1 |
| Variable | 1140 | 21.8 | 58.2 | 50.8 | 40.8 | 59.3 | 45.4 | 80.9 |
| Collection | 588 | 11.3 | 15.1 | 38.6 | 32.3 | 19.6 | 13.6 | 39.6 |
| Function call | 466 | 8.9 | 22.1 | 45.5 | 41.0 | 22.5 | 18.7 | 45.9 |
| Complex expression | 784 | 15.0 | 36.4 | 66.8 | 66.5 | 38.3 | 36.2 | 67.5 |

### Collapsed groups

| group | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| Literals (num/str/bool/None) | 2244 | 43.0 | 58.1 | 61.3 | 56.4 | 61.9 | 54.9 | 69.6 |
| Variables | 1140 | 21.8 | 58.2 | 50.8 | 40.8 | 59.3 | 45.4 | 80.9 |
| Structured (collection/call/complex) | 1838 | 35.2 | 26.0 | 52.4 | 49.1 | 28.3 | 24.5 | 53.1 |

### By target length

| bin | n | % | Pretrained | FFT | sLoRA | DRC | RAG (k=3) | Code2LoRA-Static |
|---|---|---|---|---|---|---|---|---|
| <20 chars | 4442 | 85.1 | 52.1 | 60.9 | 55.7 | 55.0 | 46.8 | 73.1 |
| >=20 chars | 780 | 14.9 | 16.5 | 27.4 | 20.5 | 18.2 | 15.6 | 27.2 |

