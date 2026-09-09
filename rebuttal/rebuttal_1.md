Thank you for the careful review, and for recognizing that Code2LoRA-Evo (a recurrent hypernetwork over sequential commit diffs, with no analogue in the Text2LoRA/Doc2LoRA line) is new, and that releasing each repository whole is a real step over retrieval-sliced benchmarks.

## Weakness 1 + Suggestion 1

> The central claim "repository knowledge is best injected parametrically rather than through context" rests on context baselines that are broken rather than merely weaker. RAG does not just underperform Code2LoRA; it falls below the no-context pretrained model on every track. A retrieval baseline that is worse than doing nothing signals a setup problem, not a fair comparison. Since the test prefix already contains the local test file, prepending retrieved chunks may simply push the completion point further away and add noise. Without a properly formatted, FIM/repo-aware in-context baseline that at least improves over pretrained, the parametric-vs-context conclusion is not established.

> RAG performs worse than the no-context model on every track. Is the in-context baseline formatted appropriately for this backbone? Can you provide a context-injection baseline that at least improves over pretrained, so the parametric-vs-context comparison is fair?


A retrieval baseline indicated a setup problem, so we investigated and have corrected it. The root cause is a naively prepended context, under which Qwen2.5-Coder degenerates into emitting its own control tokens (<|fim_suffix|>, <|file_sep|>, <|repo_name|>) instead of continuing the code. This is a formatting/decoding artifact, not evidence that context is fundamentally inferior. We rebuilt the in-context baseline, holding the retriever, index, k=3, and context window fixed, and it now beats the pretrained model on both tracks:

Static track: using the backbone's native repository-completion layout (<|repo_name|>/<|file_sep|>-delimited files) instead of raw prepending lifts RAG from 39.7 / 42.1 → 47.6 / 48.0 (CR / IR), now above pretrained (45.7 / 46.8).


Evolution (per-commit) track: here we also correct our own reporting. The submitted RAG figure (23.6 / 23.0, below pretrained's 31.5 / 29.3) came from a handicapped configuration, generic (non-AST) chunks, and no decode-time ban on the control tokens above. The updated context injection method increased the RAG baseline to 36.8/37.4 (CR/IR). 


Also, we applied the same injection methods to the DRC baseline, which increased this baseline numbers as well: 31.1 → 36.4 and 31.6 → 36.9 on the evolution track (CR/IR) and 45.5 → 56.7 on the OOD set.

Even against these corrected, pretrained-beating baselines, the conclusion is unchanged and does not rest on RAG being broken: Code2LoRA-Static (63.8 / 66.2) still leads the best in-context baseline on the static track, and Code2LoRA-Evo (60.3 / 64.5) dominates on the evolution track. We will (i) replace the RAG/DRC rows with these corrected numbers, (ii) explicitly note the configuration change from the submitted version, and (iii) describe the native-format/decoding fix in the evaluation setup.

**Supplementary Table 1: Corrected RAG and DRC baselines**
| Track | Pretrained | RAG: paper → fixed | DRC: paper → fixed | Code2LoRA |
|---|---:|---:|---:|---:|
| Static — CR | 45.7 | 39.7 → **47.6** | 48.2 → **49.1** | **63.8** |
| Static — IR | 46.8 | 42.1 → **48.0** | 49.5 → **50.0** | **66.2** |
| Evolution — CR | 31.5 | 23.6 → **36.8** | 31.1 → **36.4** | **60.3** |
| Evolution — IR | 29.3 | 23.0 → **37.4** | 31.6 → **36.9** | **64.5** |
| OOD | 44.6 | 32.6 → **55.6** | 45.5 → **56.7** | **74.1** |


## Weakness 2

> Benchmark validity is unclear because the task is very local and the targets are very short.

While the target length is short(median ≈3 tokens, inherent to assertion-RHS completion), assertion completion is a non-trivial task that requires the LLM to extract repository-level context and reason about code execution (Jain et al, 2024, LiveCodeBench) before generating the final output. As shown in Table 6, the target can include complex data structures and expressions.

The base LLM without repository context collapses on the harder-but-short cases (static CR / IR):

- By type: complex expressions 31.3 / 36.4%, function calls 20.4 / 22.1%, collections 17.5 / 15.1%,  vs. 71.4 / 74.9% on numeric literals.

- By length: 51.3 / 52.1% on targets <20 chars drops to 18.3 / 16.5% on ≥20 chars.

So over half the instances are not locally solvable, with difficulty concentrated on structured and longer targets. Code2LoRA's gains land exactly there (static CR: complex expressions 31.3 → 68.4, collections 17.5 → 42.6, variables 56.3 → 77.3), while leaving already-local numeric literals unchanged (71.4 → 70.3), and target-type composition matches across train / CR / IR within ≈2 pp (Table 6), so these are generalization, not reweighting toward easier categories.

## Weakness 3

> The headline results are oversold relative to their statistical support.

Thank you for the suggestion. We are adding statistical significance testing to support our results, and will revisit our manuscript and soften the claims of headline results if needed.

To clarify, the “Per-Repo LoRA” that demonstrates good performance on IR split should be considered as an upper-bound, because it incurs much more training cost by fine-tuning for each repository separately and does not generalize to CR or OOD splits where no per-repo training data is available. The primary comparisons should be done between Code2LoRA against our baselines under a similar level of training/inference cost.

Code2LoRA outperforms the strongest baseline by large margins for most of evaluation settings: on static track, Code2LoRA-Static outperforms DRC by +14.7pp (CR) / +16.2pp (IR); on evolution track, Code2LoRA-Evo outperforms Single LoRA by +5.2pp (CR) / +3.2pp (IR). The gap on the OOD split is relatively smaller, where Code2LoRA-Evo outperforms Single LoRA by +1.8pp, mainly because the OOD prediction targets are simpler (shorter) than those in the training set and thus the base model is more proficient in solving them. All differences have statistical significance confirmed using a paired bootstrap test under a confidence level of 95%.

## cr_test (n=6414)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) | 
|---|---|---|---|
| Code2LoRA-Static vs FFT | +12.4 | <1e-5 | +11.5 | 
| Code2LoRA-Static vs DRC | +14.7 | <1e-5 | +13.8 | 
| Code2LoRA-Static vs sLoRA | +16.4 | <1e-5 | +15.5 | 
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +23.0 | 
| Code2LoRA-Static vs Pretrained | +18.1 | <1e-5 | +17.2 | 

## ir_test (n=5222)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) |
|---|---|---|---|
| Code2LoRA-Static vs FFT | +10.4 | <1e-5 | +9.4 | 
| Code2LoRA-Static vs DRC  | +16.2 | <1e-5 | +15.1 | 
| Code2LoRA-Static vs sLoRA | +15.8 | <1e-5 | +14.8 | 
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +22.9 | 
| Code2LoRA-Static vs Pretrained | +19.4 | <1e-5 | +18.4 | 


## Suggestion 2
> What fraction of RepoPeftBench is solvable from the local prefix alone, and how does the Code2LoRA-vs-baseline gap look restricted to instances that genuinely require repository knowledge? Please also specify the exact "relaxed" EM rule.

Fraction solvable from the prefix alone. Taking "the no-context pretrained model solves it" as the definition, 45.7% (CR) / 46.8% (IR) is locally solvable; the majority (~54% / ~53%) is not, so the benchmark is not dominated by trivially local completions.

Gap on the repository-knowledge-required subset:
Restricting to the instances the local-prefix model fails:
| Method | CR | IR |
|---|---:|---:|
| Code2LoRA-Static | 38.1 | 41.5 |
| FFT | 33.7 | 40.4 |
| sLoRA | 32.1 | 36.5 |
| RAG | 10.1 | 9.7 |
| DRC | 9.1 | 9.1 |

Locally-solvable subset:
| Method | CR | IR |
|---|---:|---:|
| DRC | 94.7 | 95.4 |
| **Code2LoRA-Static** | 94.3 | 94.3 |
| RAG (corrected) | 92.2 | 91.5 |
| FFT | 72.5 | 73.5 |
| sLoRA | 65.6 | 66.2 |
Context injection barely moves these instances (~9–10%), and parametric adaptation recovers ~38–42%; on the locally-solvable complement all methods are near-ceiling (Code2LoRA 94–96%), therefore, the advantage is concentrated exactly on the instances that require repository knowledge.

Exact "relaxed" EM rule:
We strip Qwen FIM sentinels, drop trailing comments, and keep the leading segment of the generation at its natural boundary (first line; and, matching the short assertion targets, up to the first comma or first token). Both strings then receive identical normalization (strip surrounding/trailing whitespace and { : , }, collapse internal whitespace) and must be exactly equal for EM=1. Since only the start of the generation is kept, a method is credited only when it produces the answer first, never when a correct answer is generated after other code. 
