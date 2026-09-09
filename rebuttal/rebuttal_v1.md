Reply 1: to reviewer v1i3
# start here
Thank you for the careful review, and for recognizing that Code2LoRA-Evo (a recurrent hypernetwork over sequential commit diffs, with no analogue in the Text2LoRA/Doc2LoRA line) is new, and that releasing each repository whole is a real step over retrieval-sliced benchmarks.
## (Weakness 1 + Suggestion 1) 

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
| Static — CR | 45.7 | 39.7 → **47.6** | 48.2 → 49.1 | **63.8** |
| Static — IR | 46.8 | 42.1 → **48.0** | 49.5 → 50.0 | **66.2** |
| Evolution — CR | 31.5 | 23.6 → **36.8** | 31.1 → **36.4** | **60.3** |
| Evolution — IR | 29.3 | 23.0 → **37.4** | 31.6 → **36.9** | **64.5** |
| OOD | 44.6 | 32.6 → **55.6** | 45.5 → **56.7** | **74.1** |


## (Weakness 2) ……

> Benchmark validity is unclear because the task is very local and the targets are very short.

While the target length is short(median ≈3 tokens, inherent to assertion-RHS completion), assertion completion is a non-trivial task that requires the LLM to extract repository-level context and reason about code execution (Jain et al, 2024, LiveCodeBench) before generating the final output. As shown in Table 6, the target can include complex data structures and expressions.
The base LLM without repository context collapses on the harder-but-short cases (static CR / IR):
By type: complex expressions 31.3 / 36.4%, function calls 20.4 / 22.1%, collections 17.5 / 15.1%,  vs. 71.4 / 74.9% on numeric literals.
By length: 51.3 / 52.1% on targets <20 chars drops to 18.3 / 16.5% on ≥20 chars.
So over half the instances are not locally solvable, with difficulty concentrated on structured and longer targets. Code2LoRA's gains land exactly there (static CR: complex expressions 31.3 → 68.4, collections 17.5 → 42.6, variables 56.3 → 77.3), while leaving already-local numeric literals unchanged (71.4 → 70.3), and target-type composition matches across train / CR / IR within ≈2 pp (Table 6), so these are generalization, not reweighting toward easier categories.

## (Weakness 3) ……

> The headline results are oversold relative to their statistical support.

Thank you for the suggestion. We are adding statistical significance testing to support our results, and will revisit our manuscript and soften the claims of headline results if needed.

To clarify, the “Per-Repo LoRA” that demonstrates good performance on IR split should be considered as an upper-bound, because it incurs much more training cost by fine-tuning for each repository separately and does not generalize to CR or OOD splits where no per-repo training data is available. The primary comparisons should be done between Code2LoRA against our baselines under a similar level of training/inference cost.

Code2LoRA outperforms the strongest baseline by large margins for most of evaluation settings: on static track, Code2LoRA-Static outperforms DRC by +14.7pp (CR) / +16.2pp (IR); on evolution track, Code2LoRA-Evo outperforms Single LoRA by +5.2pp (CR) / +3.2pp (IR). The gap on the OOD split is relatively smaller, where Code2LoRA-Evo outperforms Single LoRA by +1.8pp, mainly because the OOD prediction targets are simpler (shorter) than those in the training set and thus the base model is more proficient in solving them. All differences have statistical significance confirmed using a paired bootstrap test under a confidence level of 95%.

## cr_test (n=6414)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) | 
|---|---|---|---|
| Code2LoRA-Static vs FFT | +12.4 | <1e-5 | +11.5 | 
| Code2LoRA-Static vs DRC | +15.6 | <1e-5 | +14.6 | 
| Code2LoRA-Static vs sLoRA | +16.4 | <1e-5 | +15.5 | 
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +23.0 | 
| Code2LoRA-Static vs Pretrained | +18.1 | <1e-5 | +17.2 | 

## ir_test (n=5222)

| Comparison | Observed ΔEM (pp) | one-sided p | 95% lower bound (pp) |
|---|---|---|---|
| Code2LoRA-Static vs FFT | +10.4 | <1e-5 | +9.4 | 
| Code2LoRA-Static vs DRC | +16.7 | <1e-5 | +15.6 | 
| Code2LoRA-Static vs sLoRA | +15.8 | <1e-5 | +14.8 | 
| Code2LoRA-Static vs RAG (k=3) | +24.1 | <1e-5 | +22.9 | 
| Code2LoRA-Static vs Pretrained | +19.4 | <1e-5 | +18.4 | 


(Suggestion 2) ……
> What fraction of RepoPeftBench is solvable from the local prefix alone, and how does the Code2LoRA-vs-baseline gap look restricted to instances that genuinely require repository knowledge? Please also specify the exact "relaxed" EM rule.

Liliana:
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


# end here
# Characters remaining: …
Reply 2: to reviewer yeRc
# start here
Thank you for the constructive review and for recognizing RepoPeftBench and the parametric-injection paradigm. Below we tighten the claims and report new analyses. We group each weakness with its matching suggestion:
## (Weakness 1 + Suggestion 1) The "zero inference cost" claim is overstated

> While the paper emphasizes zero inference-time token overhead, it effectively shifts computation rather than eliminating it. Generating repository embeddings and LoRA weights requires substantial preprocessing and hypernetwork inference. In realistic scenarios—such as frequent updates or large numbers of repositories—this introduces non-trivial latency and computational overhead. Furthermore, Code2LoRA-Evo requires per-commit updates, which entails continuous computation during usage. This contradicts the claimed notion of "zero overhead" and weakens a central argument of the paper.
> The authors should clarify the "zero inference cost" claim by explicitly accounting for preprocessing, hypernetwork inference, and update costs. Reporting end-to-end latency comparisons with strong baselines (e.g., RAG) would make the efficiency claims more realistic and convincing.

We agree that our wording could be interpreted too broadly. Our claim is specifically zero additional per-query context tokens, rather than zero end-to-end computation.
At query time the merged-adapter model reads only the test prefix (0 extra tokens), when RAG re-encodes ≈1,536 and DRC ≈4,096 context tokens on every query. Adapter generation is a one-time, per-repo-state cost amortized over all the queries a repository receives; measured end-to-end on a single H100: full cold build 34–257 s (188–1,073 commits), incremental per-commit update ≈21–23 s (cache hit near-instant), and even on CPU (8 threads) it is only 344–1,079 s cold / 30–297 s incremental. The step is CPU/parse-bound, so it might be deployed without GPU. Crucially the Evo per-commit update is an offline work done at commit time (one diff-embed + one recurrent step), not a per-inference cost, so there is no "continuous computation during usage." 
We will modify the abstract to "zero inference-time token overhead" and add this latency/ storage paragraph.



## (Weakness 2) + ## (Suggestion 2) Limited or potentially biased comparison with strong baselines 

> The comparisons with context-injection methods (e.g., RAG) and LoRA baselines are not fully convincing. The evaluation appears biased toward parameter-based approaches, making the reported gains less compelling. In particular, RAG is implemented with a limited retrieval depth (k=3), which is not representative of stronger modern configurations. Additionally, the paper does not compare against state-of-the-art long-context LLMs or hybrid approaches (e.g., retrieval combined with routing or caching). Finally, treating per-repository LoRA as an upper bound only in the in-repo (IR) setting implicitly advantages the proposed method in cross-repo evaluations.

> The evaluation should include stronger and more comprehensive baselines, such as improved RAG configurations, long-context LLMs, and hybrid retrieval–adaptation methods. Ensuring fair comparisons under similar compute budgets and clarifying the “upper bound” role of per-repo LoRA would improve credibility.

We agree that our current experiments do not exhaust the space of retrieval and hybrid repository-conditioning systems. Our conclusion should be interpreted within the controlled setting of RepoPeftBench, Qwen2.5-Coder-1.5B, and the evaluated retrieval pipelines. We note, however, that $k=3$ was not selected arbitrarily. Appendix C sweeps $k \in \{3, 5, 10\}$ and multiple chunk sizes, and $k=3$ is the strongest configuration in our setup. Thus, increasing retrieval depth alone does not close the gap. We will narrow the claim from “parameters are generally better than context” to “parametric repository adaptation outperforms the evaluated retrieve-and-prepend and dependency-resolved baselines in our setting,” 
However, we improved the RAG and DRC baseline numbers by changing the context injection method. New results:
Static track: using the backbone's native repository-completion layout (<|repo_name|>/<|file_sep|>-delimited files) instead of raw prepending lifts RAG from 39.7 / 42.1 → 47.6 / 48.0 (CR / IR), now above pretrained (45.7 / 46.8).


Evolution (per-commit) track: here we also correct our own reporting. The submitted RAG figure (23.6 / 23.0, below pretrained's 31.5 / 29.3) came from a handicapped configuration, generic (non-AST) chunks, and no decode-time ban on the control tokens above. The updated context injection method increased the RAG baseline to 36.8/37.4 (CR/IR). 
Stronger and hybrid context baselines: the properly repo/FIM-formatted retrieval baseline now beats pretrained (47.6 / 48.0) yet stays ≈16–18 pp below Code2LoRA-Static; DRC is itself a retrieval+compression hybrid that also beats pretrained (49.1 / 50.0). All remain well below Code2LoRA-Static - parametric adaptation wins even against fair, pretrained-beating, and hybrid context baselines. 
Per-repo LoRA's IR-only role is definitional: it requires per-repo training instances, which exist only for IR; CR repos are fully held out, so a per-repo adapter there would train on the eval data. And Code2LoRA-Static matches this upper bound in IR (66.2 vs. 64.0) with no per-repo training - the opposite of being flattered.
Comparison to long context LLMs: we specifically wanted to isolate the context preserving methods within the model with the context of only 32k tokens.

We believe that RAG and Code2LoRA could be complementary methods used in combination in some cases. 
Matched Budgets: Code2LoRA is in fact the cheapest method at inference. All methods share the same frozen backbone (Qwen2.5-Coder-1.5B), so per-query cost is governed by input length. The merged adapter adds 0 tokens and is latency-identical to the bare backbone (197.1 vs. 198.3 ms at in=2048/out=64, Δ≈−1.2 ms), whereas RAG re-encodes ≈1,536 and DRC ≈4,096 extra context tokens on every query. Measured on one H100 (Qwen2.5-Coder-1.5B, greedy), per-query latency at a representative ≈2,048-token prefix is:
| Method | input tok | out=8 (ms) | out=64 (ms) |
|---|---:|---:|---:|
| Pretrained / FFT / sLoRA / **Code2LoRA** | ≈2,048 | 36.0 | 198.3 |
| RAG (k=3) | ≈3,584 | 44.5 (+24%) | 206.6 (+4%) |
| DRC | ≈4,096 | 53.3 (+48%) | 216.8 (+9%) |
Approximate forward-pass FLOPs/query (dense 2·N_nonembed per token + attention + lm-head, N_nonembed≈1.31B) confirm this: Code2LoRA ≈5.4×10¹² (CR) / 7.1×10¹² (IR), vs. RAG ≈1.0–1.2×10¹³ and DRC ≈1.2×10¹³, i.e. RAG/DRC spend ≈1.7–2.3× the per-query compute, on every query, for context that Code2LoRA has already internalized parametrically. Code2LoRA's only extra compute is the one-time adapter generation. So under matched budgets Code2LoRA is the least expensive at inference.

## (Weakness 3) + ## (Suggestion 3) Concerns about scalability to real-world codebases

> The approach assumes that a single fixed-size embedding can faithfully summarize an entire repository and serve as the basis for generating all LoRA weights. However, real-world repositories often span millions of tokens and exhibit complex, heterogeneous structures. The use of mean-and-max pooled embeddings is likely to lose fine-grained dependencies and semantic relationships. This raises a fundamental concern: can such a compressed representation adequately capture repository-level semantics at scale? The paper does not provide sufficient evidence to support this assumption.

> The scalability of the repository encoding should be validated on larger and more complex repositories. Exploring richer encoding strategies (e.g., hierarchical or graph-based) and providing ablations across repository sizes would strengthen confidence in the approach.
We ran the requested ablation on binning every test instance by total repository source size and reporting EM per bin:
EM by repository size (CR-test, n=6,414 / 52 repos):
| repo size (≈tok) | N of instances | N of repos | Pretrained | DRC | FFT | sLoRA | C2L-Static |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25K | 534 | 10 | 36.7 | 38.8 | 51.3 | 47.0 | **57.9** |
| 25–75K | 1720 | 16 | 43.5 | 46.2 | 45.8 | 41.2 | **58.5** |
| 75–150K | 2293 | 15 | 47.4 | 49.6 | 52.0 | 47.7 | **65.8** |
| 150–300K | 1383 | 8 | 44.3 | 45.6 | 52.6 | 49.2 | **62.8** |
| >300K | 484 | 3 | 59.7 | 66.9 | 65.1 | 63.4 | **82.9** |

EM by repository size (IR-test, n=5,222 / 409 repos):
| repo size (≈tok) | N of instances | N of repos | Pretrained | DRC | FFT | sLoRA | C2L-Static |
|---|---:|---:|---:|---:|---:|---:|---:|
| <25K | 524 | 66 | 44.1 | 48.3 | 51.5 | 46.9 | **61.3** |
| 25–75K | 1551 | 137 | 45.6 | 48.7 | 54.2 | 49.4 | **63.4** |
| 75–150K | 1356 | 94 | 46.7 | 49.0 | 55.4 | 49.9 | **66.6** |
| 150–300K | 1021 | 66 | 50.0 | 53.3 | 61.4 | 54.5 | **72.2** |
| >300K | 770 | 46 | 47.1 | 47.9 | 55.8 | 50.4 | **66.9** |
We do note that absolute EM dips in the largest IR bin (Code2LoRA 72.2 → 66.9). Crucially this is not specific to Code2LoRA or its embedding: every method drops in lockstep there (Pretrained 50.0 → 47.1, DRC 53.3 → 47.9, FFT 61.4 → 55.8, sLoRA 54.5 → 50.4), so it reflects intrinsically harder completions in those repositories rather than the pooled embedding saturating - Code2LoRA's lead in that bin is in fact the widest on IR (+11.1 pp over FFT). Consistently, the per-repo EM-log-size correlation is flat-to-slightly-positive for all methods (Code2LoRA +0.24 CR, +0.08 IR), i.e. no systematic decline with size.
(The CR >300K bin is only 3 repos, so we rely on the "leads in every bin on both splits" result rather than that single bin; IR >300K is 46 repos.) We agree a single pooled vector may eventually limit truly gigantic repositories, and flag hierarchical/graph encoders as future work, but at the scales tested it degrades neither absolutely (the correlation is non-negative) nor relative to any baseline. We will add both tables to the paper.

The recurrent embedding scales too (Code2LoRA-Evo / GRU). We repeated the ablation on the per-commit evolution track for Code2LoRA-Evo, which summarizes the repo with a single fixed-size GRU state updated commit-by-commit (QnA-weighted EM per size bin). It is likewise flat - no size-driven decay, including on the largest repos:
| repo size (≈tok) | CR: QnAs / Evo | IR: QnAs / Evo |
|---|---:|---:|
| <25K | 4,265 / **64.7** | 3,518 / **63.3** |
| 25–75K | 13,639 / **56.5** | 12,552 / **64.7** |
| 75–150K | 13,621 / **64.1** | 10,612 / **64.8** |
| 150–300K | 11,970 / **57.4** | 8,903 / **65.5** |
| >300K | 1,237 / **73.3** | 6,476 / **62.8** |
Per-repo EM vs. log10(size) is again essentially flat (Pearson +0.105 CR, −0.046 IR), so the recurrent fixed-size embedding shows no scaling penalty either.
Across the per-bin evaluations, provided in the Weakness 3 experiments, repository source size (Qwen2.5-Coder-1.5B tokenizer, non-test Python) has median 165K, mean 285K, p95 ≈1.03M, p99 ≈1.87M, and max ≈3.0M tokens i.e. the ">300K" bin already reaches into the millions, well past a single context window, and Code2LoRA's lead there is the widest. We will report these max/percentile sizes next to the ablation.


## (Weakness 4) + ## (Suggestion 4) Potential memorisation-driven overfitting in the hypernetwork

> Although the method avoids explicit per-repository fine-tuning, it introduces a very large hypernetwork (~700M parameters). This creates the possibility that the model memorizes patterns across the training repositories rather than learning robust adaptation mechanisms. Given that training is conducted on hundreds of repositories and evaluation includes in-distribution data, the reported gains may reflect memorisation effects. The paper does not adequately rule out this alternative explanation, for example through stronger out-of-distribution or adversarial evaluations.

> The paper should better address potential memorisation in the large hypernetwork. Stronger out-of-distribution, domain-shift, or adversarial evaluations—along with perturbation or leakage analyses—would help demonstrate true generalisation rather than memorisation.

Memorization is controlled directly by our evaluation design, which reports every method across three regimes of decreasing training exposure:
In-repo (IR): the repository was seen in training, but the evaluated completions are held out, this measures use of exposed repository knowledge on unseen instances.
Cross-repo (CR): the 52 test repositories are entirely absent from training (headline 63.8% EM). Since the hypernetwork encounters each repository for the first time at test, outperforming every baseline here cannot be repository memorization.
Held-out OOD: repositories created after the training-data cutoff, so provably in neither our training set nor the backbone's pretraining, here Code2LoRA-Evo is strongest.

A direct leakage analysis: Setup: we collect the 20,861 unique answer strings (assertion right-hand sides) that appear anywhere in the training split, and label each CR-test instance by whether its answer string was seen in training or never seen (impossible to have memorized); 50.5% of CR-test answers are never-seen. We then read three quantities per method:
| CR-test (n=6,414) | EM on never-seen answers | EM on seen answers | of *correct* preds, % whose answer was seen |
|---|---:|---:|---:|
| Pretrained (never trained on our data - control) | 32.8 | 58.9 | **63.7** |
| FFT | 42.2 | 60.9 | 58.6 |
| sLoRA | 39.3 | 55.8 | 58.2 |
| **Code2LoRA-Static** | **50.5** | **77.4** | **60.0** |
Two things follow. 
(i) On answers it could not have copied (never in training), Code2LoRA scores 50.5%, above FFT (42.2) and pretrained (32.8) - it generalizes to unseen answer strings. 
(ii) The third column is a memorization probe: of the questions each method gets right, how many had a training-seen answer? The pretrained model is the control - it never trained on our data, so its 63.7% is the overlap expected purely because common values (True, 0, None, …) recur. Code2LoRA's correct answers are 60.0% seen - below that control - so it is not disproportionately winning on memorizable answers; a memorizer would sit above the control, not below. (IR shows the same pattern; there the overlap is ≈90% for every method because IR repositories were in training, so no method stands out.) We will add this analysis to the appendix.

## (Weakness 5) Potential bias in the benchmark toward the proposed method

> RepoPeftBench provides full repository access and uses assertion-completion tasks that rely on shared repository context. This setup inherently favors methods that encode global repository information into parameters, while disadvantaging retrieval-based approaches that operate on selectively retrieved context. As a result, the benchmark may implicitly bias results toward the design philosophy of Code2LoRA, limiting the generality of the conclusions.
Releasing each repository whole gives retrieval more to work with, not less, nothing is hidden from RAG/DRC. What limits retrieval is its own selection step, not information the benchmark withholds
The residual gap is a property of the methods, not the benchmark: on instances that genuinely require repository knowledge, retrieval still recovers only ≈10% vs. ≈38–42% for parametric adaptation, both see the same whole repository, but a few chunks cannot surface globally-scattered facts, which is exactly the distinction the benchmark measures. The task is also externally motivated (assertion completion from LiveCodeBench, not tailored to us), with target-type composition matched across splits within ≈2 pp. We state the scope as one task, one language, one backbone, but within it the benchmark treats retrieval fairly.


## (Weakness 6) Limited evaluation of real-world applicability and usability

> The paper does not evaluate how the method performs in realistic deployment scenarios. Key aspects such as integration into developer workflows, latency of adapter generation, and memory or storage overhead for large numbers of repositories are not addressed. Without this analysis, it remains unclear whether the approach is practically scalable, despite its strong performance in offline benchmarks.
Adapter-generation latency is a one-time, offline, CPU/parse-bound step (no GPU): full cold build 34–257 s (H100) / 344–1,079 s (8 CPU threads), incremental per-commit ≈21–23 s (GPU) / 30–297 s (CPU), amortized over the thousands of completions a repository serves. Storage scales with #repos as a strength: the per-repository footprint is ≈8 KB (repo embedding / GRU state) vs. ≈37 MB × N for stored per-repo adapters, since the shared hypernetwork (≈1.4 GB, repo-independent) regenerates any adapter in one forward pass, so storage is effectively constant. 
Workflow integration: We are building a CPU-supported integration layer for integration with coding assistants   to improve the applicability and usability. Mainly, the users will regenerate/incrementally update a repo's adapter offline at commit/push time, then serve with zero added inference tokens, identical to the bare backbone.