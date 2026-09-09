Thank you for the constructive review and for recognizing RepoPeftBench and the parametric-injection paradigm. Below we tighten the claims and report new analyses:

## Weakness 1 + Suggestion 1: The "zero inference cost" claim is overstated

We agree that our wording could be interpreted too broadly. Our claim is specifically zero additional per-query context tokens, rather than zero end-to-end computation.

At query time the merged-adapter model reads only the test prefix (0 extra tokens). Adapter generation is a one-time, per-repo-state cost amortized over all the queries a repository receives; measured end-to-end on a single H100: full cold build 34–257 s (188–1,073 commits), incremental per-commit update ≈21–23 s, and even on CPU (8 threads, no GPU) it is only 344–1,079 s cold / 30–297 s incremental. The Evo per-commit update is offline work done at commit time (one diff-embed + one recurrent step), not a per-inference cost, so there is no "continuous computation during usage."

We will modify the abstract to "zero inference-time token overhead" and add this latency/storage paragraph.


## Weakness 2 + Suggestion 2: Limited or potentially biased comparison with strong baselines

We agree our experiments do not exhaust the space of retrieval and hybrid systems; our conclusion holds within the controlled setting of RepoPeftBench, Qwen2.5-Coder-1.5B, and the evaluated pipelines. But $k=3$ is not arbitrary — the Appendix C sweep over $k\in\{3,5,10\}$ and chunk sizes finds it strongest, so more depth does not close the gap. We will narrow the claim from "parameters are generally better than context" to "parametric adaptation outperforms the evaluated retrieve-and-prepend and dependency-resolved baselines in our setting."

We also strengthened the context baselines by fixing the injection format. **Static track:** the backbone's native repository-completion layout (`<|repo_name|>`/`<|file_sep|>`) lifts RAG 39.7 / 42.1 → **47.6 / 48.0** (CR / IR). **Evolution track:** the submitted RAG used generic non-AST chunks and no decode-time control-token ban; corrected, RAG reaches **36.8 / 37.4**.


**Stronger/hybrid baselines still trail:** repo-FIM RAG (47.6 / 48.0) stays ≈16–18 pp below Code2LoRA-Static; DRC (a retrieval+compression hybrid) reaches 49.1 / 50.0; a trained+retrieval hybrid helps too (FFT+RAG 53.9 / 56.8 > FFT 51.4 / 55.9). All remain well below Code2LoRA-Static.

**Long-context LLMs:** we hold the backbone fixed across all methods to isolate the repository-injection mechanism; long-context modeling is complementary (parametric adaptation can sit on top of a long-context backbone and still avoids re-encoding the repo per query), which we flag as future work.

**Per-repo LoRA's IR-only role is definitional:** it needs per-repo training instances, which exist only for IR (CR repos are held out, so training there would leak the eval data). Yet Code2LoRA-Static matches this upper bound in IR (66.2 vs. 64.0) with no per-repo training.

**Matched budgets:** all methods share the frozen backbone, so per-query cost tracks input length. The merged adapter adds 0 tokens and is latency-identical to the bare backbone (197.1 vs. 198.3 ms at in=2048/out=64), whereas RAG/DRC re-encode ≈1,536/≈4,096 extra tokens every query (H100, greedy, ≈2,048-token prefix):

| Method | input tok | out=8 (ms) | out=64 (ms) |
|---|---:|---:|---:|
| Pretrained / FFT / sLoRA / **Code2LoRA** | ≈2,048 | 36.0 | 198.3 |
| RAG (k=3) | ≈3,584 | 44.5 (+24%) | 206.6 (+4%) |
| DRC | ≈4,096 | 53.3 (+48%) | 216.8 (+9%) |

Forward-pass FLOPs/query (dense 2·N_nonembed/token + attention + lm-head, N_nonembed≈1.31B) confirm this: Code2LoRA ≈5.4×10¹² (CR) / 7.1×10¹² (IR) vs. RAG ≈1.0–1.2×10¹³ and DRC ≈1.2×10¹³ — RAG/DRC spend ≈1.7–2.3× the per-query compute for context Code2LoRA has internalized parametrically; its only extra cost is the one-time adapter generation.

## Weakness 6: Limited evaluation of real-world applicability and usability

**Adapter-generation latency** is the one-time, offline, GPU-free step quantified in W1, amortized over the thousands of completions a repository serves. **Storage scales with repos as a strength:** the per-repository footprint is ≈8 KB (repo embedding / GRU state) vs. ≈37 MB × N for stored per-repo adapters, since the shared hypernetwork (≈1.4 GB, repo-independent) regenerates any adapter in one forward pass — so storage is effectively constant.

**Workflow integration:** we are building a CPU-supported integration layer for coding assistants to improve applicability and usability. Users will be able to regenerate/incrementally update a repo's adapter offline at commit/push time, then serve with zero added inference tokens, identical to the bare backbone.




## Weakness 3 + Suggestion 3: Concerns about scalability to real-world codebases

We ran the requested ablation, binning every test instance by total repository source size (cells are EM, CR / IR; CR n=6,414 / 52 repos, IR n=5,222 / 409 repos):

| repo size (≈tok) | Pretrained | DRC | FFT | sLoRA | C2L-Static |
|---|---:|---:|---:|---:|---:|
| <25K | 36.7 / 44.1 | 38.8 / 48.3 | 51.3 / 51.5 | 47.0 / 46.9 | **57.9 / 61.3** |
| 25–75K | 43.5 / 45.6 | 46.2 / 48.7 | 45.8 / 54.2 | 41.2 / 49.4 | **58.5 / 63.4** |
| 75–150K | 47.4 / 46.7 | 49.6 / 49.0 | 52.0 / 55.4 | 47.7 / 49.9 | **65.8 / 66.6** |
| 150–300K | 44.3 / 50.0 | 45.6 / 53.3 | 52.6 / 61.4 | 49.2 / 54.5 | **62.8 / 72.2** |
| >300K | 59.7 / 47.1 | 66.9 / 47.9 | 65.1 / 55.8 | 63.4 / 50.4 | **82.9 / 66.9** |

Code2LoRA EM dips in the largest IR bin, but this is not embedding-specific: every method drops in lockstep, reflecting harder completions rather than the embedding saturating. Per-repo EM–log-size correlation is flat-to-slightly-positive for all methods.

The recurrent embedding scales too: repeating the ablation on the evolution track for Code2LoRA-Evo, it is likewise flat, including on the largest repos:

| repo size (≈tok) | CR: QnAs / Evo | IR: QnAs / Evo |
|---|---:|---:|
| <25K | 4,265 / **64.7** | 3,518 / **63.3** |
| 25–75K | 13,639 / **56.5** | 12,552 / **64.7** |
| 75–150K | 13,621 / **64.1** | 10,612 / **64.8** |
| 150–300K | 11,970 / **57.4** | 8,903 / **65.5** |
| >300K | 1,237 / **73.3** | 6,476 / **62.8** |

Per-repo EM vs. log10(size) is again flat (Pearson +0.105 CR, −0.046 IR) as well. These repositories are large, with source size median 165K, mean 285K, p95 ≈1.03M, p99 ≈1.87M, max ≈3.0M tokens. The ">300K" bin already reaches well past a single context window. Current pooled vectors may eventually limit gigantic repositories and we flag hierarchical/graph encoders as future work, but at the scales tested they degrade neither absolutely nor relative to any baseline. We will report both tables in the paper.

## Weakness 4 + Suggestion 4: Potential memorisation-driven overfitting in the hypernetwork

Memorization is controlled directly by our evaluation design, which spans three regimes of decreasing training exposure:

* **In-repo (IR):** the repository was seen in training but the evaluated completions are held out — measures use of exposed repository knowledge on unseen instances.

* **Cross-repo (CR):** the 52 test repositories are entirely absent from training (headline 63.8% EM); the hypernetwork sees each at test for the first time, so beating every baseline here cannot be repository memorization.

* **Held-out OOD:** repositories created after the training cutoff, provably in neither our training set nor the backbone's pretraining — here Code2LoRA-Evo is strongest.

**Leakage analysis:** we collect the 20,861 unique training answer strings (assertion RHSs) and label each CR-test instance by whether its answer was seen or never seen in training (impossible to memorize); 50.5% are never-seen. Per method:

| CR-test (n=6,414) | EM on never-seen answers | EM on seen answers | of *correct* preds, % whose answer was seen |
|---|---:|---:|---:|
| Pretrained (never trained on our data - control) | 32.8 | 58.9 | **63.7** |
| FFT | 42.2 | 60.9 | 58.6 |
| sLoRA | 39.3 | 55.8 | 58.2 |
| **Code2LoRA-Static** | **50.5** | **77.4** | **60.0** |

Two things follow. **(i)** On answers it could not have copied (never in training), Code2LoRA scores 50.5%, above FFT (42.2) and pretrained (32.8) — it generalizes to unseen answers. **(ii)** The third column is a memorization probe: of the predictions each method gets right, how many had a training-seen answer? Pretrained is the control — never trained on our data, so its 63.7% is the overlap expected purely from recurring common values (True, 0, None, …). Code2LoRA's correct answers are only 60.0% seen — *below* the control, whereas a memorizer would sit above it. (IR shows the same pattern, but there the overlap is ≈90% for every method since IR repositories were in training.) We will add this analysis to the appendix.


## Weakness 5: Potential bias in the benchmark toward the proposed method

Releasing each repository whole gives retrieval more to work with, not less, nothing is hidden from RAG/DRC. What limits retrieval is its own selection step, not information the benchmark withholds.

The residual gap is a property of the methods, not the benchmark: on instances that require repository knowledge, retrieval recovers only ≈10% vs. ≈38–42% for parametric adaptation — both see the same whole repository, but a few chunks cannot surface globally-scattered facts, exactly the distinction the benchmark measures. The task is also externally motivated (assertion completion from LiveCodeBench (Jain et al. 2024), not tailored to us), with target-type composition matched across splits within ≈2 pp. We state the scope as one task, one language, one backbone, within which the benchmark treats retrieval fairly.
