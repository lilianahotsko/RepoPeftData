Reply 1: to reviewer v1i3

# start here

Thank you for the careful review, and for recognizing that Code2LoRA-Evo (a recurrent hypernetwork over sequential commit diffs, with no analogue in the Text2LoRA/Doc2LoRA line) is new, and that releasing each repository *whole*—with commit history—is a real step over retrieval-sliced benchmarks. We answer each point below, treating W1 first and in most depth. All "new" experiments are complete and their numbers reported here.

## (W1) The context baselines may be broken rather than merely weaker

> RAG falls below the no-context pretrained model on every track... Without a properly formatted, FIM/repo-aware in-context baseline that improves over pretrained, the parametric-vs-context conclusion is not established.

You are right, and the diagnosis was exactly correct. Prepending context made Qwen2.5-Coder treat the prompt as a fill-in-the-middle split and emit its own control tokens (`<|fim_suffix|>`/`<|file_sep|>`) instead of continuing the code. We rebuilt the in-context baseline in the backbone's **native repo-completion format** (`<|repo_name|>`/`<|file_sep|>`) and **banned those sentinels at decode time**, holding retriever, index, k=3, and window fixed. Corrected EM (paper → fixed):


| Track (CR / IR) | Pretrained  | RAG: paper → fixed            | DRC: paper → fixed            |
| --------------- | ----------- | ----------------------------- | ----------------------------- |
| Static          | 45.7 / 46.8 | 39.7 / 42.1 → **47.6 / 48.0** | 48.2 / 49.5                   |
| Evolution       | 31.5 / 29.3 | 23.6 / 23.0 → **36.8 / 37.4** | 31.1 / 31.6 → **36.4 / 36.9** |
| OOD             | 44.6        | 32.6 → **55.6**               | 45.5 → **56.7**               |


**Retrieval now beats pretrained on every track**; the earlier shortfall was a formatting/decoding artifact, as you suspected. In the interest of full disclosure this corrects our own reporting on the evolution/OOD tracks, where the submitted RAG used a truncated prefix and no sentinel ban; the decode-time ban is the dominant driver (e.g. evolution +9.6 / +11.8 pp), the native layout adds the rest. (OOD repos have only a generic, non-AST chunk cache, so their lift comes from the un-truncated prefix + ban.)

**The fix is fair—one uniform generation config.** It touches only the context methods (RAG/DRC); every other method already receives just the raw prefix. We verified the shared decode-ban is immaterial for them: re-run with the ban on, **sLoRA is bit-identical (57.44 → 57.44 EM)** and pretrained shifts **−0.13 pp (33.28 → 33.15)**—within noise, and if anything lower. So the correction only *raises* the context baselines, making the comparison strictly more conservative.

**The conclusion holds against the corrected baselines.** Code2LoRA-Static (**63.8 / 66.2**) still leads the best in-context baseline by **+16.2 / +18.2 pp** (repo-FIM RAG) and **+15.6 / +16.7 pp** (DRC). It never rested on RAG being broken. We will add the repo-FIM baseline to Tables 1–2, report the corrected RAG/DRC rows, and note the configuration change.

## (W2) Benchmark validity: the task is local and targets are short

> Benchmark validity is unclear because the task is very local and the targets are very short.

**Fraction solvable locally.** Taking "the no-context pretrained model solves it" as the definition, **45.7% / 46.8%** (CR / IR) is locally solvable; the majority is not. On the **repository-knowledge-required subset** (where the local-prefix model fails), Code2LoRA-Static recovers **38.1 / 41.5%**, vs. FFT 33.7 / 40.4, sLoRA 32.1 / 36.5—but even the *corrected* context baselines (W1) recover only ≈10% (repo-FIM RAG 10.1 / 9.7, DRC 9.1 / 9.1). Tellingly, the entire benefit of the RAG fix lands on the *locally-solvable* instances (77 → 92% CR), i.e. it restores local competence the naive prompt was destroying—not repository knowledge. Parametric adaptation is what recovers the instances that genuinely need the repository (details in Suggestion 2).

**Short ≠ trivial.** Targets are short (median ≈3 tokens, inherent to assertion-RHS completion, as in LiveCodeBench), yet (i) the base model misses over half; (ii) by target-length bin, Code2LoRA-Static leads in *every* bin including the longest (≥20 chars: 24.7 vs. 20.7 FFT, CR); (iii) target-type composition matches across train/CR/IR within ≈2 pp, so gains are generalization, not reweighting.

**Exact "relaxed" EM rule.** The only relaxation is truncating overgeneration to the target's natural boundary (first line / first token / pre-comma)—standard practice for left-to-right code completion, applied identically to all methods, with no fuzzy or aliased matching (`evaluation/metrics.py`; full rule in Suggestion 2). Importantly, the method ranking and every headline gap are the same under **strict** exact-match, and Code2LoRA's lead is at least as large—so no conclusion depends on the rule. We will state the rule verbatim and report both metrics.

## (W3) Headline results are oversold relative to statistical support

> The headline results are oversold relative to their statistical support.

We add uncertainty quantification and paired tests (infrastructure already in `metrics.py`). On the large per-instance samples (n=6,414 CR / 5,222 IR) the gaps are far outside sampling error: Code2LoRA-Static vs. FFT **+12.4 pp**, 95% CI **[+11.4, +13.4]**; vs. DRC +15.6; vs. sLoRA +16.4; vs. pretrained +18.1. Every headline comparison is **McNemar p < 1e-12**, paired-bootstrap P(Δ>0)=1.000, and remains significant under a conservative **repo-clustered bootstrap**. We will attach 95% CIs to every cell of Tables 1–2 and report these tests. We also agree OOD was over-emphasized (its shorter targets inflate EM): we will soften the abstract to the matched-input ≈1.8 pp figure and present OOD as a robustness check, foregrounding the static (+12 pp) and evolution (+5 pp) tracks.

## (Suggestion 1) A context-injection baseline that improves over pretrained

See W1: the repo/FIM-formatted retrieval baseline now beats pretrained (**47.6 / 48.0** vs. 45.7 / 46.8, up from 39.7 / 42.1), and DRC already did (48.2 / 49.5)—so the parametric-vs-context comparison is fair, and Code2LoRA-Static still leads by +16–18 pp.

## (Suggestion 2) Locally-solvable fraction, the repo-knowledge gap, and the exact EM rule

**Fraction solvable from the prefix alone.** Taking "the no-context pretrained model solves it" as the definition, **45.7% (CR) / 46.8% (IR)** is locally solvable; the majority (≈54% / ≈53%) is not—so the benchmark is not dominated by trivially local completions.

Splitting by whether the no-context pretrained model solves the item, with the **W1-corrected** context baselines (EM %, CR / IR):

**(a) Repository-knowledge-required subset** (pretrained fails — needs the wider repo):


| Method               | CR       | IR       |
| -------------------- | -------- | -------- |
| **Code2LoRA-Static** | **38.1** | **41.5** |
| FFT                  | 33.7     | 40.4     |
| sLoRA                | 32.1     | 36.5     |
| RAG (corrected)      | 10.1     | 9.7      |
| DRC                  | 9.1      | 9.1      |
| Pretrained           | 0.0      | 0.0      |


**(b) Locally-solvable subset** (pretrained already solves it — no repo knowledge needed):


| Method               | CR    | IR    |
| -------------------- | ----- | ----- |
| Pretrained           | 100.0 | 100.0 |
| DRC                  | 94.7  | 95.4  |
| **Code2LoRA-Static** | 94.3  | 94.3  |
| RAG (corrected)      | 92.2  | 91.5  |
| FFT                  | 72.5  | 73.5  |
| sLoRA                | 65.6  | 66.2  |


Two facts stand out. On (a), even after the W1 fix context injection recovers only **≈10%** of the instances that need repository knowledge, whereas parametric adaptation recovers **≈38–42%**—the corrected RAG's ≈8 pp overall gain lands almost entirely on subset (b) (77.4 → 92.2 CR; 80.4 → 91.5 IR), i.e. it restores local competence the naive prompt was destroying, not repository knowledge. On (b), Code2LoRA and the (corrected) context baselines stay near-ceiling, but the fine-tuned baselines **regress sharply** (sLoRA loses ≈34 pp): Code2LoRA recovers repository-knowledge instances *without* eroding local competence, unlike fine-tuning. The advantage is concentrated exactly where repository knowledge is required. (DRC values are its published no-ban numbers; a DRC+ban rerun is underway and, since DRC did not exhibit the degeneracy—its locally-solvable EM is already ≈95%—is expected to be essentially unchanged.)

**Exact "relaxed" EM rule.** The only relaxation is trimming overgeneration before comparison—standard practice for left-to-right code completion, with no fuzzy or aliased matching, applied identically to every method (`evaluation/metrics.py`): we strip Qwen FIM sentinels, drop trailing comments, and keep the leading segment of the generation at its natural boundary (first line; and, matching the short assertion targets, up to the first comma or first token). Both strings then receive identical normalization (strip surrounding/trailing whitespace and `{ : , }`, collapse internal whitespace) and must be exactly equal for `EM=1`. Since only the *start* of the generation is kept, a method is credited only when it produces the answer first—never when a correct answer is buried after distractor text.

**This rule cannot flatter our results; it only helps the baselines.** We verified with a **strict exact-match** variant (exact equality, no truncation): Code2LoRA-Static is **identical** under both metrics (63.8 / 66.2, CR / IR) because its adapter outputs terminate cleanly at the target—so none of our correct answers come from truncated overgeneration. The fine-tuned baselines, which overgenerate past the short target, are the ones the relaxation rescues (FFT 51.4 → 14.8, sLoRA 47.4 → 7.5 under strict EM). Thus the relaxation, if anything, *narrows* our lead; the method ranking and every headline gap are unchanged and Code2LoRA's margin is at least as large under strict scoring. We will state the rule verbatim and report both metrics in the revision.

# end here

Reply 2: to reviewer yeRc

# start here

Thank you for the constructive review and for recognizing RepoPeftBench and the parametric-injection paradigm. Your concerns center on *overstated wording* and *fairness/scalability* rather than correctness; below we tighten the claims and report new analyses (all complete, not promised). We group each weakness with its matching suggestion.

## (W1 + W6 + Suggestion 1) "Zero inference cost" is overstated; real-world latency/storage

> Generating embeddings and LoRA weights requires preprocessing and hypernetwork inference... Code2LoRA-Evo requires per-commit updates... this contradicts "zero overhead."

You are right the wording was loose; the precise, defensible claim is **zero inference-time *token* overhead**, not zero compute. At query time the merged-adapter model reads only the test prefix (**0 extra tokens**), whereas RAG re-encodes ≈1,536 and DRC ≈4,096 context tokens on *every* query. Adapter generation is a **one-time, per-repo-state** cost amortized over the thousands of queries a repository receives; measured end-to-end on a single **H100**: full cold build **34–257 s** (188–1,073 commits), **incremental** per-commit update **≈21–23 s** (cache hit near-instant), and even on **CPU** (8 threads) it is only **344–1,079 s** cold / **30–297 s** incremental. The step is CPU/parse-bound (GPU util ≈0% during the walk), so it needs no GPU to deploy. Crucially the Evo per-commit update is offline work done *at commit time* (one diff-embed + one recurrent step), not a per-inference cost—so there is no "continuous computation during usage." On storage, the adapter Code2LoRA generates is the *same* object as a per-repo LoRA (18.5M params, ≈37 MB bf16), so the advantage is not a smaller adapter but that it need not be **stored**: any repo's adapter is regenerated in <10 ms from the shared hypernetwork (712M params, **≈1.4 GB** bf16, *independent of #repos*) plus a **≈8 KB** per-repo state (the repo embedding / GRU hidden state, 2048 floats). Per-repo LoRA has no generator, so it must persist **≈37 MB × N** of weights (≈37 GB for 1,000 repos; our 405 trained adapters already occupy ≈30 GB) and cannot produce an adapter for held-out CR repos at all. The irreducible per-repo footprint is thus **≈8 KB vs. 37 MB** (~4,600×), with the generator amortized across all repos. (We are also correcting the efficiency table, whose "+679 MB / +65 MB" entries mislabeled the head's 679M *parameter* count and omitted the shared head from the Evo row; the qualitative conclusion is unchanged.) Deployment is not hypothetical: a working `c2l` CLI (`c2l_terminal`), a HF Space, and a companion demo. We will reword the abstract to "zero inference-time token overhead" and add this latency/amortization/storage paragraph.

## (W2 + Suggestion 2) Biased/limited baselines

> RAG uses limited depth (k=3)... no long-context or hybrid comparison... per-repo LoRA as upper bound only in IR advantages the method.

(a) **k=3 is the strongest setting, not a strawman:** the appendix k-sweep is monotonically *worse* for larger k (CR EM 39.7 → 37.5 → 36.3 for k=3,5,10), as extra chunks distract from the completion boundary—so we report RAG's best configuration. (b) **Compute is matched:** all methods share the same frozen backbone (Qwen2.5-Coder-1.5B), isolating the adaptation mechanism rather than model scale. (c) **Stronger and hybrid context baselines (new):** the properly repo/FIM-formatted retrieval baseline now beats pretrained (**47.6 / 48.0**) yet stays ≈16–18 pp below Code2LoRA-Static; DRC is itself a retrieval+compression hybrid that also beats pretrained (48.2 / 49.5); and a trained+retrieval hybrid helps too (**FFT+RAG 53.9 / 56.8 > FFT 51.4 / 55.9**). All remain well below Code2LoRA-Static—parametric adaptation wins even against fair, pretrained-beating, and hybrid context baselines. (d) **Per-repo LoRA's IR-only role is definitional, not a bias:** it requires per-repo training instances, which exist only for IR; CR repos are fully held out, so a per-repo adapter there would train on the eval data. And Code2LoRA-Static *matches* this upper bound in IR (66.2 vs. 64.0) with no per-repo training—the opposite of being flattered.

## (W3 + Suggestion 3) Scalability of a fixed-size repository embedding

> A single fixed-size embedding may not capture million-token repositories; mean/max pooling loses fine-grained dependencies.

We ran the requested ablation: binning every test instance by total repository source size and reporting EM per bin. The concern is really *relative*—does the fixed embedding degrade **compared to alternatives** as repos grow? It does not: **Code2LoRA-Static leads in every bin on both splits, and its margin over the best baseline is maintained or grows on the largest repositories** (over FFT: CR +17.8 pp at >300K vs. +6–14 pp elsewhere; IR ≈+11 pp at both 150–300K and >300K).

EM by repository size (CR-test, n=6,414 / 52 repos):


| repo size (≈tok) | inst | repos | Pretrained | DRC  | FFT  | sLoRA | **C2L-Static** |
| ---------------- | ---- | ----- | ---------- | ---- | ---- | ----- | -------------- |
| <25K             | 534  | 10    | 36.7       | 38.8 | 51.3 | 47.0  | **57.9**       |
| 25–75K           | 1720 | 16    | 43.5       | 46.2 | 45.8 | 41.2  | **58.5**       |
| 75–150K          | 2293 | 15    | 47.4       | 49.6 | 52.0 | 47.7  | **65.8**       |
| 150–300K         | 1383 | 8     | 44.3       | 45.6 | 52.6 | 49.2  | **62.8**       |
| >300K            | 484  | 3     | 59.7       | 66.9 | 65.1 | 63.4  | **82.9**       |


EM by repository size (IR-test, n=5,222 / 409 repos):


| repo size (≈tok) | inst | repos | Pretrained | DRC  | FFT  | sLoRA | **C2L-Static** |
| ---------------- | ---- | ----- | ---------- | ---- | ---- | ----- | -------------- |
| <25K             | 524  | 66    | 44.1       | 48.3 | 51.5 | 46.9  | **61.3**       |
| 25–75K           | 1551 | 137   | 45.6       | 48.7 | 54.2 | 49.4  | **63.4**       |
| 75–150K          | 1356 | 94    | 46.7       | 49.0 | 55.4 | 49.9  | **66.6**       |
| 150–300K         | 1021 | 66    | 50.0       | 53.3 | 61.4 | 54.5  | **72.2**       |
| >300K            | 770  | 46    | 47.1       | 47.9 | 55.8 | 50.4  | **66.9**       |


We do note that *absolute* EM dips in the largest IR bin (Code2LoRA 72.2 → 66.9). Crucially this is **not specific to Code2LoRA or its embedding**: every method drops in lockstep there (Pretrained 50.0 → 47.1, DRC 53.3 → 47.9, FFT 61.4 → 55.8, sLoRA 54.5 → 50.4), so it reflects intrinsically harder completions in those repositories rather than the pooled embedding saturating—Code2LoRA's lead in that bin is in fact the widest on IR (+11.1 pp over FFT). Consistently, the per-repo EM–log-size correlation is **flat-to-slightly-positive** for all methods (Code2LoRA **+0.24 CR, +0.08 IR**), i.e. no systematic decline with size. Three further checks agree that pooling is not currently lossy in a harmful way: the internalization probe saturates fast (49.6 → 67.9 EM after a single repo file); a **45× larger encoder does not improve** decoded EM at fixed hypernetwork capacity (0.591 vs. 0.603 CR); and the 52 CR-test adapters are genuinely distinct (pairwise cosine 0.01 ± 0.94), not a collapsed mean. (The CR >300K bin is only 3 repos, so we rely on the "leads in *every* bin on both splits" result rather than that single bin; IR >300K is 46 repos.) We agree a single pooled vector may eventually limit truly gigantic repositories—and flag hierarchical/graph encoders as future work—but at the scales tested it degrades neither absolutely (the correlation is non-negative) nor relative to any baseline. We will add both tables.

## (W4 + Suggestion 4) Memorization in the ≈700M hypernetwork

> The large hypernetwork may memorize patterns; the paper does not rule this out.

Memorization is controlled directly by our **evaluation design**, which reports every method across three regimes of decreasing training exposure:

- **In-repo (IR):** the repository *was* seen in training, but the evaluated completions are held out—this measures use of exposed repository knowledge on unseen instances.
- **Cross-repo (CR):** the 52 test repositories are **entirely absent from training** (headline **63.8%** EM). Since the hypernetwork encounters each repository for the first time at test, outperforming every baseline here cannot be repository memorization.
- **Held-out OOD:** repositories created **after the training-data cutoff**, so provably in neither our training set nor the backbone's pretraining—here **Code2LoRA-Evo is strongest**.

A direct **leakage analysis** confirms this. *Setup:* we collect the **20,861** unique answer strings (assertion right-hand sides) that appear anywhere in the training split, and label each CR-test instance by whether its answer string was **seen** in training or **never seen** (impossible to have memorized); **50.5%** of CR-test answers are never-seen. We then read three quantities per method:


| CR-test (n=6,414)                                | EM on never-seen answers | EM on seen answers | of *correct* preds, % whose answer was seen |
| ------------------------------------------------ | ------------------------ | ------------------ | ------------------------------------------- |
| Pretrained (never trained on our data — control) | 32.8                     | 58.9               | **63.7**                                    |
| FFT                                              | 42.2                     | 60.9               | 58.6                                        |
| sLoRA                                            | 39.3                     | 55.8               | 58.2                                        |
| **Code2LoRA-Static**                             | **50.5**                 | **77.4**           | **60.0**                                    |


Two things follow. (i) On answers it **could not have copied** (never in training), Code2LoRA scores **50.5%**, above FFT (42.2) and pretrained (32.8)—it generalizes to unseen answer strings. (ii) The third column is a memorization probe: of the questions each method gets *right*, how many had a training-seen answer? The **pretrained** model is the control—it never trained on our data, so its **63.7%** is the overlap expected purely because common values (`True`, `0`, `None`, …) recur. Code2LoRA's correct answers are **60.0%** seen—*below* that control—so it is not disproportionately winning on memorizable answers; a memorizer would sit *above* the control, not below. (IR shows the same pattern; there the overlap is ≈90% for *every* method because IR repositories were in training, so no method stands out.) Combined with the 10-repo result (≈2% of data already beats FFT-on-all, 57.7 vs. 51.4), the evidence points to adaptation, not memorization. We will add this analysis and the perturbation (mismatched-adapter) test to the appendix.

## (W5) Benchmark bias toward the proposed method

> Full-repository access and assertion-completion favor parametric methods and disadvantage retrieval.

Releasing each repository **whole** gives retrieval *more* to work with, not less—**nothing is hidden from RAG/DRC**. What limits retrieval is its own selection step, not information the benchmark withholds, and the evidence bears this out: if the task were starving retrieval, better *formatting* could not help, yet once formatted correctly both RAG and DRC **beat the pretrained backbone** (repo-FIM RAG **47.6 / 48.0**; DRC **48.2 / 49.5**, CR / IR). The earlier gap was a formatting artifact, not missing information.

The residual gap is thus a property of the *methods*, not the benchmark: on instances that genuinely require repository knowledge, corrected retrieval still recovers only **≈10%** vs. **≈38–42%** for parametric adaptation—both see the same whole repository, but a few chunks cannot surface globally-scattered facts, which is exactly the distinction the benchmark measures. The task is also externally motivated (assertion completion from LiveCodeBench, not tailored to us), with target-type composition matched across splits within **≈2 pp**. We state the scope honestly (one task, one language, one backbone), but within it the benchmark treats retrieval fairly—retrieval simply meets a ceiling parametric adaptation does not.

# end here

