# Response to Reviewers — Code2LoRA (Submission 9808), Second Resubmission

We thank the reviewers and the action editor for the careful and constructive feedback on the previous round. Below we summarize, for each reviewer, the changes that were **proposed** and how we have **addressed** them in this resubmission. All changes are reflected in the revised manuscript; new numbers supersede the corresponding rows/claims in the submitted version.

## Summary of changes in this revision

- **Corrected and strengthened the context-injection (RAG/DRC) baselines** by fixing a decoding/format artifact; all in-context baselines now improve over the pretrained model, and the parametric-vs-context conclusion holds against these stronger baselines.
- **Added statistical significance testing** (paired bootstrap, 95%) for the headline comparisons and softened over-broad claims.
- **Added a repository-knowledge-required vs. locally-solvable analysis** and a precise definition of the "relaxed" EM rule.
- **Added repository-size ablations** for both Code2LoRA-Static and Code2LoRA-Evo.
- **Added a leakage / memorization analysis** on seen vs. never-seen answers.
- **Clarified the "zero inference cost" claim** and added end-to-end latency, FLOPs, and storage analyses.
- **Narrowed the scope of the central claim** to the evaluated setting.

---

## Reviewer v1i3

### Proposed: The context (RAG) baseline is broken — it falls below the no-context pretrained model on every track; provide a properly formatted, FIM/repo-aware in-context baseline that at least improves over pretrained.

**Addressed.** We traced the failure to a decoding/format artifact: under naive context prepending, Qwen2.5-Coder degenerates into emitting its own control tokens (`<|fim_suffix|>`, `<|file_sep|>`, `<|repo_name|>`) instead of continuing the code — a formatting issue, not evidence that context is inferior. Holding the retriever, index, k=3, and context window fixed, we rebuilt the in-context baseline using the backbone's native repository-completion layout and a decode-time ban on the control tokens. The corrected baselines now beat the pretrained model on every track:

- Static (CR / IR): RAG 39.7 / 42.1 → **47.6 / 48.0** (pretrained 45.7 / 46.8).
- Evolution (CR / IR): RAG 23.6 / 23.0 → **36.8 / 37.4** (pretrained 31.5 / 29.3).
- The same fix applied to DRC: evolution 31.1 / 31.6 → 36.4 / 36.9; OOD 45.5 → 56.7.

Even against these corrected, pretrained-beating baselines the conclusion is unchanged: Code2LoRA-Static (63.8 / 66.2) leads on the static track and Code2LoRA-Evo (60.3 / 64.5) leads on the evolution track. We have (i) replaced the RAG/DRC rows with the corrected numbers, (ii) noted the configuration change from the submitted version, and (iii) described the native-format/decoding fix in the evaluation setup.

### Proposed: Benchmark validity is unclear because the task is very local and the targets are very short.

**Addressed.** Although targets are short (median ≈3 tokens, inherent to assertion-RHS completion), the task is non-trivial: it requires extracting repository-level context and reasoning about code execution. We added a breakdown showing the base model collapses on the harder-but-short cases (static CR / IR): complex expressions 31.3 / 36.4%, function calls 20.4 / 22.1%, collections 17.5 / 15.1%, vs. 71.4 / 74.9% on numeric literals; and by length, 51.3 / 52.1% (<20 chars) drops to 18.3 / 16.5% (≥20 chars). Over half the instances are not locally solvable, and Code2LoRA's gains land exactly on the structured/longer targets (e.g., complex expressions 31.3 → 68.4, collections 17.5 → 42.6) while leaving already-local numeric literals unchanged (71.4 → 70.3). Target-type composition matches across train / CR / IR within ≈2 pp (Table 6), indicating generalization rather than reweighting toward easier categories.

### Proposed: The headline results are oversold relative to their statistical support.

**Addressed.** We added paired-bootstrap significance tests (95%) for all headline comparisons and softened the claims. Per-Repo LoRA is now explicitly framed as an upper bound (it trains per repository and does not generalize to CR/OOD), and comparisons are made under matched training/inference cost. Representative results: on CR (n=6,414), Code2LoRA-Static vs. DRC +14.7 pp (95% lower bound +13.8, p<1e-5); on IR (n=5,222), +16.2 pp (lower bound +15.1, p<1e-5). All reported differences are statistically significant.

### Proposed: What fraction of RepoPeftBench is solvable from the local prefix alone; how does the gap look on instances that require repository knowledge; specify the exact "relaxed" EM rule.

**Addressed.** Using "the no-context pretrained model solves it" as the definition, 45.7% (CR) / 46.8% (IR) is locally solvable; the majority (~54% / ~53%) is not. Restricted to instances the local-prefix model fails, context injection barely moves the needle (RAG ~10.1 / 9.7, DRC ~9.1 / 9.1) while parametric adaptation recovers ~38–42% (Code2LoRA-Static 38.1 / 41.5); on the locally-solvable complement all methods are near ceiling (Code2LoRA 94–96%). We also added the exact relaxed-EM rule: strip Qwen FIM sentinels, drop trailing comments, keep the leading segment at its natural boundary (first line, and up to the first comma/token), apply identical normalization to both strings (strip surrounding/trailing whitespace and `{ : , }`, collapse internal whitespace), and require exact equality; credit is given only when the answer is produced first.

---

## Reviewer yeRc

### Proposed: The "zero inference cost" claim is overstated — it shifts computation (repo embedding, hypernetwork inference, per-commit updates) rather than eliminating it; report end-to-end latency.

**Addressed.** We clarified that the claim is specifically **zero additional per-query context tokens**, not zero end-to-end computation, and revised the abstract to "zero inference-time token overhead." Adapter generation is a one-time, per-repo-state cost amortized over all queries a repository serves: measured on a single H100, full cold build 34–257 s (188–1,073 commits), incremental per-commit update ≈21–23 s; even CPU-only (8 threads) it is 344–1,079 s cold / 30–297 s incremental. The Evo per-commit update is offline work at commit time (one diff-embed + one recurrent step), not a per-inference cost, so there is no continuous computation during usage. We added this latency/storage paragraph.

### Proposed: Limited or potentially biased comparison with baselines — RAG uses shallow retrieval (k=3); no long-context or hybrid baselines; per-repo LoRA as upper bound only in IR advantages the method.

**Addressed.** We strengthened the context baselines (see v1i3 W1) and added hybrid comparisons: repo-FIM RAG (47.6 / 48.0) stays ≈16–18 pp below Code2LoRA-Static; DRC (retrieval+compression hybrid) reaches 49.1 / 50.0; a trained+retrieval hybrid (FFT+RAG 53.9 / 56.8 > FFT 51.4 / 55.9) helps but still trails Code2LoRA-Static. The Appendix C sweep over k and chunk size confirms k=3 is strongest, so more depth does not close the gap. We keep the backbone fixed across all methods to isolate the injection mechanism, and flag long-context modeling as complementary future work (parametric adaptation can sit on top of a long-context backbone). We clarified that per-repo LoRA's IR-only role is definitional (per-repo training data exists only for IR; training on CR would leak the eval set), and note that Code2LoRA-Static matches this upper bound in IR (66.2 vs. 64.0) with no per-repo training. We narrowed the central claim from "parameters are generally better than context" to "parametric adaptation outperforms the evaluated retrieve-and-prepend and dependency-resolved baselines in our setting."

### Proposed: Concerns about scalability — a single fixed-size pooled embedding may not summarize million-token repositories; validate across repository sizes; consider richer (hierarchical/graph) encoders.

**Addressed.** We added a repository-size ablation (bins from <25K to >300K tokens) for Code2LoRA-Static and repeated it on the evolution track for Code2LoRA-Evo. Code2LoRA does not degrade relative to any baseline: EM dips in the largest IR bin only because every method drops in lockstep (harder completions), and per-repo EM–log(size) correlation is flat (Evo: Pearson +0.105 CR, −0.046 IR). These repositories are large (source-size median 165K, mean 285K, p95 ≈1.03M, p99 ≈1.87M, max ≈3.0M tokens; the >300K bin already exceeds a single context window). We note pooled vectors may eventually limit gigantic repositories and flag hierarchical/graph encoders as future work, but at tested scales they degrade neither absolutely nor relative to baselines. Both tables are added to the paper.

### Proposed: Potential memorization-driven overfitting in the ~700M hypernetwork; provide stronger OOD / leakage analyses.

**Addressed.** We emphasize that memorization is controlled by the evaluation design (three regimes of decreasing exposure): IR (repo seen, completions held out), CR (52 test repos entirely absent from training; headline 63.8% EM), and held-out OOD (repos created after the training cutoff, in neither our training set nor the backbone's pretraining). We added a leakage analysis over the 20,861 unique training answer strings: 50.5% of CR-test answers are never-seen. On never-seen answers Code2LoRA-Static scores 50.5% vs. FFT 42.2 and pretrained 32.8 (it generalizes to unseen answers). A memorization probe (share of correct predictions whose answer was training-seen) puts Code2LoRA at 60.0% — below the pretrained control (63.7%), whereas a memorizer would sit above it. This analysis is added to the appendix.

### Proposed: Potential benchmark bias toward the proposed method — full-repo access with assertion-completion favors parametric methods over retrieval.

**Addressed.** Releasing each repository whole gives retrieval *more* to work with, not less — nothing is hidden from RAG/DRC. The residual gap is a property of the methods: on repository-knowledge-required instances, retrieval recovers only ≈10% vs. ≈38–42% for parametric adaptation, though both see the same whole repository — a few chunks cannot surface globally-scattered facts. The task is externally motivated (assertion completion from LiveCodeBench, Jain et al. 2024), with target-type composition matched across splits within ≈2 pp. We state the scope explicitly as one task, one language, one backbone, within which the benchmark treats retrieval fairly.

### Proposed: Limited evaluation of real-world applicability — integration, adapter-generation latency, and storage overhead for many repositories.

**Addressed.** Adapter generation is the one-time, offline, GPU-free step quantified above, amortized over the thousands of completions a repository serves. Storage scales favorably: the per-repository footprint is ≈8 KB (repo embedding / GRU state) vs. ≈37 MB × N for stored per-repo adapters, since the shared hypernetwork (≈1.4 GB, repo-independent) regenerates any adapter in one forward pass — so storage is effectively constant in the number of repositories. At query time the merged-adapter model is latency-identical to the bare backbone (197.1 vs. 198.3 ms at in=2048/out=64), whereas RAG/DRC re-encode ≈1,536 / ≈4,096 extra tokens per query (RAG +24% at out=8; DRC +48%); per-query FLOPs confirm RAG/DRC spend ≈1.7–2.3× the compute for context Code2LoRA has internalized. We also describe an in-progress, CPU-supported integration layer that regenerates/incrementally updates a repo's adapter at commit/push time and then serves with zero added inference tokens.

---

We believe these revisions directly address the reviewers' concerns and strengthen both the empirical support and the scoping of our claims. We thank the reviewers again for feedback that improved the paper.
