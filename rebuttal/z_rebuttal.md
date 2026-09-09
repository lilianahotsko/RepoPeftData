Official Review of Submission9808 by Reviewer v1i3
Official Reviewby Reviewer v1i302 Jul 2026, 04:10 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer v1i3, Commitment ReadersRevisions
Paper Summary:
The paper proposes Code2LoRA, a hypernetwork that maps a whole repository to a LoRA adapter for a frozen code LM, so repository knowledge is injected into parameters with zero inference-time token cost. Two variants: Code2LoRA-Static and Code2LoRA-Evo. The authors also build RepoPeftBench, 604 Python repositories released whole with a static and an evolution track and a temporal OOD holdout.

Summary Of Strengths:
The how/when framing is clean, and Code2LoRA-Evo is a genuinely new idea: aggregating sequential commit diffs into a recurrent hypernetwork state has no analogue in the Text2LoRA/Doc2LoRA line, and the bursty-commit motivation (Figure 2) is convincing.
RepoPeftBench releasing each repository whole, including first-parent commit history, is a real contribution over prior benchmarks that ship only retrieval-selected slices, and enables the evolution setting that most repo-level work cannot express.
Summary Of Weaknesses:
The central claim "repository knowledge is best injected parametrically rather than through context" rests on context baselines that are broken rather than merely weaker. RAG does not just underperform Code2LoRA; it falls below the no-context pretrained model on every track. A retrieval baseline that is worse than doing nothing signals a setup problem, not a fair comparison. Since the test prefix already contains the local test file, prepending retrieved chunks may simply push the completion point further away and add noise. Without a properly formatted, FIM/repo-aware in-context baseline that at least improves over pretrained, the parametric-vs-context conclusion is not established.
Benchmark validity is unclear because the task is very local and the targets are very short.
The headline results are oversold relative to their statistical support.
Comments Suggestions And Typos:
RAG performs worse than the no-context model on every track. Is the in-context baseline formatted appropriately for this backbone? Can you provide a context-injection baseline that at least improves over pretrained, so the parametric-vs-context comparison is fair?
What fraction of RepoPeftBench is solvable from the local prefix alone, and how does the Code2LoRA-vs-baseline gap look restricted to instances that genuinely require repository knowledge? Please also specify the exact "relaxed" EM rule.
Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
Add:
Seeking your reply to our rebuttal
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)14 Jul 2026, 21:56 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer v1i3, Commitment ReadersRevisions
Comment:
Dear Reviewer v1i3,

Thank you for your valuable comments on our paper. As the discussion phase is approaching its end (July 15 AoE), we would like to kindly confirm whether we have sufficiently addressed all of your concerns. If you are satisfied with our responses, we would greatly appreciate your consideration in raising the overall score accordingly.

Best regards, Authors of Submission 9808

Add:
Rebuttal by Authors
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)12 Jul 2026, 01:38 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer v1i3, Authors, Commitment ReadersRevisions
Comment:
Thank you for the careful review, and for recognizing that Code2LoRA-Evo (a recurrent hypernetwork over sequential commit diffs, with no analogue in the Text2LoRA/Doc2LoRA line) is new, and that releasing each repository whole is a real step over retrieval-sliced benchmarks.

Weakness 1 + Suggestion 1
The central claim "repository knowledge is best injected parametrically rather than through context" rests on context baselines that are broken rather than merely weaker. RAG does not just underperform Code2LoRA; it falls below the no-context pretrained model on every track. A retrieval baseline that is worse than doing nothing signals a setup problem, not a fair comparison. Since the test prefix already contains the local test file, prepending retrieved chunks may simply push the completion point further away and add noise. Without a properly formatted, FIM/repo-aware in-context baseline that at least improves over pretrained, the parametric-vs-context conclusion is not established.

RAG performs worse than the no-context model on every track. Is the in-context baseline formatted appropriately for this backbone? Can you provide a context-injection baseline that at least improves over pretrained, so the parametric-vs-context comparison is fair?

A retrieval baseline indicated a setup problem, so we investigated and have corrected it. The root cause is a naively prepended context, under which Qwen2.5-Coder degenerates into emitting its own control tokens (<|fim_suffix|>, <|file_sep|>, <|repo_name|>) instead of continuing the code. This is a formatting/decoding artifact, not evidence that context is fundamentally inferior. We rebuilt the in-context baseline, holding the retriever, index, k=3, and context window fixed, and it now beats the pretrained model on both tracks:

Static track: using the backbone's native repository-completion layout (<|repo_name|>/<|file_sep|>-delimited files) instead of raw prepending lifts RAG from 39.7 / 42.1 → 47.6 / 48.0 (CR / IR), now above pretrained (45.7 / 46.8).

Evolution (per-commit) track: here we also correct our own reporting. The submitted RAG figure (23.6 / 23.0, below pretrained's 31.5 / 29.3) came from a handicapped configuration, generic (non-AST) chunks, and no decode-time ban on the control tokens above. The updated context injection method increased the RAG baseline to 36.8/37.4 (CR/IR).

Also, we applied the same injection methods to the DRC baseline, which increased this baseline numbers as well: 31.1 → 36.4 and 31.6 → 36.9 on the evolution track (CR/IR) and 45.5 → 56.7 on the OOD set.

Even against these corrected, pretrained-beating baselines, the conclusion is unchanged and does not rest on RAG being broken: Code2LoRA-Static (63.8 / 66.2) still leads the best in-context baseline on the static track, and Code2LoRA-Evo (60.3 / 64.5) dominates on the evolution track. We will (i) replace the RAG/DRC rows with these corrected numbers, (ii) explicitly note the configuration change from the submitted version, and (iii) describe the native-format/decoding fix in the evaluation setup.

Supplementary Table 1: Corrected RAG and DRC baselines

Track	Pretrained	RAG: paper → fixed	DRC: paper → fixed	Code2LoRA
Static — CR	45.7	39.7 → 47.6	48.2 → 49.1	63.8
Static — IR	46.8	42.1 → 48.0	49.5 → 50.0	66.2
Evolution — CR	31.5	23.6 → 36.8	31.1 → 36.4	60.3
Evolution — IR	29.3	23.0 → 37.4	31.6 → 36.9	64.5
OOD	44.6	32.6 → 55.6	45.5 → 56.7	74.1
Weakness 2
Benchmark validity is unclear because the task is very local and the targets are very short.

While the target length is short(median ≈3 tokens, inherent to assertion-RHS completion), assertion completion is a non-trivial task that requires the LLM to extract repository-level context and reason about code execution (Jain et al, 2024, LiveCodeBench) before generating the final output. As shown in Table 6, the target can include complex data structures and expressions.

The base LLM without repository context collapses on the harder-but-short cases (static CR / IR):

By type: complex expressions 31.3 / 36.4%, function calls 20.4 / 22.1%, collections 17.5 / 15.1%, vs. 71.4 / 74.9% on numeric literals.

By length: 51.3 / 52.1% on targets <20 chars drops to 18.3 / 16.5% on ≥20 chars.

So over half the instances are not locally solvable, with difficulty concentrated on structured and longer targets. Code2LoRA's gains land exactly there (static CR: complex expressions 31.3 → 68.4, collections 17.5 → 42.6, variables 56.3 → 77.3), while leaving already-local numeric literals unchanged (71.4 → 70.3), and target-type composition matches across train / CR / IR within ≈2 pp (Table 6), so these are generalization, not reweighting toward easier categories.

Add:
Rebuttal by Authors
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)12 Jul 2026, 01:42 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer v1i3, Authors, Commitment ReadersRevisions
Comment:
Weakness 3
The headline results are oversold relative to their statistical support.

Thank you for the suggestion. We are adding statistical significance testing to support our results, and will revisit our manuscript and soften the claims of headline results if needed.

To clarify, the “Per-Repo LoRA” that demonstrates good performance on IR split should be considered as an upper-bound, because it incurs much more training cost by fine-tuning for each repository separately and does not generalize to CR or OOD splits where no per-repo training data is available. The primary comparisons should be done between Code2LoRA against our baselines under a similar level of training/inference cost.

Code2LoRA outperforms the strongest baseline by large margins for most of evaluation settings: on static track, Code2LoRA-Static outperforms DRC by +14.7pp (CR) / +16.2pp (IR); on evolution track, Code2LoRA-Evo outperforms Single LoRA by +5.2pp (CR) / +3.2pp (IR). The gap on the OOD split is relatively smaller, where Code2LoRA-Evo outperforms Single LoRA by +1.8pp, mainly because the OOD prediction targets are simpler (shorter) than those in the training set and thus the base model is more proficient in solving them. All differences have statistical significance confirmed using a paired bootstrap test under a confidence level of 95%.

cr_test (n=6414)
Comparison	Observed ΔEM (pp)	one-sided p	95% lower bound (pp)
Code2LoRA-Static vs FFT	+12.4	<1e-5	+11.5
Code2LoRA-Static vs DRC	+14.7	<1e-5	+13.8
Code2LoRA-Static vs sLoRA	+16.4	<1e-5	+15.5
Code2LoRA-Static vs RAG (k=3)	+24.1	<1e-5	+23.0
Code2LoRA-Static vs Pretrained	+18.1	<1e-5	+17.2
ir_test (n=5222)
Comparison	Observed ΔEM (pp)	one-sided p	95% lower bound (pp)
Code2LoRA-Static vs FFT	+10.4	<1e-5	+9.4
Code2LoRA-Static vs DRC	+16.2	<1e-5	+15.1
Code2LoRA-Static vs sLoRA	+15.8	<1e-5	+14.8
Code2LoRA-Static vs RAG (k=3)	+24.1	<1e-5	+22.9
Code2LoRA-Static vs Pretrained	+19.4	<1e-5	+18.4
Suggestion 2
What fraction of RepoPeftBench is solvable from the local prefix alone, and how does the Code2LoRA-vs-baseline gap look restricted to instances that genuinely require repository knowledge? Please also specify the exact "relaxed" EM rule.

Fraction solvable from the prefix alone. Taking "the no-context pretrained model solves it" as the definition, 45.7% (CR) / 46.8% (IR) is locally solvable; the majority (~54% / ~53%) is not, so the benchmark is not dominated by trivially local completions.

Gap on the repository-knowledge-required subset:

Restricting to the instances the local-prefix model fails:

Method	CR	IR
Code2LoRA-Static	38.1	41.5
FFT	33.7	40.4
sLoRA	32.1	36.5
RAG	10.1	9.7
DRC	9.1	9.1
Locally-solvable subset:

Method	CR	IR
DRC	94.7	95.4
Code2LoRA-Static	94.3	94.3
RAG (corrected)	92.2	91.5
FFT	72.5	73.5
sLoRA	65.6	66.2
Context injection barely moves these instances (~9–10%), and parametric adaptation recovers ~38–42%; on the locally-solvable complement all methods are near-ceiling (Code2LoRA 94–96%), therefore, the advantage is concentrated exactly on the instances that require repository knowledge.

Exact "relaxed" EM rule:

We strip Qwen FIM sentinels, drop trailing comments, and keep the leading segment of the generation at its natural boundary (first line; and, matching the short assertion targets, up to the first comma or first token). Both strings then receive identical normalization (strip surrounding/trailing whitespace and { : , }, collapse internal whitespace) and must be exactly equal for EM=1. Since only the start of the generation is kept, a method is credited only when it produces the answer first, never when a correct answer is generated after other code.

Add:
Official Review of Submission9808 by Reviewer yeRc
Official Reviewby Reviewer yeRc23 Jun 2026, 08:32 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer yeRc, Commitment ReadersRevisions
Paper Summary:
This paper introduces Code2LoRA, a hypernetwork-based framework for generating repository-specific LoRA adapters for code language models. Its key objective is to efficiently inject repository-level knowledge into code LLMs without increasing inference-time token costs. The framework proposes two variants: Code2LoRA-Static and Code2LoRA-Evo. The former generates adapters from a single repository snapshot, while the latter employs a GRU-based recurrent mechanism to incrementally update adapters based on code diffs, enabling adaptation to evolving repositories. In addition, the authors introduce RepoPeftBench, a benchmark comprising 604 Python repositories with both static and evolution-based tasks. Experimental results show that the approach matches or surpasses per-repository LoRA methods while eliminating the need for retraining and outperforming context-injection approaches such as RAG.

Comments after Author Discussion I thank the authors for the rebuttal. I have read all other reviewers' comments and authors' responses. However, my main concerns are not resolved. Please see below for overall summary after rebuttal:

The core concern remains that the paper draws broad methodological conclusions from a narrow experimental setting, as the evaluation is limited to a single small backbone, one programming language, and one assertion-completion task, while still concluding that repository knowledge is generally “best injected parametrically.” Although corrected retrieval baselines improve the comparison, they also reveal that implementation choices substantially affect RAG performance, leaving open the possibility that stronger retrieval systems could narrow or reverse the observed advantage. Similarly, the repository-size ablation shows no degradation within the tested benchmark but does not demonstrate that the compressed fixed-size representation preserves the fine-grained structural information required for reasoning over large industrial repositories. The leakage analysis weakens simple verbatim memorisation hypotheses but does not exclude broader repository-pattern or template memorisation, and stronger out-of-distribution evaluations would be needed to isolate genuine adaptation. Finally, practical utility is supported mainly by offline metrics and cost analyses rather than realistic developer-facing or execution-based tasks, leaving functional effectiveness insufficiently demonstrated.

Summary Of Strengths:
Potential valuable benchmark contribution The introduction of RepoPeftBench is a meaningful contribution, offering a large-scale, repository-level benchmark with both static and evolution settings. It provides a useful platform for evaluating parameter-efficient adaptation methods at the repository level, an area that remains relatively underexplored.

Novel and interesting idea The paper presents a compelling paradigm shift by injecting repository-level knowledge into model parameters via hypernetworks, rather than relying on input context or per-repository fine-tuning. This is a creative and impactful direction that addresses key limitations of both RAG-based approaches and traditional LoRA methods.

Summary Of Weaknesses:
The "zero inference cost" claim is overstated While the paper emphasizes zero inference-time token overhead, it effectively shifts computation rather than eliminating it. Generating repository embeddings and LoRA weights requires substantial preprocessing and hypernetwork inference. In realistic scenarios—such as frequent updates or large numbers of repositories—this introduces non-trivial latency and computational overhead. Furthermore, Code2LoRA-Evo requires per-commit updates, which entails continuous computation during usage. This contradicts the claimed notion of "zero overhead" and weakens a central argument of the paper.

Limited or potentially biased comparison with strong baselines The comparisons with context-injection methods (e.g., RAG) and LoRA baselines are not fully convincing. The evaluation appears biased toward parameter-based approaches, making the reported gains less compelling. In particular, RAG is implemented with a limited retrieval depth (k=3), which is not representative of stronger modern configurations. Additionally, the paper does not compare against state-of-the-art long-context LLMs or hybrid approaches (e.g., retrieval combined with routing or caching). Finally, treating per-repository LoRA as an upper bound only in the in-repo (IR) setting implicitly advantages the proposed method in cross-repo evaluations.

Concerns about scalability to real-world codebases The approach assumes that a single fixed-size embedding can faithfully summarize an entire repository and serve as the basis for generating all LoRA weights. However, real-world repositories often span millions of tokens and exhibit complex, heterogeneous structures. The use of mean-and-max pooled embeddings is likely to lose fine-grained dependencies and semantic relationships. This raises a fundamental concern: can such a compressed representation adequately capture repository-level semantics at scale? The paper does not provide sufficient evidence to support this assumption.

Potential memorisation-driven overfitting in the hypernetwork Although the method avoids explicit per-repository fine-tuning, it introduces a very large hypernetwork (~700M parameters). This creates the possibility that the model memorizes patterns across the training repositories rather than learning robust adaptation mechanisms. Given that training is conducted on hundreds of repositories and evaluation includes in-distribution data, the reported gains may reflect memorisation effects. The paper does not adequately rule out this alternative explanation, for example through stronger out-of-distribution or adversarial evaluations.

Potential bias in the benchmark toward the proposed method RepoPeftBench provides full repository access and uses assertion-completion tasks that rely on shared repository context. This setup inherently favors methods that encode global repository information into parameters, while disadvantaging retrieval-based approaches that operate on selectively retrieved context. As a result, the benchmark may implicitly bias results toward the design philosophy of Code2LoRA, limiting the generality of the conclusions.

Limited evaluation of real-world applicability and usability The paper does not evaluate how the method performs in realistic deployment scenarios. Key aspects such as integration into developer workflows, latency of adapter generation, and memory or storage overhead for large numbers of repositories are not addressed. Without this analysis, it remains unclear whether the approach is practically scalable, despite its strong performance in offline benchmarks.

Comments Suggestions And Typos:
The authors should clarify the "zero inference cost" claim by explicitly accounting for preprocessing, hypernetwork inference, and update costs. Reporting end-to-end latency comparisons with strong baselines (e.g., RAG) would make the efficiency claims more realistic and convincing.

The evaluation should include stronger and more comprehensive baselines, such as improved RAG configurations, long-context LLMs, and hybrid retrieval–adaptation methods. Ensuring fair comparisons under similar compute budgets and clarifying the “upper bound” role of per-repo LoRA would improve credibility.

The scalability of the repository encoding should be validated on larger and more complex repositories. Exploring richer encoding strategies (e.g., hierarchical or graph-based) and providing ablations across repository sizes would strengthen confidence in the approach.

The paper should better address potential memorisation in the large hypernetwork. Stronger out-of-distribution, domain-shift, or adversarial evaluations—along with perturbation or leakage analyses—would help demonstrate true generalisation rather than memorisation.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 2.5
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
Add:
Seeking your reply to our rebuttal
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)14 Jul 2026, 21:56 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer yeRc, Commitment ReadersRevisions
Comment:
Dear Reviewer yeRc,

Thank you for your valuable comments on our paper. As the discussion phase is approaching its end (July 15 AoE), we would like to kindly confirm whether we have sufficiently addressed all of your concerns. If you are satisfied with our responses, we would greatly appreciate your consideration in raising the overall score accordingly.

Best regards, Authors of Submission 9808

Add:
Rebuttal by Authors
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)12 Jul 2026, 03:34 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer yeRc, Authors, Commitment ReadersRevisions
Comment:
Weakness 3 + Suggestion 3: Concerns about scalability to real-world codebases
We ran the requested ablation, binning every test instance by total repository source size (cells are EM, CR / IR; CR n=6,414 / 52 repos, IR n=5,222 / 409 repos):

repo size (≈tok)	Pretrained	DRC	FFT	sLoRA	C2L-Static
<25K	36.7 / 44.1	38.8 / 48.3	51.3 / 51.5	47.0 / 46.9	57.9 / 61.3
25–75K	43.5 / 45.6	46.2 / 48.7	45.8 / 54.2	41.2 / 49.4	58.5 / 63.4
75–150K	47.4 / 46.7	49.6 / 49.0	52.0 / 55.4	47.7 / 49.9	65.8 / 66.6
150–300K	44.3 / 50.0	45.6 / 53.3	52.6 / 61.4	49.2 / 54.5	62.8 / 72.2
>300K	59.7 / 47.1	66.9 / 47.9	65.1 / 55.8	63.4 / 50.4	82.9 / 66.9
Code2LoRA EM dips in the largest IR bin, but this is not embedding-specific: every method drops in lockstep, reflecting harder completions rather than the embedding saturating. Per-repo EM–log-size correlation is flat-to-slightly-positive for all methods.

The recurrent embedding scales too: repeating the ablation on the evolution track for Code2LoRA-Evo, it is likewise flat, including on the largest repos:

repo size (≈tok)	CR: QnAs / Evo	IR: QnAs / Evo
<25K	4,265 / 64.7	3,518 / 63.3
25–75K	13,639 / 56.5	12,552 / 64.7
75–150K	13,621 / 64.1	10,612 / 64.8
150–300K	11,970 / 57.4	8,903 / 65.5
>300K	1,237 / 73.3	6,476 / 62.8
Per-repo EM vs. log10(size) is again flat (Pearson +0.105 CR, −0.046 IR) as well. These repositories are large, with source size median 165K, mean 285K, p95 ≈1.03M, p99 ≈1.87M, max ≈3.0M tokens. The ">300K" bin already reaches well past a single context window. Current pooled vectors may eventually limit gigantic repositories and we flag hierarchical/graph encoders as future work, but at the scales tested they degrade neither absolutely nor relative to any baseline. We will report both tables in the paper.

Weakness 4 + Suggestion 4: Potential memorisation-driven overfitting in the hypernetwork
Memorization is controlled directly by our evaluation design, which spans three regimes of decreasing training exposure:

In-repo (IR): the repository was seen in training but the evaluated completions are held out - measures use of exposed repository knowledge on unseen instances.

Cross-repo (CR): the 52 test repositories are entirely absent from training (headline 63.8% EM); the hypernetwork sees each at test for the first time, so beating every baseline here cannot be repository memorization.

Held-out OOD: repositories created after the training cutoff, provably in neither our training set nor the backbone's pretraining - here Code2LoRA-Evo is strongest.

Leakage analysis: we collect the 20,861 unique training answer strings (assertion RHSs) and label each CR-test instance by whether its answer was seen or never seen in training (impossible to memorize); 50.5% are never-seen. Per method:

CR-test (n=6,414)	EM on never-seen answers	EM on seen answers	of correct preds, % whose answer was seen
Pretrained (never trained on our data - control)	32.8	58.9	63.7
FFT	42.2	60.9	58.6
sLoRA	39.3	55.8	58.2
Code2LoRA-Static	50.5	77.4	60.0
Two things follow. (i) On answers it could not have copied (never in training), Code2LoRA scores 50.5%, above FFT (42.2) and pretrained (32.8) - it generalizes to unseen answers. (ii) The third column is a memorization probe: of the predictions each method gets right, how many had a training-seen answer? Pretrained is the control - never trained on our data, so its 63.7% is the overlap expected purely from recurring common values (True, 0, None, …). Code2LoRA's correct answers are only 60.0% seen - below the control, whereas a memorizer would sit above it. (IR shows the same pattern, but there the overlap is ≈90% for every method since IR repositories were in training.) We will add this analysis to the appendix.

Weakness 5: Potential bias in the benchmark toward the proposed method
Releasing each repository whole gives retrieval more to work with, not less, nothing is hidden from RAG/DRC. What limits retrieval is its own selection step, not information the benchmark withholds.

The residual gap is a property of the methods, not the benchmark: on instances that require repository knowledge, retrieval recovers only ≈10% vs ≈38–42% for parametric adaptation - both see the same whole repository, but a few chunks cannot surface globally-scattered facts, exactly the distinction the benchmark measures. The task is also externally motivated (assertion completion from LiveCodeBench (Jain et al. 2024), not tailored to us), with target-type composition matched across splits within ≈2 pp. We state the scope as one task, one language, one backbone, within which the benchmark treats retrieval fairly.

Add:
Rebuttal by Authors
Official Commentby Authors (Liliana Hotsko, Yuntian Deng, Yinxi Li, Pengyu Nie)12 Jul 2026, 03:28 (modified: 03 Aug 2026, 10:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer yeRc, Authors, Commitment ReadersRevisions
Comment:
Thank you for the constructive review and for recognizing RepoPeftBench and the parametric-injection paradigm. Below we tighten the claims and report new analyses:

Weakness 1 + Suggestion 1: The "zero inference cost" claim is overstated
We agree that our wording could be interpreted too broadly. Our claim is specifically zero additional per-query context tokens, rather than zero end-to-end computation.

At query time the merged-adapter model reads only the test prefix (0 extra tokens). Adapter generation is a one-time, per-repo-state cost amortized over all the queries a repository receives; measured end-to-end on a single H100: full cold build 34–257 s (188–1,073 commits), incremental per-commit update ≈21–23 s, and even on CPU (8 threads, no GPU) it is only 344–1,079 s cold / 30–297 s incremental. The Evo per-commit update is offline work done at commit time (one diff-embed + one recurrent step), not a per-inference cost, so there is no "continuous computation during usage."

We will modify the abstract to "zero inference-time token overhead" and add this latency/storage paragraph.

Weakness 2 + Suggestion 2: Limited or potentially biased comparison with strong baselines
We agree our experiments do not exhaust the space of retrieval and hybrid systems; our conclusion holds within the controlled setting of RepoPeftBench, Qwen2.5-Coder-1.5B, and the evaluated pipelines. But 
 is not arbitrary — the Appendix C sweep over 
 and chunk sizes finds it strongest, so more depth does not close the gap. We will narrow the claim from "parameters are generally better than context" to "parametric adaptation outperforms the evaluated retrieve-and-prepend and dependency-resolved baselines in our setting."

We also strengthened the context baselines by fixing the injection format. Static track: the backbone's native repository-completion layout (<|repo_name|>/<|file_sep|>) lifts RAG 39.7 / 42.1 → 47.6 / 48.0 (CR / IR). Evolution track: the submitted RAG used generic non-AST chunks and no decode-time control-token ban; corrected, RAG reaches 36.8 / 37.4.

Stronger/hybrid baselines still trail: repo-FIM RAG (47.6 / 48.0) stays ≈16–18 pp below Code2LoRA-Static; DRC (a retrieval+compression hybrid) reaches 49.1 / 50.0; a trained+retrieval hybrid helps too (FFT+RAG 53.9 / 56.8 > FFT 51.4 / 55.9). All remain well below Code2LoRA-Static.

Long-context LLMs: we hold the backbone fixed across all methods to isolate the repository-injection mechanism; long-context modeling is complementary (parametric adaptation can sit on top of a long-context backbone and still avoids re-encoding the repo per query), which we flag as future work.

Per-repo LoRA's IR-only role is definitional: it needs per-repo training instances, which exist only for IR (CR repos are held out, so training there would leak the eval data). Yet Code2LoRA-Static matches this upper bound in IR (66.2 vs. 64.0) with no per-repo training.

Matched budgets: all methods share the frozen backbone, so per-query cost tracks input length. The merged adapter adds 0 tokens and is latency-identical to the bare backbone (197.1 vs. 198.3 ms at in=2048/out=64), whereas RAG/DRC re-encode ≈1,536/≈4,096 extra tokens every query (H100, greedy, ≈2,048-token prefix):

Method	input tok	out=8 (ms)	out=64 (ms)
Pretrained / FFT / sLoRA / Code2LoRA	≈2,048	36.0	198.3
RAG (k=3)	≈3,584	44.5 (+24%)	206.6 (+4%)
DRC	≈4,096	53.3 (+48%)	216.8 (+9%)
Forward-pass FLOPs/query (dense 2·N_nonembed/token + attention + lm-head, N_nonembed≈1.31B) confirm this: Code2LoRA ≈5.4×10¹² (CR) / 7.1×10¹² (IR) vs. RAG ≈1.0–1.2×10¹³ and DRC ≈1.2×10¹³ — RAG/DRC spend ≈1.7–2.3× the per-query compute for context Code2LoRA has internalized parametrically; its only extra cost is the one-time adapter generation.

Weakness 6: Limited evaluation of real-world applicability and usability
Adapter-generation latency is the one-time, offline, GPU-free step quantified in W1, amortized over the thousands of completions a repository serves. Storage scales with repos as a strength: the per-repository footprint is ≈8 KB (repo embedding / GRU state) vs. ≈37 MB × N for stored per-repo adapters, since the shared hypernetwork (≈1.4 GB, repo-independent) regenerates any adapter in one forward pass — so storage is effectively constant.

Workflow integration: we are building a CPU-supported integration layer for coding assistants to improve applicability and usability. Users will be able to regenerate/incrementally update a repo's adapter offline at commit/push time, then serve with zero added inference tokens, identical to the bare backbone.