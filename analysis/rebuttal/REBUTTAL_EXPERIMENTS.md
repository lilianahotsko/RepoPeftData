# Rebuttal experiment suite — Code2LoRA (Submission9808)

One consolidated set of experiments covering **both** reviewers (v1i3 and yeRc).
Each entry lists the reviewer concern it answers, the script, the output file, how
to (re)run it, and current status/headline numbers. All analysis scripts live in
`analysis/rebuttal/`; outputs land in `analysis/rebuttal/outputs/`.

Environment for every command below:

```bash
module load StdEnv/2023 python/3.12 gcc/12.3 arrow
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
export BASELINES_DIR=$SCRATCH/BASELINES
export REPOS_ROOT=$SCRATCH/REPO_DATASET/repositories
```

**Do NOT run the model / large parquet scans on the login node.** Wrap them in
`srun --account=def-yuntian --cpus-per-task=4 --mem=32G --time=0:30:00 <cmd>`
(or submit the provided SLURM scripts). Only the small stored-prediction
reanalyses are safe to run directly.

---

## Coverage matrix

| Reviewer / weakness | Experiment | Output | Status |
|---|---|---|---|
| v1i3-W1 (RAG < pretrained; context baseline weak) · yeRc-W2 (context baselines under-tuned) | Repo-aware **FIM** context baseline (RAG & DRC) | `BASELINES_V2/{rag,drc}_rebuttal/` | **done** (GPU) |
| v1i3-W2 (how much is locally solvable / relaxed-EM / gains on trivial targets) | Fraction-solvable, repo-knowledge subset gap, EM-by-target-length, strict-EM | `outputs/reanalysis.md` | **done** |
| v1i3-W3 (no CIs / significance) | 95% bootstrap CIs (instance + repo-clustered), McNemar, paired bootstrap | `outputs/reanalysis.md` | **done** |
| yeRc-W1 / W6 ("zero inference cost" overclaim; efficiency) | Token-overhead + amortization; CPU adapter-gen latency | `outputs/amortization.md` | **done** (latency backfills from CPU job) |
| yeRc-W2 / Suggestion 2 (matched compute budgets) | Per-query input tokens + approximate FLOPs (+ optional GPU latency) per method | `outputs/compute_budget.md` | **done** (tokens/FLOPs; latency optional GPU) |
| yeRc-W3 (fixed embedding can't scale to large repos) | EM vs repository size | `outputs/repo_size_ablation.md` | **done** |
| yeRc-W4 (memorization / leakage) | Novel-target rate, EM on novel targets, seen-fraction of correct preds | `outputs/leakage_analysis.md` | **done** |

---

## 1. Repo-aware FIM context baseline — v1i3-W1, yeRc-W2

**Claim tested.** RAG falls *below* the pretrained model not because context is
useless but because naive prepending (`context + "\n\n\n" + prefix`) triggers
Qwen2.5-Coder's fill-in-the-middle code path (the model emits `<|fim_suffix|>`
and predicts the wrong half). We add a `--prompt-format repo_fim` mode that lays
context out in the backbone's *native* repository-level completion format
(`<|repo_name|> … <|file_sep|> …`) and re-run RAG and DRC.

Code: `evaluation/run_baselines_v2.py` (`_format_repo_fim_prompt`,
`--prompt-format {plain,repo_fim}`); launcher: `PROMPT_FORMAT` env in
`scripts/slurm/eval_baselines_v2_sharded.sh`. Output files tagged `_fim` so they
never overwrite the plain baselines.

Submitted (H100, `--account=rrg-yuntian`, cr_test + ir_test, 4 shards each):

```bash
for M in rag drc; do for F in plain repo_fim; do
  PROMPT_FORMAT=$F METHOD=$M SUFFIX=rebuttal NUM_SHARDS=4 SUITES="cr_test ir_test" \
    RAG_CACHE_DIR=$SCRATCH/RAG_CHUNK_CACHE_AST_COMMITS RAG_TOP_K=3 \
    DRC_CACHE_DIR=$SCRATCH/ORACLE_CONTEXT_CACHE_COMMITS DRC_MAX_TOKENS=4096 \
    sbatch --account=rrg-yuntian --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
done; done
```

Merge shards when finished, then read the summaries:

```bash
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $SCRATCH/TRAINING_CHECKPOINTS/BASELINES_V2/rag_rebuttal
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $SCRATCH/TRAINING_CHECKPOINTS/BASELINES_V2/drc_rebuttal
```

**Final matched results (V2 evolution track, n=44,732 cr / 42,061 ir, all shards
finalized):**

| Method | cr_test EM | ir_test EM |
|---|---|---|
| Pretrained | 31.5 | 29.3 |
| RAG (plain prepend) | 36.7 | 36.9 |
| RAG (repo_fim) | 36.8 | 37.4 |
| DRC (plain prepend) | 36.5 | 36.9 |
| DRC (repo_fim) | 36.3 | 36.8 |
| **Code2LoRA-Evo (GRU)** | **60.3** | **64.5** |

**Interpretation (important — this is *not* the early partial signal).** On the
evolution track the "RAG < pretrained" pathology does **not** appear: RAG and DRC
already beat pretrained by +5–7 pp. Native repo-FIM formatting gives only a
marginal RAG gain (+0.1 cr / +0.5 ir) and is neutral-to-slightly-negative for DRC
(−0.2 / −0.1). The V2 driver already suppresses FIM degeneracy at decode time via
`bad_words_ids` (banning `<|fim_suffix|>` etc.), so *both* the plain and FIM
variants are competent context baselines — and they plateau near ~37%. The
headline: **even a properly repo-formatted retrieval/DRC baseline stays ~23–27 pp
below Code2LoRA-Evo**, so the reviewers' "context baselines are under-tuned"
concern does not explain the gap. The original static-track RAG<pretrained number
(V1 pipeline, Table 1) should be presented as a decoding/formatting artifact of
that pipeline rather than evidence retrieval is useless.

Jobs: 17386784 / 787 / 788 / 789 (all COMPLETED).

### Static-track repo_fim RAG (directly targets the Table-1 RAG<pretrained number)

**Key discovery:** the V1 static RAG script `baselines/rag/test_rag.py` **already
bans** the Qwen FIM/`file_sep`/`repo_name` tokens at decode (`bad_words_ids`,
lines ~149–169). So the paper's static RAG=39.7/42.1 was produced *with* FIM
degeneracy already suppressed — the "naive prepend triggers FIM degeneracy"
mechanism does **not** explain why static RAG trails pretrained (45.7/46.8).

To test whether *native repo-level formatting* (rather than token-banning) helps,
`test_rag.py` now takes `--prompt-format {plain,repo_fim}` and
`scripts/slurm/rerun_rag_v1_fim_ban.sh` takes `PROMPT_FORMAT`. Submitted
(job **17409780**, top_k=3, max_input=16384, cache `RAG_CHUNK_CACHE`):

```bash
PROMPT_FORMAT=repo_fim OUT_DIR=$SCRATCH/RAG_V1_FIM_BAN TOP_K=3 \
  sbatch --account=rrg-yuntian --array=0-1 scripts/slurm/rerun_rag_v1_fim_ban.sh
# -> $SCRATCH/RAG_V1_FIM_BAN/rag_top3_repofim_{cr,ir}_test_fim_ban.json
```

Reference points on the same static split: plain RAG 39.7/42.1, pretrained
45.7/46.8, DRC 48.2/49.5, Code2LoRA-Static 63.8/66.2. **Status: RUNNING**
(~3 h wall; cumulative EM is repo-order-confounded until the run finalizes — read
`exact_match_pct` from the JSONs only at n == n_total).

## 2. Local-solvability, subset gap, target-length, strict-EM — v1i3-W2

Script: `python analysis/rebuttal/reanalysis.py` → `outputs/reanalysis.md`.

- **Fraction solvable from the local prefix** (= pretrained EM): **45.7%** (CR) /
  46.8% (IR). So ~54% of test instances require repository knowledge the model
  cannot get from the local prefix.
- **Repo-knowledge-required subset** (instances pretrained gets wrong): on this
  subset Code2LoRA-Static recovers **38.1%** (CR) / 41.5% (IR), vs FFT 33.7 / 40.4,
  sLoRA 32.1 / 36.5, and — critically — DRC only 9.1 / 9.1 and RAG 8.0 / 8.4.
  Context injection barely helps exactly where repo knowledge is needed;
  parametric adaptation is what moves the needle.
- **EM by target length:** Code2LoRA leads in *every* length bin, including the
  longest (≥20 chars: 24.7 vs 20.7 FFT on CR), refuting "gains only on trivial
  short targets."
- **Strict-EM (no overgeneration truncation), applied identically to all
  methods:** Code2LoRA-Static is essentially unchanged (63.8 → 63.8 CR), whereas
  FFT collapses 51.4 → 14.8 and sLoRA 47.4 → 7.5. The relaxed-EM rule the
  reviewer worried about *flatters the fine-tuned baselines, not Code2LoRA*;
  under strict scoring Code2LoRA's lead widens.

## 3. Statistical rigor — v1i3-W3

Same script/output. Per-method **95% bootstrap CIs** (instance-level and
repo-clustered) are in the first table. Headline gaps are all decisive:

| Comparison (CR) | ΔEM | 95% CI | McNemar p |
|---|---|---|---|
| Code2LoRA-Static − FFT | +12.4 | [+11.4, +13.4] | <1e-12 |
| − DRC | +15.6 | [+14.4, +16.8] | <1e-12 |
| − sLoRA | +16.4 | [+15.3, +17.5] | <1e-12 |
| − Pretrained | +18.1 | [+17.0, +19.2] | <1e-12 |

(IR track analogous; see `outputs/reanalysis.md`.)

## 4. "Zero inference cost" + efficiency — yeRc-W1, yeRc-W6

Script: `python analysis/rebuttal/amortization.py` → `outputs/amortization.md`.

- Code2LoRA adds **0** input tokens per query (adapter merged into weights);
  RAG re-encodes up to **1,536** and DRC up to **4,096** context tokens on
  *every* query. The claim is precisely "zero inference-time *token* overhead."
- Adapter generation is a **one-time, per-repo-state** cost amortized over many
  completions (queries/repo are in the thousands on our suites).
- CPU wall-clock measured by `scripts/slurm/benchmark_c2l_adapt_cpu.sh` (SLURM job
  **17387367**, COMPLETED; Intel Xeon 6972P, 4 threads). Full snapshot walk:
  **344–1,079 s** (188–1,073 commits); **incremental** `/adapt` for one new commit:
  **30–297 s**; unchanged repo (cache hit): **~23–296 s**. This is paid once per
  repo-state, then every subsequent completion has zero adapter cost — vs. RAG/DRC
  re-encoding 1.5–4K context tokens on *every* query. (The earlier job 17386812
  failed because c2l 0.2.1 dropped the `--device/--offline` adapt flags; now passed
  via `C2L_DEVICE`/`C2L_OFFLINE` env.)

## 4b. Matched compute budgets: tokens / FLOPs / latency — yeRc-W2, Suggestion 2

Script: `python analysis/rebuttal/compute_budget.py` → `outputs/compute_budget.md`
(CPU-only for tokens + FLOPs; add `--measure-latency` on a GPU node, or submit
`scripts/slurm/compute_budget.sh`, to also record ms/query).

Shows Code2LoRA uses **much lower inference compute** than the context baselines,
because all methods share the frozen Qwen2.5-Coder-1.5B backbone and per-query
cost is set by input length. Approximate forward-pass FLOPs/query (dense
2·N_nonembed/token + attention + lm-head, N_nonembed≈1.31B):

| Method | input tok/query | ≈FLOPs/query (CR) | ×vs Code2LoRA |
|---|---|---|---|
| Code2LoRA / FFT / sLoRA / Pretrained | 1,940 (prefix only) | 5.4e12 | 1.00× |
| RAG (k=3) | 3,476 (+1,536) | 1.02e13 | 1.87× |
| DRC | 4,096 (fills cap) | 1.22e13 | 2.25× |

The merged adapter adds **0** input tokens; RAG/DRC re-encode 1.5K/4K context
tokens on **every** query (≈1.7–2.3× the FLOPs, CR/IR). Code2LoRA's only extra
compute is the one-time adapter generation, amortized over ≈1,375–9,342
queries/repo (§4). Measured H100 latency (from `amortization.md`) tracks the FLOPs:
+24%/+48% prefill for RAG/DRC at out=8.

## 5. Scaling to large repositories — yeRc-W3

Script: `python analysis/rebuttal/repo_size_ablation.py` → `outputs/repo_size_ablation.md`.

Repository size = total chars of non-test `*.py` files. Code2LoRA-Static keeps
its lead across **all** size bins, including the largest:

| repo size (≈tokens) | Pretrained | FFT | Code2LoRA-Static |
|---|---|---|---|
| <25K (CR) | 36.7 | 51.3 | **57.9** |
| >300K (CR) | 59.7 | 65.1 | **82.9** |
| >300K (IR) | 47.1 | 55.8 | **66.9** |

Per-repo EM vs `log10(size)` correlation is flat-to-slightly-positive for
Code2LoRA (+0.24 CR, +0.08 IR) — no degradation on large repos, refuting the
"fixed-size pooled embedding can't scale" concern.

The same ablation now covers **Code2LoRA-Evo (GRU)** on the per-commit evolution
track (`--gru-eval-dir`, per-commit EM QnA-weighted per size bin). It is likewise
flat: EM stays high across every bin (CR 56.5–73.3, IR 62.8–65.5, including the
largest >300K repos), with per-repo EM vs `log10(size)` Pearson **+0.105 (CR) /
−0.046 (IR)** — so the *recurrent* fixed-size embedding also shows no size-driven
degradation.

## 6. Memorization / leakage — yeRc-W4

Script: `python analysis/rebuttal/leakage_analysis.py` → `outputs/leakage_analysis.md`.

Training-target set built from the static-track train split (20,861 unique
normalized targets).

- **CR test (fully held-out repos):** 50.5% of targets are **novel** (never a
  training target). Code2LoRA-Static scores **50.5%** EM on the novel-target
  subset (vs FFT 42.2, pretrained 32.8) — it generalizes to answer strings never
  seen in training.
- Among each method's **correct** predictions, the fraction whose target was seen
  in training is **60.0%** for Code2LoRA vs **63.7%** for the *never-trained*
  pretrained model. Code2LoRA's correct answers are **not** disproportionately
  "seen" strings — the opposite of a memorization signature.

---

## How to refresh once the GPU/CPU jobs finish

```bash
# 1. FIM baselines (merge shards, read merged summaries)
python evaluation/merge_eval_shards.py --auto-detect --input-dir $SCRATCH/TRAINING_CHECKPOINTS/BASELINES_V2/rag_rebuttal
python evaluation/merge_eval_shards.py --auto-detect --input-dir $SCRATCH/TRAINING_CHECKPOINTS/BASELINES_V2/drc_rebuttal
# 2. adapter-gen latency -> amortization table
srun --account=def-yuntian --cpus-per-task=4 --mem=32G --time=0:20:00 \
    python analysis/rebuttal/amortization.py
```
