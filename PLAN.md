# EMNLP Experiment Execution Plan

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| Dataset (560 repos, 73K QnA) | Done | `$SCRATCH/REPO_DATASET/` |
| Splits (train/ir_val/ir_test/cr_val/cr_test) | Done | `$SCRATCH/REPO_DATASET/*.json` |
| Repo-level embeddings | Done | `$SCRATCH/REPO_DATASET/repositories/*/REPO_METADATA.json` |
| Hypernetwork (1.5B, r=16) | Trained | `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos/` |
| Hypernetwork eval (cr_test) | Done | 58.02% EM, 0.7921 EditSim |
| Pretrained baseline (cr_test) | Done | 45.7% EM, 0.597 EditSim |
| File-level embeddings | **NOT DONE** | Code ready, repos need re-embedding |
| RAG baseline | Done | 39.9% EM k=3, 37.8% k=5, 35.8% k=10 (cr_test) |
| ICL baseline | Done | 42.8% EM (3-shot, partial), 41.0% (5-shot cr_test) |
| Oracle context | Done | 46.2% EM (cr_test), 47.7% (ir_test) |
| Fine-tuned baseline | **NOT DONE** | Code at `baselines/finetuned/` |
| Single LoRA baseline | **NOT DONE** | Code at `baselines/single_lora/` |
| Per-repo LoRA (IR) | **NOT DONE** | Code at `baselines/lora_per_repo/` |
| Composable hypernetwork | **NOT DONE** | Code at `hypernetwork/hypernetwork_composable.py` |
| Scale experiments (0.5B, 3B) | **NOT DONE** | Script at `scripts/run_scale_experiment.sh` |
| Ablation studies | **NOT DONE** | Script at `scripts/run_ablations.sh` |
| Analysis + figures | **NOT DONE** | Code at `analysis/` |

## Execution Order

All jobs are submitted via SLURM. Each phase produces checkpoints/results that the next phase depends on. Submit in order, waiting for each to complete before submitting the next.

### Phase 0: Setup (local, no SLURM)

```bash
# Install missing packages into your venv
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
pip install codebleu scikit-learn matplotlib
```

### Phase 1: Re-embed repos with file-level embeddings (GPU, ~2h)

Adds per-file embeddings to `REPO_METADATA.json` and recreates splits.

```bash
sbatch scripts/slurm/phase1_embed.sh
```

**Output**: Updated `$SCRATCH/REPO_DATASET/*.json` with `file_embeddings` fields.

### Phase 2: Inference-only baselines (GPU, ~4h)

Pretrained, RAG (k=3,5,10), ICL (3,5-shot) -- no training needed.

**Fair setup (implemented):** All use `max_input_tokens=16384`, shared `load_split` (comma filter), ICL uses retrieval-based example selection for CR.

```bash
sbatch scripts/slurm/phase2_inference_baselines.sh
# Or run individual scripts: phase2a_pretrained.sh, phase2b_rag.sh, phase2c_icl.sh, phase2d_oracle.sh
# IR splits: phase2b_rag_ir.sh, phase2c_icl_ir.sh, phase2d_oracle_ir.sh
```

**Output**: `$SCRATCH/BASELINES/pretrained_*.json`, `rag_*.json`, `icl_*.json`

### Phase 3: Training baselines (GPU, ~8h)

Fine-tuned and Single LoRA: train then evaluate.

```bash
sbatch scripts/slurm/phase3_training_baselines.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/FINETUNED/`, `SINGLE_LORA/`, `$SCRATCH/BASELINES/finetuned_*.json`, `single_lora_*.json`

### Phase 4: Per-repo LoRA baseline (GPU, ~6h)

Train and evaluate individual LoRA adapters for a representative sample of repos.

```bash
sbatch scripts/slurm/phase4_lora_per_repo.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/PER_REPO_LORA/`, aggregate results

### Phase 5: Re-evaluate hypernetwork on all splits (GPU, ~2h)

The existing hypernetwork checkpoint can be reused. Evaluate on both cr_test and ir_test.

```bash
sbatch scripts/slurm/phase5_hypernet_eval.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos_results/`

### Phase 6: Composable hypernetwork -- weighted strategy (GPU, ~10h)

Train the composable hypernetwork with the weighted composition strategy (primary).

```bash
sbatch scripts/slurm/phase6_composable.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET_COMPOSABLE_weighted/`

### Phase 7: Composable hypernetwork -- other strategies + incremental (GPU, ~10h)

Train additive and gated variants, then run incremental adaptation experiment.

```bash
sbatch scripts/slurm/phase7_composable_variants.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET_COMPOSABLE_additive/`, `_gated/`, incremental results

### Phase 8: Scale experiments (GPU, ~8h)

Train hypernetwork with 0.5B and 3B base models.

```bash
sbatch scripts/slurm/phase8_scale.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/scale_0.5B/`, `scale_3B/`

### Phase 9: Ablation studies (GPU, ~10h)

LoRA rank, hypernetwork width, data size ablations (1 epoch each, eval on cr_val).

```bash
sbatch scripts/slurm/phase9_ablations.sh
```

**Output**: `$SCRATCH/TRAINING_CHECKPOINTS/ABLATIONS/`

### Phase 10: Analysis and figures (GPU, ~1h)

Generate all analysis plots, tables, and paper figures.

```bash
sbatch scripts/slurm/phase10_analysis.sh
```

**Output**: `analysis/output/`, `analysis/figures/`

## Paper Evaluation Design

### Table 1: Main Results (all methods × both splits × 3 metrics)

**Config:** `max_input_tokens=16384`, `max_new_tokens=128`, comma-leading targets filtered.
**Metrics per cell:** Exact Match (EM), Edit Similarity, CodeBLEU.

| Method | CR-test | IR-test | Notes |
|--------|---------|---------|-------|
| **Inference-only (no training)** | | | |
| Pretrained | ✓ | ✓ | Lower bound — no context, no adaptation |
| RAG (k=3) | ✓ | ✓ | Retrieval-augmented context |
| ICL (3-shot) | ✓ | ✓ | Few-shot in-context examples |
| Oracle Context | ✓ | ✓ | Import-resolved source — context upper bound |
| **Trained (prefix → target)** | | | |
| FFT | ✓ | ✓ | Full fine-tuning on all training QnA |
| Single LoRA | ✓ | ✓ | One LoRA across all repos |
| Per-repo LoRA | N/A | ✓ | Adapter per repo — IR upper bound |
| Code2LoRA | ✓ | ✓ | Our method |
| **Trained (oracle + prefix → target)** | | | |
| FFT + Oracle | ✓ | ✓ | Full fine-tuning with oracle context |
| Single LoRA + Oracle | ✓ | ✓ | One LoRA with oracle context |
| Per-repo LoRA + Oracle | N/A | ✓ | Per-repo adapter with oracle context |
| Code2LoRA + Oracle | ✓ | ✓ | Our method with oracle context |

Per-repo LoRA is N/A on CR because no adapter exists for unseen repos — this is exactly the gap Code2LoRA fills.

### Table 2: Oracle Cross-Evaluation Ablation (Code2LoRA only)

Tests robustness: does oracle-trained model degrade gracefully when oracle is unavailable at test time?

| Train condition | Test w/o oracle | Test w/ oracle |
|-----------------|-----------------|----------------|
| Code2LoRA (no oracle) | **main result** | oracle helps at test time only? |
| Code2LoRA (+ oracle) | robustness check | **main result** |

Evaluate on both CR-test and IR-test. The key finding:
- If "train +oracle, test −oracle" ≈ "train −oracle, test −oracle" → oracle training doesn't help without oracle at test time (expected)
- If "train +oracle, test −oracle" > "train −oracle, test −oracle" → model learned something transferable from oracle (strong result)

### Additional Analyses (reviewer expectations)

#### Analysis 1: Per-repo Performance Distribution (Figure)

Box plot or violin plot of EM across individual repos for IR-test.
Shows variance in adaptation quality. Expected:
- Per-repo LoRA: high variance (some repos have few examples)
- Code2LoRA: lower variance (hypernetwork regularizes across repos)
- FFT/sLoRA: uniform (no repo-specific signal)

**Data needed:** per-repo breakdown from IR-test evaluation of all methods.
**Script:** `analysis/per_repo_boxplot.py`

#### Analysis 2: Scaling with Number of Training Repos (Figure)

Train Code2LoRA on 10, 20, 30, 50, 100, all repos → evaluate on CR-test.
Shows the approach benefits from seeing more repos.

**Runs needed:** 5 additional hypernetwork training runs with `--limit-train-repos`.
**Script:** `analysis/scaling_repos.py`

#### Analysis 3: Qualitative Examples (Table in paper)

3–4 success cases where Code2LoRA outperforms baselines + 1–2 failure cases.
Pick from CR-test (most compelling — unseen repos).

Select examples where:
- Code2LoRA correct, pretrained/FFT wrong → adaptation helps
- Code2LoRA correct, pLoRA N/A on CR → shows generalization
- Code2LoRA wrong but close (high edit sim) → shows partial credit

**Script:** `analysis/qualitative_examples.py`

#### Analysis 4: Effect of Oracle Context Coverage (Analysis paragraph)

~87.5% of QnA pairs have oracle context (rest are stdlib/third-party only imports).
Compare performance on the two subsets:
- QnAs WITH oracle context: does oracle help?
- QnAs WITHOUT oracle context: is performance comparable?

**Data needed:** partition test set by oracle availability, compute metrics on each subset.
**Script:** `analysis/oracle_coverage_analysis.py`

#### Analysis 5: LoRA Weight Similarity (Figure)

For Code2LoRA: compute cosine similarity between generated LoRA weights for different repos.
For per-repo LoRA: same metric between trained adapters.

Expected: Code2LoRA generates more similar adapters than independently trained per-repo LoRAs (smooth hypernetwork mapping). Visualize as heatmap or t-SNE of generated LoRA weights.

**Script:** `analysis/lora_similarity.py`

#### Analysis 6: Performance vs. Repo Size / Training Examples (Scatter plot)

Scatter: x = number of training QnA pairs per repo, y = IR-test EM for that repo.
One series per method (pLoRA, Code2LoRA, FFT).

Expected: pLoRA improves with more data per repo; Code2LoRA is more stable (benefits from cross-repo transfer).

**Script:** `analysis/performance_vs_repo_size.py`

### Evaluation Runs Needed

**Total cells to fill (main table):** 22 (12 methods × 2 splits − 2 pLoRA CR)

| Run | SLURM script | Status |
|-----|-------------|--------|
| Pretrained CR/IR | `phase2a_pretrained.sh` | Done |
| RAG CR/IR | `phase2b_rag.sh`, `phase2b_rag_ir.sh` | Done |
| ICL CR/IR | `phase2c_icl.sh`, `phase2c_icl_ir.sh` | Done |
| Oracle CR/IR | `phase2d_oracle.sh`, `phase2d_oracle_ir.sh` | Done |
| FFT CR/IR | eval after `train_fft.sh` | Pending |
| FFT+Oracle CR/IR | eval after `train_fft_oracle.sh` | Pending |
| Single LoRA CR/IR | eval after `train_single_lora.sh` | Pending |
| Single LoRA+Oracle CR/IR | eval after `train_single_lora_oracle.sh` | Pending |
| Per-repo LoRA IR | eval via `train_per_repo_lora.sh` | Pending |
| Per-repo LoRA+Oracle IR | eval via `train_per_repo_lora_oracle.sh` | Pending |
| Code2LoRA CR/IR | eval after `train_hypernet.sh` | Pending |
| Code2LoRA+Oracle CR/IR | eval after `train_hypernet_oracle.sh` | Pending |
| Code2LoRA cross-eval (4 cells) | oracle ablation | Pending |

## Results Table (as of Mar 8)

**Config:** `max_input_tokens=16384`, `max_new_tokens=128`, comma-leading targets filtered.

| Method | CR Test EM | CR Test CodeBLEU | CR Test EditSim | IR Test EM | IR Test CodeBLEU | IR Test EditSim | Status |
|--------|-----------|-----------------|-----------------|-----------|-----------------|-----------------|--------|
| Pretrained | 45.7% | 0.0\* | 0.597 | 46.8% | 0.0\* | 0.615 | Done (n=6414/5222) |
| RAG (k=3) | 39.9% | 0.439 | 0.534 | 42.3% | 0.447 | 0.554 | Done |
| ICL (3-shot) | 42.8% | 0.466 | 0.551 | 45.3% | 0.482 | 0.593 | **CR partial** (3450/6414); IR done |
| Oracle context | 46.2% | 0.489 | 0.601 | 47.7% | 0.491 | 0.618 | Done |
| FFT | — | — | — | — | — | — | Training |
| FFT + Oracle | — | — | — | — | — | — | Pending |
| Single LoRA | — | — | — | — | — | — | Training |
| Single LoRA + Oracle | — | — | — | — | — | — | Pending |
| Per-repo LoRA | N/A | N/A | N/A | — | — | — | Training |
| Per-repo LoRA + Oracle | N/A | N/A | N/A | — | — | — | Pending |
| Code2LoRA | 58.0%† | 0.0\* | 0.792† | — | — | — | Training; old eval needs re-run |
| Code2LoRA + Oracle | — | — | — | — | — | — | Pending |

**Issues to resolve before submission:**

1. **\*CodeBLEU = 0.0:** Pretrained and Code2LoRA were evaluated before `codebleu` package was installed. Re-run to get consistent CodeBLEU.
2. **†Code2LoRA on old data:** Hypernetwork eval used n=7620 (pre-comma-filter) vs n=6414 (post-filter). Re-evaluate with current data.
3. **Incomplete run:** ICL 3-shot CR (3450/6414) timed out. Re-run with more wall time.
4. **RAG hurts performance:** RAG (39.9%) < Pretrained (45.7%). Likely noisy retrieved chunks dilute the prefix. Discuss in paper; may drop RAG k=5,10 and keep only k=3.
5. **ICL < Pretrained on CR:** ICL 3-shot (42.8%) trails pretrained (45.7%) on CR. Needs discussion — context budget, example quality, prompt format.
6. **Training baselines in progress:** FFT, Single LoRA, Per-repo LoRA, Code2LoRA retraining underway.

## Baseline Analysis: ICL vs Pretrained (Top-Tier Fairness)

### Is it normal that ICL underperforms pretrained?

**No — this is atypical.** At top venues (EMNLP, ACL, NeurIPS), reviewers expect:
- ICL ≥ pretrained (few-shot should help)
- 5-shot ≥ 3-shot (more examples should help or at least not hurt)

Your results show the opposite: Pretrained (45.6% EM) > ICL 3-shot (41.7%) > ICL 5-shot (41.0%). This will raise red flags unless explained.

### Likely causes (implementation / setup)

1. **Context budget mismatch**
   - Pretrained: `max_input_tokens=2048` (prefix only)
   - ICL: `max_input_tokens=4096` (examples + prefix)
   - Truncation keeps the *last* N tokens. With 5 examples (~2–3K tokens) + prefix, truncation often drops the start of the prompt → partial or missing examples, which can confuse the model.

2. **Example selection**
   - CR: Examples from *nearest training repo* by repo-level embedding. Repo similarity ≠ assertion-style similarity; examples can be misleading.
   - IR: Random sample from same-repo train pairs. Random sampling can pick weak or noisy examples.

3. **Prompt format**
   - ICL uses meta-instructions: `# Examples of assertion completions:` and `# Now complete the following assertion:`. These may distract or mislead the model.

4. **More shots → more truncation**
   - 5-shot uses more tokens than 3-shot. With fixed 4096, 5-shot is more likely to truncate and lose useful context, which can explain 3-shot > 5-shot.

### Recommendations for a fair, defensible comparison

| Action | Purpose |
|--------|---------|
| **Align `max_input_tokens`** | Use 8192 or 16384 for pretrained, ICL, RAG, oracle so all methods see the same context budget. |
| **Retrieval-based ICL** | For CR, retrieve examples by *prefix* similarity (e.g., embedding of prefix), not just nearest repo. |
| **Example quality** | For IR, try similarity-based selection instead of random sampling. |
| **Ablation on format** | Test simpler prompts (no meta-instructions) to see if format hurts. |
| **Document in paper** | Add a short discussion: “ICL underperformed pretrained; we hypothesize X, Y, Z” and cite the above factors. |

### Verdict for top-tier submission

- **Current state:** Results are likely to be questioned. “Pretrained beats ICL” without explanation will look like a weak or unfair baseline.
- **After fixes (implemented):** Fair setup applied: unified data loading (comma filter), max_input_tokens=16384 for all, ICL retrieval-based example selection for CR. Re-run Phase 2 baselines and update the results table. Code2LoRA’s gain over pretrained (58% vs 45.6%) remains strong; the main risk is baseline credibility.
- **Oracle context** (46.2% EM) is only slightly above pretrained (45.6%), which is plausible: oracle adds relevant code but not the exact answer. That is acceptable.

## Estimated Total GPU Time

| Phase | GPU Hours |
|-------|-----------|
| Phase 1 (embed) | ~2h |
| Phase 2 (inference baselines) | ~4h |
| Phase 3 (training baselines) | ~8h |
| Phase 4 (per-repo LoRA) | ~6h |
| Phase 5 (hypernet eval) | ~1h |
| Phase 6 (composable weighted) | ~10h |
| Phase 7 (composable variants) | ~10h |
| Phase 8 (scale) | ~8h |
| Phase 9 (ablations) | ~10h |
| Phase 10 (analysis) | ~1h |
| **Total** | **~60h H100** |

## Quick Commands Reference

```bash
# Submit all phases sequentially (check each completes before next)
sbatch scripts/slurm/phase1_embed.sh
# ... wait ...
sbatch scripts/slurm/phase2_inference_baselines.sh
# ... wait ...
sbatch scripts/slurm/phase3_training_baselines.sh
# etc.

# Or submit with dependencies:
JOB1=$(sbatch --parsable scripts/slurm/phase1_embed.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase2_inference_baselines.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase3_training_baselines.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase4_lora_per_repo.sh)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase5_hypernet_eval.sh)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase6_composable.sh)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB6 scripts/slurm/phase7_composable_variants.sh)
JOB8=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase8_scale.sh)
JOB9=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/phase9_ablations.sh)
JOB10=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3:$JOB4:$JOB5:$JOB6:$JOB7:$JOB8:$JOB9 scripts/slurm/phase10_analysis.sh)

# Monitor
squeue -u $USER
# Check output
tail -f slurm-*.out
```
