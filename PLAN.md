# EMNLP Experiment Execution Plan

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| Dataset (560 repos, 73K QnA) | Done | `$SCRATCH/REPO_DATASET/` |
| Splits (train/ir_val/ir_test/cr_val/cr_test) | Done | `$SCRATCH/REPO_DATASET/*.json` |
| Repo-level embeddings | Done | `$SCRATCH/REPO_DATASET/repositories/*/REPO_METADATA.json` |
| Hypernetwork (1.5B, r=16) | Trained | `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos/` |
| Hypernetwork eval (cr_test) | Done | 58.02% EM, 0.7921 EditSim |
| Pretrained baseline (cr_test) | Done | 45.6% EM, 0.596 EditSim |
| File-level embeddings | **NOT DONE** | Code ready, repos need re-embedding |
| RAG baseline | **NOT DONE** | Code at `baselines/rag/test_rag.py` (no results in BASELINES) |
| ICL baseline | Done | 41.0% EM (5-shot cr_test), 44.9% (ir_test) |
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

## Expected Results Table

**Note:** After fair baseline setup (unified data loading, max_input_tokens=16384, ICL retrieval), re-run Phase 2 and update this table.

| Method | CR Test EM | CR Test EditSim | IR Test EM | IR Test EditSim | Notes |
|--------|-----------|-----------------|-----------|-----------------|-------|
| Pretrained | 45.6% | 0.596 | 46.8% | 0.615 | Done (pre-fair; re-run for updated) |
| RAG (k=5) | TBD | TBD | TBD | TBD | Phase 2 (no results yet) |
| ICL (3-shot) | 41.7% | 0.555 | 45.3% | 0.593 | Done (pre-fair; re-run with retrieval) |
| ICL (5-shot) | 41.0% | 0.547 | 44.9% | 0.587 | Done (pre-fair; re-run with retrieval) |
| Oracle context | 46.2% | 0.601 | 47.7% | 0.618 | Done |
| Fine-tuned | TBD | TBD | TBD | TBD | Phase 3 |
| Single LoRA | TBD | TBD | TBD | TBD | Phase 3 |
| LoRA per repo | N/A | N/A | TBD | TBD | Phase 4 |
| Code2LoRA | 58.0% | 0.792 | TBD | TBD | Done + Phase 5 |
| Code2LoRA + Compose | TBD | TBD | TBD | TBD | Phase 6-7 |

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
