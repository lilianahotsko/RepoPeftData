# EMNLP Experiment Execution Plan

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| Dataset (560 repos, 73K QnA) | Done | `$SCRATCH/REPO_DATASET/` |
| Splits (train/ir_val/ir_test/cr_val/cr_test) | Done | `$SCRATCH/REPO_DATASET/*.json` |
| Repo-level embeddings | Done | `$SCRATCH/REPO_DATASET/repositories/*/REPO_METADATA.json` |
| Hypernetwork (1.5B, r=16) | Trained | `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/full_repos/` |
| Hypernetwork eval (cr_test) | Done | 58.02% EM, 0.7921 EditSim |
| Pretrained baseline (cr_test) | Done | 34.63% EM, 0.5302 EditSim |
| File-level embeddings | **NOT DONE** | Code ready, repos need re-embedding |
| RAG baseline | **NOT DONE** | Code at `baselines/rag/test_rag.py` |
| ICL baseline | **NOT DONE** | Code at `baselines/icl/test_icl.py` |
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

```bash
sbatch scripts/slurm/phase2_inference_baselines.sh
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

| Method | CR Test EM | CR Test EditSim | IR Test EM | Notes |
|--------|-----------|-----------------|-----------|-------|
| Pretrained | 34.6% | 0.530 | TBD | Done |
| RAG (k=5) | TBD | TBD | -- | Phase 2 |
| ICL (5-shot) | TBD | TBD | -- | Phase 2 |
| Fine-tuned | TBD | TBD | TBD | Phase 3 |
| Single LoRA | TBD | TBD | TBD | Phase 3 |
| LoRA per repo | N/A | N/A | TBD | Phase 4 |
| Code2LoRA | 58.0% | 0.792 | TBD | Done + Phase 5 |
| Code2LoRA + Compose | TBD | TBD | TBD | Phase 6-7 |

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
