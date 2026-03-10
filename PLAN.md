# EMNLP Experiment Execution Plan

**Last updated:** Mar 9, 2026

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| Dataset (560 repos, 73K QnA) | Done | `$SCRATCH/REPO_DATASET/` |
| Splits (train/ir_val/ir_test/cr_val/cr_test) | Done | `$SCRATCH/REPO_DATASET/*.json` |
| Repo-level embeddings | Done | `$SCRATCH/REPO_DATASET/repositories/*/REPO_METADATA.json` |
| Pretrained baseline | **Done** | 45.7% CR / 46.8% IR |
| RAG baseline (k=3) | **Done** | 39.9% CR / 42.3% IR |
| ICL baseline (3-shot) | **Done** | 42.2% CR / 45.4% IR |
| Oracle context baseline | **Done** | 46.2% CR / 47.7% IR |
| FFT baseline | **Done** | 51.4% CR / 55.9% IR |
| Single LoRA (r=16, bf16) | **Done but weak** | 45.0% CR / 48.4% IR — r=64 retrain needed |
| Per-repo LoRA (bf16) | **Partial** | 57.6% IR (10 repos only, bf16) — needs full run |
| Hypernetwork — Direct-Projection | **Done** | 63.8% CR / 66.2% IR |
| Hypernetwork — PAW (Shared-Basis) | **Done** | 64.1% CR / 65.8% IR |
| Scaling (training repos: 50/100/200) | **Training** | `scripts/slurm/scale_hypernet_{50,100,200}.sh` |
| Single LoRA r=64 retrain | **Pending** | Config updated in `train_single_lora.py` |
| All +Oracle variants | **NOT DONE** | Training scripts ready |
| Composable hypernetwork | **NOT DONE** | Code at `hypernetwork/hypernetwork_composable.py` |
| Scale experiments (0.5B, 3B) | **NOT DONE** | Script at `scripts/slurm/phase8_scale.sh` |
| File-level embeddings | **NOT DONE** | Code ready, repos need re-embedding |
| Analysis + figures | **NOT DONE** | Code at `analysis/` |

## Results Table (Mar 10 — after metric fix)

**Config:** `max_input_tokens=16384`, `max_new_tokens=128`, comma-leading targets filtered, all bf16.

**Mar 10 fix:** Fixed `postprocess_prediction()` to truncate overgenerated predictions to match
target format. FFT/sLoRA EditSim was drastically underreported (0.37→0.70, 0.27→0.64).
See `EXPERIMENT_LOG.md` for details.

| Method | CR Test EM | CR EditSim | IR Test EM | IR EditSim | Status |
|--------|-----------|------------|-----------|------------|--------|
| **Inference-only** | | | | | |
| Pretrained | 45.71% | 0.605 | 46.82% | 0.624 | Re-eval pending (CodeBLEU) |
| RAG (k=3) | 39.93% | 0.535 | 42.32% | 0.560 | Re-eval pending (CodeBLEU) |
| ICL (3-shot) | 42.22% | 0.559 | 45.35% | 0.596 | Re-eval pending (CodeBLEU) |
| Oracle Context | 47.47% | 0.614 | 48.74% | 0.627 | Re-eval pending (CodeBLEU) |
| **Trained** | | | | | |
| FFT | 51.42% | 0.696 | 55.88% | 0.727 | Re-eval pending (CodeBLEU) |
| Single LoRA (r=16) | 45.00% | 0.645 | 48.35% | 0.674 | r=64 eval pending |
| Single LoRA (r=64) | — | — | — | — | Eval pending (job 10102798) |
| Per-repo LoRA | N/A | N/A | 57.58% (10 repos) | 0.685 | Full run: jobs 10102795-10102801 |
| Code2LoRA (Direct) | **63.81%** | **0.784** | **66.24%** | **0.806** | Re-eval pending (CodeBLEU) |
| Code2LoRA (PAW) | **64.09%** | **0.785** | 65.84% | 0.805 | Re-eval pending (CodeBLEU) |
| **+Oracle (training submitted)** | | | | | |
| FFT + Oracle | — | — | — | — | Training: job 10102804 |
| Single LoRA + Oracle | — | — | — | — | Training: job 10102806 |
| Code2LoRA + Oracle | — | — | — | — | Training: job 10102805 |
| Code2LoRA PAW + Oracle | — | — | — | — | Training: job 10102807 |

### Architecture Comparison (Paper Table)

| Variant | CR EM | IR EM | Mapper Params |
|---------|-------|-------|---------------|
| Direct-Projection | 63.81% | 66.24% | ~120M |
| Shared-Basis (PAW) | 64.09% | 65.84% | ~14M |

PAW achieves comparable performance with ~8.5x fewer mapper parameters.

### Scaling Experiments (Paper Table 13 — in progress)

| Training Repos | CR Test EM (%) |
|----------------|----------------|
| 50 (12%) | *training* |
| 100 (24%) | *training* |
| 200 (49%) | *training* |
| 409 (100%) | 63.81% |

Scripts: `scripts/slurm/scale_hypernet_{50,100,200}.sh`

## Key Issues & Next Steps

### Critical (for paper)

1. **Single LoRA too weak (45% vs FFT 51.4%).** Root cause: r=16 only 18.4M params, aggressive grad clipping (0.3), small batch (grad_accum=4), dropout (0.05). Fix applied: r=64, alpha=128, grad_accum=8, dropout=0, max_grad_norm=1.0, save_total_limit=3. **Action:** `sbatch scripts/slurm/train_single_lora.sh` then re-evaluate.

2. **Per-repo LoRA incomplete.** Only 10 repos evaluated (99 examples). Updated to bf16 (was 4-bit). **Action:** Full run with all IR test repos.

3. **CodeBLEU=0.0 for Pretrained.** Evaluated before `codebleu` package was installed. **Action:** Re-run pretrained eval.

4. **Scaling experiments still training.** 50/100/200 repo training runs submitted, need to complete and eval on CR test.

### Important (for completeness)

5. **+Oracle variants not run.** FFT+Oracle, sLoRA+Oracle, pLoRA+Oracle, Code2LoRA+Oracle all pending. Training scripts exist.

6. **Oracle cross-evaluation ablation.** Train Code2LoRA with oracle, test with/without. Not yet started.

7. **Composable hypernetwork.** Code exists but not trained/evaluated.

### Nice-to-have (for analysis)

8. **Scale experiments (0.5B, 3B base models).** Script at `phase8_scale.sh`. Not yet run.
9. **Per-repo performance distribution.** Needs all methods evaluated on IR test.
10. **Qualitative examples.** Can be done once all results are in.
11. **LoRA weight similarity analysis.** Code2LoRA vs pLoRA adapter similarity.

## SLURM Scripts Reference

### Training

| Script | Method | Status |
|--------|--------|--------|
| `scripts/slurm/train_hypernet.sh` | Hypernetwork (Direct) | Done |
| `scripts/slurm/train_hypernet_paw.sh` | Hypernetwork (PAW) | Done |
| `scripts/slurm/train_fft.sh` | FFT | Done |
| `scripts/slurm/train_single_lora.sh` | Single LoRA | Done (r=16); **r=64 retrain pending** |
| `scripts/slurm/train_per_repo_lora.sh` | Per-repo LoRA | Partial (10 repos) |
| `scripts/slurm/train_hypernet_oracle.sh` | Hypernetwork + Oracle | Pending |
| `scripts/slurm/train_hypernet_paw_oracle.sh` | PAW + Oracle | Pending |
| `scripts/slurm/train_fft_oracle.sh` | FFT + Oracle | Pending |
| `scripts/slurm/train_single_lora_oracle.sh` | sLoRA + Oracle | Pending |
| `scripts/slurm/train_per_repo_lora_oracle.sh` | pLoRA + Oracle | Pending |
| `scripts/slurm/scale_hypernet_50.sh` | Scaling (50 repos) | Training |
| `scripts/slurm/scale_hypernet_100.sh` | Scaling (100 repos) | Training |
| `scripts/slurm/scale_hypernet_200.sh` | Scaling (200 repos) | Training |

### Evaluation

| Script | Method | Status |
|--------|--------|--------|
| `scripts/slurm/eval_hypernet_no_oracle.sh` | Hypernetwork (Direct) | Done |
| `scripts/slurm/eval_hypernet_paw_no_oracle.sh` | Hypernetwork (PAW) | Done |
| `scripts/slurm/eval_fft_no_oracle.sh` | FFT | Done |
| `scripts/slurm/eval_single_lora_no_oracle.sh` | Single LoRA | Done (r=16) |
| `scripts/slurm/eval_per_repo_lora_no_oracle.sh` | Per-repo LoRA | Done (partial) |

### Inference-only Baselines

| Script | Method | Status |
|--------|--------|--------|
| `scripts/slurm/phase2a_pretrained.sh` | Pretrained | Done |
| `scripts/slurm/phase2b_rag.sh` | RAG | Done |
| `scripts/slurm/phase2c_icl.sh` | ICL | Done |
| `scripts/slurm/phase2d_oracle.sh` | Oracle Context | Done |

## Execution Priority

### Immediate (submit now)

```bash
# 1. Re-train Single LoRA with r=64
sbatch scripts/slurm/train_single_lora.sh

# 2. Wait for scaling experiments to complete, then check results
# (already running: scale_h50, scale_h100, scale_h200)
```

### After sLoRA retrain completes

```bash
# 3. Evaluate re-trained Single LoRA
sbatch scripts/slurm/eval_single_lora_no_oracle.sh

# 4. Full per-repo LoRA run (all IR repos)
sbatch scripts/slurm/train_per_repo_lora.sh

# 5. Re-run pretrained eval (for CodeBLEU)
sbatch scripts/slurm/phase2a_pretrained.sh
```

### +Oracle variants (after above)

```bash
# 6. Train all +Oracle methods
sbatch scripts/slurm/train_fft_oracle.sh
sbatch scripts/slurm/train_single_lora_oracle.sh
sbatch scripts/slurm/train_per_repo_lora_oracle.sh
sbatch scripts/slurm/train_hypernet_oracle.sh
# Then evaluate each
```

### Analysis (after all results)

```bash
# 7. Generate figures and analysis
sbatch scripts/slurm/phase10_analysis.sh
```

## Paper Table Mapping

| Paper Table | Content | Data Source |
|-------------|---------|-------------|
| Table 1 (Main Results) | All methods × CR/IR × EM/CodeBLEU/EditSim | Results table above |
| Table 2 (Oracle Cross-Eval) | Code2LoRA ±oracle train × ±oracle test | Not yet run |
| Table 13 (Training Set Size) | 50/100/200/409 repos → CR EM | Scaling experiments (training) |
| Table (Architecture) | Direct vs PAW: EM, params | Done: 63.8 vs 64.1 CR |
| Figure (Per-repo Distribution) | Box plot of per-repo EM | Needs all IR results |
| Figure (Scaling w/ Repos) | CR EM vs #repos | Scaling experiments |

## Key Findings So Far

1. **Code2LoRA dominates all baselines.** 63.8–64.1% CR EM vs next-best FFT at 51.4%. That's a +12.4pp improvement. After metric fix, EditSim gap is smaller (0.784 vs 0.696) but EM gap is unchanged and decisive.

2. **PAW matches Direct-Projection with 8.5x fewer params.** PAW (64.1% CR) ≈ Direct (63.8%) but 14M vs 120M mapper parameters. Strong efficiency result.

3. **FFT > sLoRA > Pretrained on both splits.** Expected ranking, but sLoRA gap to FFT is larger than typical (~6pp). The r=64 retrain should close this. With fixed metrics, sLoRA EditSim (0.645) is now in the expected range.

4. **RAG and ICL hurt performance vs pretrained.** RAG (39.9%) and ICL (42.2%) both trail Pretrained (45.7%) on CR. Likely caused by long-context truncation and noisy retrieved examples displacing useful code prefix.

5. **Oracle context provides modest gain.** Oracle (47.5%) vs Pretrained (45.7%) = +1.8pp on CR. The import-resolved context helps but is not transformative without adaptation.

6. **IR consistently better than CR.** All methods score higher on IR (in-repo) than CR (cross-repo), as expected — training data for those repos exists.

7. **Mar 10 metric fix: FFT/sLoRA EditSim was drastically underreported.** Overgeneration after `### Target:\n` caused `postprocess_prediction()` to keep hundreds of extra tokens. Fixed by truncating predictions to match target format. FFT EditSim: 0.37→0.70, sLoRA: 0.27→0.64. Code2LoRA unaffected.
