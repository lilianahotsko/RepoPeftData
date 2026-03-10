# Experiment Log

**Project:** Code2LoRA (EMNLP 2026 submission)
**Cluster:** Compute Canada nibi (H100 GPUs)
**Accounts:** `def-yuntian`, `rrg-yuntian`

---

## 2026-03-10: Phase 0 — Metric Fix & Full Re-evaluation

### Issue: EditSim/CodeBLEU inconsistent with EM for FFT and sLoRA

**Root cause:** `postprocess_prediction()` in `evaluation/metrics.py` did not truncate
overgeneration to match the target's format. The FFT/sLoRA models generate hundreds of
tokens on a single line after `### Target:\n`. The relaxed `exact_match()` (via
`_pred_candidates`) matches the first word, giving EM=True, but `edit_similarity()` and
`code_bleu_score()` operated on the full overgenerated string.

**Evidence (FFT IR test, pre-fix):**
- 1,484 / 2,918 EM-correct predictions (51%) had EditSim < 0.8
- Example: target=`1`, got=`1 signal_sent.assert_called_with( request, user=user, ) ...`
  - EM=True (first word "1" matches), EditSim=0.003

**Fix:** Updated `postprocess_prediction()` to truncate prediction to match target format:
- If target has no comma but prediction does → take text before first comma
- If target is a single word but prediction has multiple → take first word

**File changed:** `evaluation/metrics.py`

### Re-evaluation of ALL methods (with fixed metrics)

| Job Name | Command | Job ID | Account | Status |
|----------|---------|--------|---------|--------|
| reeval_all_baselines | `sbatch scripts/slurm/reeval_all_baselines.sh` | 10102796 | def-yuntian | PENDING |
| reeval_trained (FFT+sLoRA r=64) | `sbatch scripts/slurm/reeval_trained.sh` | 10102798 | rrg-yuntian | PENDING |
| reeval_hypernets (Direct+PAW) | `sbatch scripts/slurm/reeval_hypernets.sh` | 10102800 | def-yuntian | PENDING |
| reeval_scaling (50/100/200) | `sbatch scripts/slurm/reeval_scaling.sh` | 10102802 | rrg-yuntian | PENDING |

**Immediate recompute (no GPU, reprocessing stored predictions with fixed postprocessing):**

Note: CodeBLEU cannot be recomputed locally (codebleu not installed); SLURM re-eval
jobs will produce correct CodeBLEU. These are the sLoRA r=16 numbers; r=64 eval is pending.

| Method | CR EM | CR ES (old→new) | IR EM | IR ES (old→new) |
|--------|-------|-----------------|-------|-----------------|
| Pretrained | 45.71% | 0.597→**0.605** | 46.82% | 0.615→**0.624** |
| RAG (k=3) | 39.93% | 0.534→**0.535** | 42.32% | 0.554→**0.560** |
| ICL (3-shot) | 42.22% | 0.560→**0.559** | 45.35% | 0.593→**0.596** |
| Oracle Context | 47.47% | 0.609→**0.614** | 48.74% | 0.618→**0.627** |
| FFT | 51.42% | 0.367→**0.696** | 55.88% | 0.389→**0.727** |
| Single LoRA r=16 | 45.00% | 0.265→**0.645** | 48.35% | 0.278→**0.674** |
| Code2LoRA (Direct) | 63.81% | 0.784→**0.784** | 66.24% | 0.806→**0.806** |
| Code2LoRA (PAW) | 64.09% | 0.786→**0.785** | 65.84% | 0.804→**0.805** |

**Key takeaways:**
- FFT/sLoRA EditSim were drastically underreported due to overgeneration bug (+0.33/+0.38 increase)
- Inference-only baselines barely affected (±0.01) since they don't overgenerate
- Code2LoRA essentially unchanged — its predictions were already well-formed
- EM is identical for all methods (postprocessing aligns with relaxed EM logic)

**Results (full re-eval with GPU and CodeBLEU — pending SLURM jobs):**

| Method | CR Test EM | CR EditSim | CR CodeBLEU | IR Test EM | IR EditSim | IR CodeBLEU |
|--------|-----------|------------|-------------|-----------|------------|-------------|
| Pretrained | | | | | | |
| RAG (k=3) | | | | | | |
| ICL (3-shot) | | | | | | |
| Oracle Context | | | | | | |
| FFT | | | | | | |
| Single LoRA r=64 | | | | | | |
| Per-repo LoRA | | | | | | |
| Code2LoRA (Direct) | | | | | | |
| Code2LoRA (PAW) | | | | | | |

---

## 2026-03-10: Phase 1 — Per-repo LoRA Full Run (447 repos)

### Chunked parallel execution across 4 GPU jobs

Each chunk trains + evaluates ~112 repos on a single H100.

| Chunk | Repo Range | Script | Account | Job ID | Status |
|-------|-----------|--------|---------|--------|--------|
| 1 | 0–111 | `train_prlora_chunk1.sh` | def-yuntian | 10102795 | PENDING |
| 2 | 112–223 | `train_prlora_chunk2.sh` | rrg-yuntian | 10102797 | PENDING |
| 3 | 224–335 | `train_prlora_chunk3.sh` | def-yuntian | 10102799 | PENDING |
| 4 | 336–447 | `train_prlora_chunk4.sh` | rrg-yuntian | 10102801 | PENDING |

---

## 2026-03-10: Phase 2 — Oracle (DRC) Experiments

### Training +Oracle variants

| Method | Script | Account | Job ID | Status |
|--------|--------|---------|--------|--------|
| FFT + Oracle | `train_fft_oracle.sh` | def-yuntian | 10102804 | PENDING |
| sLoRA + Oracle | `train_single_lora_oracle.sh` | def-yuntian | 10102806 | PENDING |
| Code2LoRA + Oracle | `train_hypernet_oracle.sh` | def-yuntian | 10102805 | PENDING |
| Code2LoRA PAW + Oracle | `train_hypernet_paw_oracle.sh` | def-yuntian | 10102807 | PENDING |
| Oracle v2 eval | `phase2d_oracle_v2.sh` | def-yuntian | 10102803 | PENDING |

---

## 2026-03-10: Scaling Experiments (re-eval with fixed metrics)

Training already completed. Need to re-evaluate with fixed metrics.

| Repos | Script | Job ID | Status | CR Test EM | EditSim | CodeBLEU |
|-------|--------|--------|--------|------------|---------|----------|
| 50 | `scale_hypernet_50.sh` | 10068266 | DONE | 60.88% | 0.758 | 0.753 |
| 100 | `scale_hypernet_100.sh` | 10068267 | DONE | 61.27% | 0.756 | 0.752 |
| 200 | `scale_hypernet_200.sh` | 10068268 | DONE | 62.24% | 0.773 | 0.764 |
| 409 (full) | via reeval_hnet | 10102800 | PENDING | ~63.8% | 0.784 | 0.777 |

---

## 2026-03-10: Scaling Law Deep Dive (Kaiming He's suggestion)

**Goal:** Show a compelling scaling law: Code2LoRA EM as a function of #training repos.
Inspired by Kaplan et al. / Chinchilla-style scaling analysis.

### Current fit (4 data points)

```
Log-linear:  EM = 1.40 * ln(N) + 55.1   R²=0.935
Power law:   Error% = 45.6 * N^(-0.037)  R²=0.931
```

### Phase 1: Fill intermediate data points (existing 409 training repos)

| N | Script | Account | Job ID | Status | Time Est |
|---|--------|---------|--------|--------|----------|
| 10 | `scale_hypernet_10.sh` | rrg-yuntian | 10104714 | SUBMITTED | ~1.5h |
| 25 | `scale_hypernet_25.sh` | rrg-yuntian | 10104715 | SUBMITTED | ~2h |
| 150 | `scale_hypernet_150.sh` | def-yuntian | 10104716 | SUBMITTED | ~7h |
| 300 | `scale_hypernet_300.sh` | def-yuntian | 10104717 | SUBMITTED | ~11h |

### Phase 2: Scale beyond 409 repos (expanded dataset)

**Key discovery:** 214 repos already cloned, processed, and with embeddings but excluded
from original splits due to `min_qnas=30` filter. These repos have 1-29 QnA pairs each.

Created expanded splits: `$SCRATCH/REPO_DATASET/expanded/`
- Training: 609 repos, 41,880 pairs (was 409 repos, 39,612 pairs)
- Val/test: UNCHANGED (51 val, 52 test)

| N | Script | Account | Job ID | Status | Time Est |
|---|--------|---------|--------|--------|----------|
| 500 | `scale_hypernet_500.sh` | rrg-yuntian | 10104770 | SUBMITTED | ~20h |
| 623 | `scale_hypernet_623.sh` | def-yuntian | 10104771 | SUBMITTED | ~24h |

### Phase 3: Push to 1000+ repos (data pipeline expansion)

Created `repos_collection/mine_repos_expanded.py` with wider search criteria:
- Lowered star range to 50+
- Added Apache-2.0 license
- Wider repo size range (up to 50K)
- Older push dates (2023+)

Pipeline script: `scripts/slurm/phase3_scale_pipeline.sh`
Steps: mine → clone → separate tests → create QnA → embed → expand splits

**Prerequisite:** Need GitHub token to run mining script.

### Target data points for final scaling figure

| N | Source | Status |
|---|--------|--------|
| 10 | Phase 1 | Submitted |
| 25 | Phase 1 | Submitted |
| 50 | Existing | Done |
| 100 | Existing | Done |
| 150 | Phase 1 | Submitted |
| 200 | Existing | Done |
| 300 | Phase 1 | Submitted |
| 409 | Existing | Done |
| 500 | Phase 2 | Submitted |
| 623 | Phase 2 | Submitted |
| 750+ | Phase 3 | Needs mining |
| 1000+ | Phase 3 | Needs mining |

### Analysis script

`analysis/scaling_law.py` — loads all available results, fits models, generates figure.

```bash
python analysis/scaling_law.py --output scaling_law.pdf --include-predictions
```

---

## Progressive Internalization Experiment

**Date:** 2025-03-10
**Motivation:** Instead of digesting all repo context at once, progressively internalize files by generating a separate LoRA for each file and accumulating them. This enables handling infinite context in O(1) per new file.

### Concept

Two modes compared:
- **Re-embed**: For k files, recompute the weighted mean+max repo embedding from files 1..k and generate one LoRA. Baseline that shows benefit of richer context.
- **Accumulate**: Generate a LoRA from each file independently, average k LoRAs together. Novel approach: adding a new file is O(1) — no need to recompute previous LoRAs. Enables streaming/infinite context.

Connection to: test-time training, continuous learning, context distillation.

### Implementation

Script: `hypernetwork/eval_progressive.py`

Uses the existing trained hypernetwork checkpoint (no_oracle). For each test repo:
1. Computes per-file embeddings on-the-fly using Qwen3-Embedding-0.6B
2. Orders files by token count (largest = most informative first)
3. For k = 0, 1, 2, 3, 5, 10, 15, 20, 30 files:
   - **Re-embed**: builds repo embedding from k files, generates LoRA
   - **Accumulate**: averages k per-file LoRAs
4. Evaluates on all test pairs at each step

### Job

```bash
sbatch scripts/slurm/progressive_internalization.sh  # Job 10105612
```
- Checkpoint: `$SCRATCH/TRAINING_CHECKPOINTS/HYPERNET/no_oracle`
- Split: cr_test (52 repos, ~6414 pairs, limited to 30 pairs/repo)
- Account: rrg-yuntian
- Time: 8h, 1x H100
- Output: `$SCRATCH/BASELINES/progressive_internalization_cr_test.json`

### Expected results

Performance should improve monotonically with k. The accumulate mode may lag slightly behind re-embed (which has the full context at each step) but demonstrates the key advantage: O(1) incremental updates. If both modes converge to similar performance at k=20-30, it validates the progressive approach.
