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

| Method | CR EM | CR ES | CR CB | IR EM | IR ES | IR CB | Status |
|--------|-------|-------|-------|-------|-------|-------|--------|
| **Inference-only** | | | | | | | |
| Pretrained | 45.71% | 0.605 | 0.646† | 46.82% | 0.624 | 0.655† | Pending real CB |
| RAG (k=3) | 39.93% | 0.535 | 0.439 | 42.32% | 0.560 | 0.448 | Done |
| ICL (3-shot) | 42.22% | 0.559 | 0.470 | 45.35% | 0.596 | 0.482 | Done |
| Oracle Context | 47.47% | 0.614 | 0.484 | 48.74% | 0.627 | 0.486 | Done |
| **Trained** | | | | | | | |
| FFT | 51.42% | 0.696 | 0.678 | 55.88% | 0.727 | 0.714 | Done |
| sLoRA r=16 | ~45.0%‡ | ~0.645‡ | — | ~48.4%‡ | ~0.674‡ | — | Retrain: 10133136 |
| sLoRA r=64 | 36.03% | 0.575 | 0.580 | 39.20% | 0.616 | 0.610 | **Over-param** |
| Per-repo LoRA | — | — | — | 64.56% | 0.797 | 0.790 | 191/~380 repos |
| Code2LoRA Direct | **63.81%** | **0.784** | **0.777** | **66.24%** | **0.806** | **0.795** | Done |
| Code2LoRA PAW | **64.09%** | **0.786** | **0.777** | 65.84% | 0.804 | 0.790 | Done |
| **+Oracle** | | | | | | | |
| FFT + Oracle | — | — | — | — | — | — | Job 10133592 |
| sLoRA + Oracle | — | — | — | — | — | — | Job 10133591 |
| Code2LoRA + Oracle | — | — | — | — | — | — | Job 10133593 |
| Code2LoRA PAW + Oracle | — | — | — | — | — | — | Job 10133595 |

† Fallback BLEU, ‡ Previous r=16 model (adapter overwritten)

### Architecture Comparison (Paper Table)

| Variant | CR EM | IR EM | Mapper Params |
|---------|-------|-------|---------------|
| Direct-Projection | 63.81% | 66.24% | ~120M |
| Shared-Basis (PAW) | 64.09% | 65.84% | ~14M |

PAW achieves comparable performance with ~8.5x fewer mapper parameters.

### Scaling Experiments (Paper Table 13)

| Training Repos | CR Test EM (%) | EditSim | CodeBLEU |
|----------------|----------------|---------|----------|
| 10 (2%) | 57.73% | 0.732 | 0.732 |
| 25 (6%) | 60.88% | 0.760 | 0.753 |
| 50 (12%) | 60.88% | 0.760 | 0.755 |
| 100 (24%) | 61.27% | 0.756 | 0.754 |
| 200 (49%) | 62.24% | 0.773 | 0.765 |
| 409 (100%) | 63.81% | 0.784 | 0.777 |
| 500 (expanded)* | 61.18% | 0.767 | 0.759 |

*500 repos includes sparse repos (1-29 pairs), performance drops.
Pending: 150, 300, 623 repos.

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




- at least 1 commit between 2024-2025
- at least one before 2024 
- filter the ood by the end date 
- double check if the filtering criterion is the same for the regular repos and the ood 
- cap: all the evaluation (8 qnas per commit)

- reevaluate the code2lora- nongru version (retrain the hypernet on the cutoff, and evaluate on each next diffs)



===========


## Premise

The first-version EMNLP paper (results above) compared methods at a *single*
snapshot per repo. The new pitch: study how a generated LoRA's accuracy
**decays over the repository's commit history**, and demonstrate that a
commit-sequential hypernetwork (Code2LoRA-GRU) decays slower than a static
one (Code2LoRA-direct), with controlled OOD analysis.

## Work done so far

### Dataset
- [x] **Original Code2LoRA dataset** (`commit_parquet_hf`): 400 train +
  49 cr_val + 51 cr_test repos with per-commit `in_repo_split`
  (chronological 80/10/10) and `cross_repo_split` partitioning.
- [x] **OOD-original dataset** (92 repos) — first OOD run, kept as
  secondary comparison set.
- [x] **OOD-matched mining** (`scripts/slurm/mine_ood_matched.sh`):
  re-mined 146 repos using filters that match the training distribution
  (stars 50-2000, size 3000-15000 KB, MIT/Apache-2.0, pushed≥2025-01-01,
  100% `uses_pytest`). `created>2026-02-11` cutoff inferred from baseline.
- [x] **Smart-cap QnA parquet** (`commit_parquet_hf_smartcap`): per-commit
  cap (max 4/file, max 8/commit) used by the GRU training set.

### Static-commit snapshot embeddings
- [x] **Static-commit manifest** (`static_commit/manifest.tsv`):
  29,471 (repo, commit) rows with `anchor_sha`, `anchor_index`,
  `n_commits_after_anchor`. Roles: `train_snapshot` (400), `ir_test`
  (6,179), `ir_val` (5,710), `cr_test_{train,val,test}` (6,618 total),
  `cr_val_{train,val,test}` (8,614 total), `ood_test` (1,950).
- [x] **Per-blob Qwen3 embedding cache**: ~600 repos, ~24k unique snapshots,
  via blob-SHA dedup (~10× speedup over naive re-embedding).
- [x] **Flat 2048-d snapshot embeddings** (`snapshot_embeddings.json`,
  ≈1 GB, 23,761 entries) ready for both training and per-commit eval.

### Models
- [x] **Code2LoRA-GRU<sub>commit</sub> (smart-cap, epoch-0)** trained on
  the smart-cap parquet at `commit_level_h100_5ep_smartcap_pf4_pc8/
  snapshots/epoch0_for_eval/code2lora_gru_best.pt`.
- [x] **Code2LoRA-direct (legacy)** from EMNLP version retained as
  baseline.

### Evaluation infrastructure
- [x] **GRU per-commit eval driver**
  (`hypernetwork/eval_code2lora_gru_commits_metrics.py`): walks every
  commit, regenerates LoRA at each commit, scores held-out assertions.
  Supports `--timeline-mode all` (every commit) plus 5K-bootstrap CIs.
- [x] **Static per-commit eval driver**
  (`evaluation/eval_code2lora_static_per_commit.py`): mirrors the GRU
  driver's output schema; LoRA generated from the snapshot embedding at
  each commit. Same scoring helpers, same QnA set per commit → directly
  comparable curves.
- [x] **Unified decay-curve plotter**
  (`analysis/plot_per_commit_decay_all_suites.py`): overlays multiple
  bench JSONs with optional `--label` for static-vs-GRU side-by-side
  plots. Outputs both **absolute** (`n_commits_after_first_kept_commit`,
  symlog) and **normalized** (`position ∈ [0,1]`) decay figures.
- [x] **Data builders**:
  `create_dataset/build_static_commit_all_splits.py` — re-extracts QnAs
  at each (repo, commit) using the same v2 extractor that built the
  original Code2LoRA training data, so we capture every live assertion
  at each snapshot (~270 selected per repo on average; vs ~25 from the
  smart-cap GRU subset).

### Currently running / queued (May 11)
- [ ] **GRU per-commit eval** (job `13688463`, RUNNING, 15h+ elapsed):
  full `--timeline-mode=all` across `in_repo_val`, `in_repo_test`,
  `cross_repo_cr_val`, `cross_repo_cr_test`. ETA finish: mid-week.
- [ ] **Static-splits build** (4 parallel jobs `13747157-60`,
  RUNNING/PENDING): produces `ir_test.json`, `ood_test.json`,
  `ir_val.json`, `cr_val.json` (the `train.json` and `cr_test.json`
  files were already produced by the cancelled monolithic run).
- [ ] **Static Code2LoRA-direct training** (job `13747161`, waiting on
  splits): 5 epochs on the new at-anchor splits using
  `hypernetwork_sampled.py` with `cr_val.json` as early-stop signal.
- [ ] **Static per-commit eval** (job `13747162`, waiting on training):
  decay curves on all 5 suites in the same schema as the GRU eval.

## Still to do (in execution order)

### Immediate (depends on running jobs)
1. **Finish GRU per-commit eval** → produces
   `bench_per_commit_timeline_full.json` with per-commit metrics for
   IR-val, IR-test, CR-val, CR-test on the smart-cap epoch-0 ckpt.
2. **Verify the 4 parallel static-splits jobs complete** within their
   time budget; if not, shard further by repo range inside the suite.
3. **Run static Code2LoRA training** + per-commit eval (auto-chained).
4. **Build OOD-matched parquet** (`scripts/slurm/build_ood_matched_parquet.sh`)
   — clone 146 OOD-matched repos and run the same processing pipeline.
5. **Embed OOD-matched snapshots** via the existing manifest+embed
   script (will become a second `static_embed_*` SLURM batch).
6. **Run GRU per-commit eval on OOD-matched** using
   `scripts/slurm/eval_gru_commits_ood_matched.sh`.

### Decay-curve analyses (after evals)
7. **Overlay figure** static-vs-GRU EM%, EditSim, CodeBLEU vs.
   `n_commits_after_anchor` (paper-grade primary figure).
8. **Normalized-time figure** same metrics vs. `position ∈ [0,1]`.
9. **`n_files_changed_since_anchor` curve** — semantic-distance decay
   (run a one-shot post-hoc job that populates this metadata column;
   currently skipped to make the splits build fit in 12 h).
10. **Per-difficulty bucket decay** using v2's `difficulty` tag
    (easy/medium/hard) — separate curves to show that hard assertions
    decay faster than easy ones, and that GRU's advantage is largest
    on hard.
11. **Difficulty calibration check**: make sure OOD-original was not
    biased toward trivial targets (we already verified this is a
    confound of the original OOD).

### Baselines for the decay claim
12. **Base-LLM floor**: same per-commit eval but with no LoRA → shows
    how much the LoRA buys at each commit.
13. **Oracle per-commit ceiling** on ~10 demo repos: train one LoRA per
    commit (expensive but only on a small demo set) → upper bound on
    "what perfect adaptation would look like".

### Statistical rigour
14. **Macro-average bootstrap** over **repos** (not just assertions)
    with 5K bootstrap. Currently CIs are micro-averaged.
15. **Holm-Bonferroni multi-comparison correction** for static-vs-GRU
    at every x-bucket.

### Robustness / contamination
16. **Embedding-model contamination check**: split OOD-matched into
    `created>2026-02-11` (strictly post-cutoff) vs. the rest, compare
    decay curves. If identical, contamination is not driving results.
17. **Time-OOD as primary OOD story**: report the post-cutoff OOD
    slice as the headline robustness number.

### Paper assembly
18. Final decay-curve figure (single page-wide, 3 panels: EM, ES, CB,
    5 suites overlaid, GRU vs static).
19. Headline table: static vs GRU vs base-LLM × {IR-test, CR-test,
    OOD-matched, OOD-time} at three commit-distance buckets
    (`[0,1)`, `[5,25)`, `[100,∞)`).
20. Update `README.md` with the dataset card for the static-commit
    splits and the per-commit eval protocol.

## Job-ID cheatsheet (currently active)

| Job | Description | Status |
|---|---|---|
| `13688463` | GRU per-commit eval (4 suites, all-timeline) | RUNNING ~15 h |
| `13699664-67` | Static-commit embed shards (extras) | COMPLETED |
| `13747157` | Splits build (`ir_test`) | RUNNING |
| `13747158` | Splits build (`ood_test`) | RUNNING |
| `13747159` | Splits build (`ir_val`) | RUNNING |
| `13747160` | Splits build (`cr_val`) | RUNNING |
| `13747161` | Train Code2LoRA-direct (static-commit) | PENDING (dep) |
| `13747162` | Static per-commit eval | PENDING (dep) |

Run `bash scripts/status_static_pipeline.sh` for a live snapshot.
