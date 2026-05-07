# Agent Handoff — RepoPeft / Code2LoRA (EMNLP submission)

Short briefing so a new agent can pick up the work. For the full plan see `PLAN.md`.

## Goal

EMNLP "vision paper" on hypernetwork-based repository-aware LLM adaptation.
Task: Python assertion completion on `RepoPeftBench` (52 cross-repo / 409 in-repo
held-out repos) with a third **OOD test suite** (92 post-cutoff repos).

## Model family

* **Code2LoRA (direct)** — hypernetwork takes a static repo embedding (Qwen3-Embedding mean+max) and projects it to per-layer LoRA weights for Qwen2.5-Coder-1.5B (q/k/v/o + gate/up/down\_proj). ≈340M trainable. (`hypernetwork/hypernetwork.py`, ckpt `$CKPT_DIR/HYPERNET/no_oracle/`).
* **Code2LoRA-GRU<sub>file</sub>** (static encoder) — walks chronologically-ordered file embeddings through a GRU; PAW-style shared-basis generator, ≈33M trainable. Mamba2 preamble warm-start. Paper headline EM = **64.4%** CR-test. (`hypernetwork/code2lora_gru.py`, ckpt `$CKPT_DIR/CODE2LORA_GRU/mamba2_h1024_bptt32_chronological_default/`).
* **Code2LoRA-GRU<sub>commit</sub>** (streaming encoder) — same generator, walks per-commit `production_code_diff` embeddings. $O(1)$ per-commit update after `h_T`. Trained on `commit_parquet_hf/`. **Preliminary checkpoint: 100 repos, 1 epoch, `--max-assertions-per-commit 8`** at `$CKPT_DIR/CODE2LORA_GRU/commit_level_h100_1ep_train100_cached/code2lora_gru_best.pt`.

## Datasets

| location | what |
|---|---|
| `$SCRATCH/REPO_DATASET/{train,cr_test,ir_test}.json` | static RepoPeftBench (409 / 52 / 409 repos) — used by all baselines |
| `$SCRATCH/REPO_DATASET/{gru_cr_test,gru_ir_test}.json` | same QnAs + `embedding`, `file_embeddings`, `commit_history` (used by GRU<sub>file</sub>) |
| `$SCRATCH/REPO_DATASET/commit_parquet_hf/` | HF-layout parquet (400 train / 49 cr_val / 51 cr_test repos) — used by GRU<sub>commit</sub>. **9 train + 1 cr_test repos missing** vs. JSON splits, plus an extra 49-repo cr_val slice (see `scripts/slurm/missing_parquet_v2_repos.txt`). |
| `$SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/` | Smart-capped snapshot of the above (5/6/26). Same 400 / 49 / 51 repos. `commits/`, `splits/`, `cr_val.parquet`, `cr_test.parquet` are symlinked to the canonical parquet; only `qna/train.parquet` is rewritten — **215 129 train QnAs (8.1 % of 2.67 M)** after a trivial-target filter (drops 550 k bare `)` / `,` rows) and round-robin per (test_file, test_function) capping (`max_per_file=4, max_per_commit=8`). All 400 repos retained, 45 516 / 46 728 commits keep ≥1 QnA. Provenance in `SMART_CAP_README.json`. Build: `scripts/slurm/build_smart_capped_qna.sh` (cpu, 64 GB, ~4 min). |
| `$SCRATCH/REPO_DATASET/commit_parquet_ood/` | OOD parquet, concat layout, 92 repos, 1,950 commits, 181,500 raw QnAs (deduped to 9,942 in `ood_test.json`). Cutoff `2025-04-01`. |
| `$SCRATCH/REPO_DATASET/{ood_test,gru_ood_test}.json` | flattened from OOD parquet by `evaluation/build_ood_bench_json.py` (cap 120 pairs/repo, 32K-char prefix cap) |

## Evaluation

Single source of truth: **`evaluation/run_repopeft_bench.py`** (unified driver, 5K-bootstrap CIs).
Launcher: **`scripts/slurm/eval_repopeft_bench.sh`** with envs:
* `METHOD=direct|gru_file|gru_commit`, `CHECKPOINT=...`
* `SUITES="cr ir ood"` (a la carte, default all three) — added to dodge OOM resumes
* `--parquet-prefer auto` (concat → hf → shards), recently fixed from `hf` default
Baselines (pretrained / FFT / sLoRA / RAG / ICL / DRC / T2L) use their existing `baselines/*/test_*.py` with `--split ood_test` etc., wrapped by `scripts/slurm/eval_baselines_ood.sh`.

## Real numbers in hand (5/6 13:45)

| method | CR-test | IR-test | OOD-test |
|---|---|---|---|
| Pretrained | 45.71 | 46.82 | 45.63 |
| FFT | 51.42 | 55.88 | 58.12 (↑) |
| sLoRA | 45.62 | 48.16 | 34.08 (↓) |
| GRU<sub>file</sub> | **64.4** | 66.4 | TBD (needs OOD per-file embeddings) |
| direct \method | 63.8 | 66.2 | TBD (needs OOD per-repo embedding) |
| GRU<sub>commit</sub> (100-rep prelim) | 49.19 [48.0, 50.5] | — (OOM) | running (job 13393367) |

**Important caveat on FFT/sLoRA OOD deltas**: the OOD bank uses commit-parquet-style prefixes (median 7.9 KB) while cr_test JSONs use ~0.9 KB prefixes. Direct CR→OOD deltas mix prefix-shape effects with generalization. Cleanest fix: rebuild OOD bench with cr_test-style short prefixes. Marked as a caveat in `RepoPeft_Paper/text/new.tex` `tab:ood_results`.

## In-flight jobs

| jobid | what | ETA |
|---|---|---|
| 13383516 | `precompute_gru_cache` for full-scale retrain | RUNNING ~3h22m |
| 13383517 | full-scale GRU<sub>commit</sub> retrain (409 repos / 5 ep / MAX_SEQ_LEN=4096) | PENDING (dep on above), 72h budget |
| 13393367 | GRU<sub>commit</sub> OOD-only eval, 192 GB, after parquet-prefer fix | just submitted |

## Open issues / next steps (priority order)

1. **OOM in unified driver IR-test pass** for GRU<sub>commit</sub>: `load_commit_sequences_from_parquet` materializes all train-split repos upfront (>128 GB). Switch to per-repo streaming before the 409-repo retrain finishes, otherwise the camera-ready eval will OOM at scale. (`evaluation/run_repopeft_bench.py:533`)
2. **Apples-to-apples filter**: GRU<sub>commit</sub> eval is on 51/52 cr_test (1 missing from parquet) and ≤400/409 ir_test (9 missing). Other methods are on the full set. Add a `--restrict-to-method-supported-repos` filter so all rows in Table 1 score the same intersection.
3. **OOD per-repo embeddings**: needed to evaluate direct \method, GRU<sub>file</sub>, T2L-code, ICL on OOD. One pass of `embed_repos/4_construct_embeddings.py` over `$SCRATCH/REPO_DATASET/repositories_ood/`.
4. **OOD bench with short prefixes** (optional but reviewer-friendly): rebuild from `commit_parquet_ood` constraining prefix to ~1 KB (cr_test-style) so OOD-vs-CR Δ is interpretable.
5. **cr_val disclosure**: the 49-repo `cr_val` split exists in the parquet but not the JSONs and is not yet documented in §4 of the paper.
6. **Refresh paper numbers**: replace TBDs in `RepoPeft_Paper/text/new.tex` (`tab:main_results`, `tab:ood_results`) with the real numbers above; the GRU<sub>commit</sub> rows should keep the $^\S$ "preliminary 100-repo / 1-epoch" footnote until the full-scale checkpoint lands.

## Smart-cap GRU<sub>commit</sub> training subset (5/6/26)

The full `commit_parquet_hf` train.parquet has 2.67 M QnAs over 46 728 commits;
the existing trainer cap (`--max-assertions-per-commit 8`) is *file-blind* —
a 200-assertion `tests/test_a.py` change starves a co-changed `test_b.py`.

`create_dataset/build_smart_capped_qna_parquet.py` rewrites the train
parquet with three filters (`val` / `test` rows pass through verbatim so
cross-eval is unchanged):

1. **Quality**: drop `len(target.strip()) < 4` and bare `)` / leading-comma
   targets — removes 550 621 / 2.67 M rows (20.6 %).
2. **Per-(file, function) round-robin**: interleave samples across
   `(test_file, test_function)` groups; max 4 per file.
3. **Per-commit cap M = 8** (`max_per_commit`).

Result: 215 129 train QnAs (8.1 %), 400 / 400 repos retained, 45 516 / 46 728
commits keep ≥1 QnA (97.4 %), 80.1 % of unique `(commit, file)` pairs preserved.

Use it by setting `USE_SMARTCAP=1` when launching
`scripts/slurm/train_code2lora_gru_commits.sh`. The trainer's existing
`--max-assertions-per-commit 8` becomes a no-op safety net (every commit
in the snapshot already has ≤8 rows). The GRU-commit cache is keyed off
the parquet mtime, so it rebuilds automatically into
`$SCRATCH/REPO_DATASET/commit_cache_smartcap/`.

Quality artifacts:
* `analysis/output/smart_cap_qna/train_in=train.json` — distribution stats
  + cap-strategy simulation (random-K vs per-file-K vs combined).
* `analysis/output/smart_cap_qna_quality/train_in=train_summary.json` —
  pre/post comparison (per-commit / per-file dist, top assertion types,
  repo coverage).
* `analysis/output/smart_cap_qna_quality/train_in=train_samples.txt` — 30
  random capped QnAs for human eyeballing (reservoir-sampled).

## Key files

* Paper draft: `RepoPeft_Paper/text/new.tex` (Tables 1 & 6, §3.7 static-vs-streaming, §6 deployment scenarios, §7 limitations)
* Unified eval driver: `evaluation/run_repopeft_bench.py`
* OOD bench builder: `evaluation/build_ood_bench_json.py`
* Bootstrap CI util: `evaluation/metrics.py:bootstrap_ci, aggregate_metrics_with_ci`
* GRU<sub>commit</sub> training: `hypernetwork/train_code2lora_gru_commits.py` + `scripts/slurm/train_code2lora_gru_commits.sh` (set `USE_SMARTCAP=1` to train on the 215 k-row snapshot)
* Smart-cap snapshot builder: `create_dataset/build_smart_capped_qna_parquet.py` + `scripts/slurm/build_smart_capped_qna.sh`
* Smart-cap distribution analysis: `analysis/smart_cap_qna_analysis.py`
* Smart-cap quality check (pre/post comparison + sampled QnAs): `analysis/smart_cap_quality_check.py`
* GRU<sub>commit</sub> eval (separate, has the n=0 fix): `hypernetwork/eval_code2lora_gru_commits_metrics.py`
* Plotting: `analysis/plot_commit_chronology.py` (multi-source: canonical + OOD, see `--extra-parquet-dir`/`--extra-repos-root`/`--extra-source-label` and `--sort source_then_name`)
* T2L sanity: `baselines/text2lora/diagnose_t2l_injection.py` (confirmed LoRA injection works; low EM is conditioning-strategy limitation, not bug)
* pLoRA data sparsity: `analysis/per_repo_data_size_vs_em.py` (justifies N/A entry for pLoRA+DRC)
* Exec pilot: `evaluation/exec_pilot.py` + `scripts/exec_pilot_slice.json`
* Plan / detailed roadmap: `PLAN.md`

## Conventions

* Always source `scripts/slurm/common.sh` first to activate `qwen-cu126-py312` venv and set `$SCRATCH`, `$CKPT_DIR`, `$BASELINES_DIR`, `$SPLITS_DIR`.
* Memory in sbatch: prefer `--mem=192G` for any GRU<sub>commit</sub> eval that touches the train cross-repo split.
* Don't push generated figures (the SVG is 222 MB). Use `.gitignore` for `analysis/output/**.svg` if not already.
