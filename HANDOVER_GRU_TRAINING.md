# Code2LoRA-GRU<sub>commit</sub> — Training & Evaluation Handover

This guide is the minimum that another person needs to (1) recreate the
environment, (2) pull the dataset from the HuggingFace Hub, (3) train the
GRU hypernetwork, and (4) run the per-commit evaluation that produces the
paper's decay-curve numbers. All instructions target the **Nibi**
cluster (or any SLURM cluster with H100 / A100 GPUs and Compute Canada's
StdEnv/2023 module stack).

---

## Repos and artifacts

| Asset | Location | Notes |
|---|---|---|
| Code | `github.com/lilianahotsko/RepoPeftData` | Branch: `main` |
| Dataset (parquet) | `huggingface.co/datasets/<HF_REPO_ID>` | Pushed once with `scripts/push_gru_dataset_to_hf.py`. ~14 GB. |
| Base LLM | `Qwen/Qwen2.5-Coder-1.5B` | Pulled from HF on first run. |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` | Pulled from HF on first run. |

> Replace `<HF_REPO_ID>` below with the actual repo (e.g.
> `nanigock/repopeft-gru-commits`).

---

## Step 0 — One-time on the *current* (Liliana's) cluster

Push the dataset once. Skip this if it's already on HF.

```bash
cd /home/lhotsko/RepoPeftData
source scripts/slurm/common.sh

# Log in to HF once (or set HF_TOKEN env var instead of interactive login).
huggingface-cli login

# Submit the push job. Takes ~1-2 h depending on network.
sbatch --export=ALL,HF_REPO_ID=nanigock/repopeft-gru-commits,HF_TOKEN=$(cat ~/.cache/huggingface/token) \
       scripts/slurm/push_gru_dataset_to_hf.sh
```

What gets uploaded:

| Path in the HF dataset | Source | Size |
|---|---|---|
| `commits/{train,cr_val,cr_test}.parquet` | base parquet | ~0.5 GB |
| `qna/train.parquet` | **smart-cap** filtered (paper-grade) | ~6.2 GB |
| `qna/{cr_val,cr_test}.parquet` | base parquet | ~7 GB |
| `splits/*.json` | repo-level split definitions | <1 MB |
| `README.md`, `SMART_CAP_README.json` | provenance & schema | <100 KB |

---

## Step 1 — Environment setup on the new cluster

**Canonical instructions (frozen `pip freeze` + modules):** see
[`ENV_COMPUTE_CANADA.md`](ENV_COMPUTE_CANADA.md) in this repo.

```bash
# Pick a workspace under your $SCRATCH (or $HOME if scratch isn't available).
export WORK_ROOT=$SCRATCH/RepoPeft
mkdir -p $WORK_ROOT && cd $WORK_ROOT

# Clone the code.
git clone https://github.com/lilianahotsko/RepoPeftData.git
cd RepoPeftData

# Load the module stack (Compute Canada modules; on other clusters use the
# equivalent CUDA 12.x + Python 3.12 + Arrow modules).
module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow

# Create the venv.
python -m venv $SCRATCH/venvs/qwen-cu126-py312
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate

# Pinned dependency set (matches the one used to produce the paper numbers).
pip install --no-index --upgrade pip
pip install --no-index \
    "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" \
    "transformers==4.57.6" "tokenizers==0.22.2" \
    "accelerate==1.12.0" "peft==0.18.1" "trl==0.27.0" \
    "datasets==4.5.0" "huggingface_hub==0.36.0" \
    "pyarrow" "pandas==3.0.0" "numpy==2.4.2" \
    "matplotlib==3.10.8" "tqdm" "wandb==0.21.2" \
    "bitsandbytes==0.49.0" "sentencepiece" "protobuf"

# Optional but useful: flash-attn if available on your cluster.
pip install --no-index "flash_attn==2.8.3+torch29.computecanada" || true

# Verify GPU sanity.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then **edit `scripts/slurm/common.sh`** for the new cluster:

```bash
# scripts/slurm/common.sh — change these two lines:
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate
cd $WORK_ROOT/RepoPeftData
# and update the partition / account in every #SBATCH header to match Nibi's
# account name (replace --account=def-yuntian).
```

---

## Step 2 — Pull the dataset from HF

```bash
cd $WORK_ROOT/RepoPeftData
source scripts/slurm/common.sh

# Download into the canonical layout. ~14 GB.
export HF_REPO_ID=nanigock/repopeft-gru-commits
export PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf

mkdir -p $PARQUET_DIR
huggingface-cli download $HF_REPO_ID \
    --repo-type dataset \
    --local-dir $PARQUET_DIR \
    --local-dir-use-symlinks False

# Sanity check.
ls $PARQUET_DIR/commits/
ls $PARQUET_DIR/qna/
python -c "
import pyarrow.parquet as pq
for split in ['train', 'cr_val', 'cr_test']:
    n = pq.read_table(f'$PARQUET_DIR/qna/{split}.parquet',
                      columns=['repo_id']).num_rows
    print(f'qna/{split}.parquet: {n:,} rows')
"
```

Expected output:

```
qna/train.parquet:    ~660K rows (smart-capped from 3.6M)
qna/cr_val.parquet:   ~642K rows
qna/cr_test.parquet:  ~476K rows
```

---

## Step 3 — Training

```bash
cd $WORK_ROOT/RepoPeftData
mkdir -p slurm_logs

# Paper-grade run: 5 epochs, smart-cap, max 8 assertions per commit, full
# 400 train repos. ~70 h on a single H100; <72 h time-limit fits.
sbatch \
    --export=ALL,USE_SMARTCAP=0,PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf,SUFFIX=h100_5ep_smartcap_pf4_pc8,EPOCHS=5,MAX_ASSERTIONS_PER_COMMIT=8 \
    scripts/slurm/train_code2lora_gru_commits.sh
```

> **Why `USE_SMARTCAP=0` even though we want the smart-cap data?** Because
> we already shipped the smart-cap QnA directly as `qna/train.parquet` in
> the HF dataset. The `USE_SMARTCAP=1` flag is only meaningful when both
> a base and a smart-cap parquet dir exist locally; on a fresh Nibi
> install we just point at the published (smart-cap) parquet directly.

Outputs (under `$SCRATCH/TRAINING_CHECKPOINTS/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/`):

| File | When | Notes |
|---|---|---|
| `code2lora_gru_best.pt` | every eval (~step 500) | best-val checkpoint |
| `code2lora_gru_last.pt` | end of training | final epoch ckpt |
| `snapshots/epoch{0..4}_for_eval/code2lora_gru_best.pt` | epoch boundaries | for paper-grade eval |

Useful knobs (all `--export=ALL,VAR=val`):

| Var | Default | What it controls |
|---|---|---|
| `EPOCHS` | 5 | training epochs |
| `MAX_ASSERTIONS_PER_COMMIT` | 8 | per-commit loss cap (matches smart-cap) |
| `LIMIT_TRAIN_REPOS` | 0 | smoke-test with e.g. `10` |
| `LIMIT_EVAL_REPOS` | 64 | val/test repos seen during in-training eval |
| `LR` | `1e-4` | learning rate |
| `GRAD_ACCUM` | 8 | effective batch = 8 (LM_MICRO_BATCH=1) |
| `EVAL_STEPS` | 500 | val frequency |
| `SAVE_STEPS` | 500 | ckpt frequency |
| `MAX_SEQ_LEN` | 4096 | LLM context cap |

---

## Step 4 — Evaluation (per-commit decay curves)

The headline experiment is a **per-commit** evaluation: for every
commit in every test repo, regenerate the LoRA from the GRU's hidden
state and score the held-out assertions visible at that commit.
This produces the decay-curve figure used in the paper.

```bash
# Pick the best checkpoint (or a snapshot at a specific epoch).
export CKPT=$SCRATCH/TRAINING_CHECKPOINTS/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/snapshots/epoch0_for_eval/code2lora_gru_best.pt
ls "$CKPT" || echo "Set CKPT to a real path before continuing."

# Full per-commit eval across the four canonical suites (no OOD here).
# 24-48 h on a single H100; the script writes incremental per-suite JSONs
# so partial results survive a timeout.
sbatch --export=ALL,CHECKPOINT=$CKPT \
       scripts/slurm/eval_gru_commits_full_timeline.sh
```

Output (next to the checkpoint):

```
bench_results/
    bench_per_commit_timeline_full.json                # full merged JSON
    bench_per_commit_timeline_full__in_repo_val.json   # per-suite sidecars
    bench_per_commit_timeline_full__in_repo_test.json
    bench_per_commit_timeline_full__cross_repo_cr_val.json
    bench_per_commit_timeline_full__cross_repo_cr_test.json
```

### Plot the decay curves

```bash
python analysis/plot_per_commit_decay_all_suites.py \
    --bench-result $(dirname $CKPT)/bench_results/bench_per_commit_timeline_full.json \
    --out-prefix    analysis/figures/gru_decay/gru_decay
```

Produces, under `analysis/figures/gru_decay/`:

- `gru_decay_absolute.{png,pdf}` — EM, EditSim, CodeBLEU vs
  `n_commits_after_first_kept_commit` (symlog axis).
- `gru_decay_normalized.{png,pdf}` — same vs `position ∈ [0,1]`.
- `gru_decay_*.json` — bucketed numeric values for paper figures.

---

## Step 5 — (Optional) OOD evaluation

If the OOD parquet has also been pushed (see `mine_ood_matched.sh`
+ `build_ood_parquet_from_mined_jsonl.py`), run the OOD per-commit
eval the same way:

```bash
sbatch --export=ALL,CHECKPOINT=$CKPT,OOD_PARQUET=$SCRATCH/REPO_DATASET/commit_parquet_ood_matched \
       scripts/slurm/eval_gru_commits_ood_matched.sh
```

---

## Job-list cheat-sheet

```bash
# 1. (one time, current cluster) push dataset to HF
sbatch --export=ALL,HF_REPO_ID=<owner>/<repo>,HF_TOKEN=$HF_TOKEN \
       scripts/slurm/push_gru_dataset_to_hf.sh

# 2. (on Nibi) train
sbatch --export=ALL,PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf,SUFFIX=h100_5ep_smartcap_pf4_pc8,EPOCHS=5,MAX_ASSERTIONS_PER_COMMIT=8 \
       scripts/slurm/train_code2lora_gru_commits.sh

# 3. (on Nibi) per-commit eval (in-repo + cross-repo)
sbatch --export=ALL,CHECKPOINT=$SCRATCH/TRAINING_CHECKPOINTS/CODE2LORA_GRU/commit_level_h100_5ep_smartcap_pf4_pc8/snapshots/epoch0_for_eval/code2lora_gru_best.pt \
       scripts/slurm/eval_gru_commits_full_timeline.sh

# 4. (on Nibi) plot the decay curves
python analysis/plot_per_commit_decay_all_suites.py \
       --bench-result <path-from-step-3>/bench_per_commit_timeline_full.json \
       --out-prefix analysis/figures/gru_decay/gru_decay
```

---

## Sanity smoke test (run this *before* the big training job)

```bash
# Tiny end-to-end: 5 train repos, 2 eval repos, 1 epoch — finishes in ~30 min.
sbatch --export=ALL,PARQUET_DIR=$SCRATCH/REPO_DATASET/commit_parquet_hf,SUFFIX=smoke,EPOCHS=1,LIMIT_TRAIN_REPOS=5,LIMIT_EVAL_REPOS=2,EVAL_STEPS=50,SAVE_STEPS=50 \
       scripts/slurm/train_code2lora_gru_commits.sh
```

If this finishes and you see val EM > 0, the full run can be launched
with confidence.

---

## Known gotchas

1. **HF dataset symlinks.** Don't pass `--local-dir-use-symlinks True`
   when downloading; the trainer reads parquet rows directly and
   pyarrow follows symlinks fine, but for portability we use plain
   files.
2. **Out-of-memory during dataset cache build.** The first training run
   builds a per-repo commit cache under
   `$SCRATCH/REPO_DATASET/commit_cache_smartcap`. If this OOMs, lower
   `MAX_SEQ_LEN` (e.g. 2048) for the first run, then bump it back to
   4096 once the cache is built.
3. **Reproducibility.** Seed is `--seed 3407` (set in the SLURM
   launcher). Don't change it if you want the paper-exact numbers.
4. **Wandb.** The trainer logs to wandb; if you don't have a wandb
   account, pass `WANDB_MODE=offline` in `--export=ALL,...`.
5. **`def-yuntian` account.** All SLURM headers use this account
   alias; replace with your Nibi account before submission.

---

## Contact

Open an issue at the GitHub repo, or ping Liliana Hotsko / the original
author for questions about the dataset construction logic.
