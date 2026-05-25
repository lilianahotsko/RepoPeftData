# V2 Experiments — Handover Runbook

This document gives the **exact commands** needed to finish the V2 experiments
for the paper. It assumes the colleague has SSH access to the same cluster
(Compute Canada / DRAC, login node with SLURM and a `def-yuntian` allocation),
read+write access to the repo `~/RepoPeftData`, and a working `$SCRATCH`.

Estimated total wall-time on a single H100: roughly **2–3 weeks** if FFT V2
and Code2LoRA-GRU V2 are retrained from scratch; **3–5 days** if you reuse the
current partial checkpoints.

---

## 0 · One-time setup

### 0.1 Repo + scratch layout (already in place)

```bash
# Repo (read+write):
cd /home/lhotsko/RepoPeftData

# All env vars are set automatically by:
source scripts/slurm/common.sh
# After sourcing you'll have:
#   $SCRATCH          (your scratch quota; provided by the cluster)
#   $SPLITS_DIR       = $SCRATCH/REPO_DATASET
#   $BASELINES_DIR    = $SCRATCH/BASELINES
#   $CKPT_DIR         = $SCRATCH/TRAINING_CHECKPOINTS
# and the Python venv activated:
#   $SCRATCH/venvs/qwen-cu126-py312
```

> **Important:** every SLURM script in `scripts/slurm/` sources `common.sh`
> itself, so when you submit via `sbatch` you do **not** need to source it
> first. You only need to source it manually when running interactive
> Python (e.g. the merge step at the end of section 4).

### 0.2 Verify the V2 datasets are present

These were produced once and live in `$SCRATCH/REPO_DATASET/`. Verify with:

```bash
source scripts/slurm/common.sh

# V2 snapshot dataset (for the static / direct-projection trainer & most eval suites)
ls $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/commits/
ls $SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/
# Expect: train.parquet ir_val.parquet ir_test.parquet cr_val.parquet cr_test.parquet
#         in BOTH commits/ and qna/.

# V2 per-commit dataset (for the GRU trainer)
ls $SCRATCH/REPO_DATASET/commit_parquet_hf_v2/commits/
# Same five splits.

# Smart-capped QnA dataset (used for the GRU training loss only)
ls $SCRATCH/REPO_DATASET/commit_parquet_hf_smartcap/qna/train.parquet

# OOD (post-cutoff) commit dataset
ls $SCRATCH/REPO_DATASET/commit_parquet_ood/
# Expect: commits.parquet  qna_pairs.parquet  _done.*.jsonl

# OOD bundle (with per-suite repo ID list, optional)
ls $SCRATCH/REPO_DATASET/ood_bundle/
```

If any of these are missing, do NOT proceed — open a ticket with the original
author. They were produced by the scripts under `scripts/slurm/build_*.sh` and
take days to regenerate.

### 0.3 Verify existing V2 checkpoints

```bash
ls $CKPT_DIR/CODE2LORA_STATIC/           # h100_v2_static_3ep (likely empty — see §1)
ls $CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/   # gru_head.best.pt, gru_head.latest.pt
ls $CKPT_DIR/FFT_V2/h100_v2/              # checkpoint-2000 .. checkpoint-64000 (partial)
ls $CKPT_DIR/SLORA_V2/h100_v2/            # adapter-2000 .. adapter-24000 (looks complete)
ls $CKPT_DIR/BASELINES_V2/                # pretrained_h100_v2, slora_h100_v2_a24000_sharded
ls $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_best_sharded/
```

---

## 1 · Finish the in-flight trainings

### 1.1 Code2LoRA-direct (static V2)

**Status:** the previous run errored on `retain_graph=True`. The trainer is
fixed. A job is already queued (`squeue -u $USER` to confirm). If it
disappears without producing `head.best.pt`, resubmit:

```bash
sbatch scripts/slurm/train_code2lora_static_v2.sh
```

Expect ~3h on one H100 (3 epochs, max-seq-len 8192). Output:

```
$CKPT_DIR/CODE2LORA_STATIC/h100_v2_static_3ep/
    head.best.pt   head.latest.pt   head.ep0.pt   head.ep1.pt   head.ep2.pt
    metrics.jsonl
```

Check it finished cleanly:

```bash
ls $CKPT_DIR/CODE2LORA_STATIC/h100_v2_static_3ep/head.best.pt
tail $CKPT_DIR/CODE2LORA_STATIC/h100_v2_static_3ep/metrics.jsonl
```

### 1.2 FFT V2 (full fine-tuning, V2 snapshots)

**Status:** previous run was killed at the SLURM wall-time after
**step 64950 / 74210 of epoch 0** (i.e. ~87 % of *one* epoch out of three).
Continue from `checkpoint-64000`:

```bash
# Resume from the latest checkpoint. Adjust --time if your QoS allows >36h.
sbatch --time=36:00:00 \
    --export=ALL,RESUME_FROM=$CKPT_DIR/FFT_V2/h100_v2/checkpoint-64000 \
    scripts/slurm/train_fft_v2.sh
```

(Re-read `scripts/slurm/train_fft_v2.sh` first to confirm it reads
`$RESUME_FROM`; if not, edit it to call `python ... --resume_from_checkpoint
$RESUME_FROM` and resubmit.) Realistically this needs **2–3 more 36-hour
sessions** to reach the end of epoch 3 — chain them by running this same
command again as soon as each job ends.

### 1.3 Code2LoRA-GRU V2 (recommended retrain)
For a clean 3-epoch result:

```bash
sbatch scripts/slurm/train_code2lora_gru_v2.sh
```

Expect ~24–36 h depending on QoS. Output replaces
`$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/`. If you want to **keep the current
checkpoint** while running a parallel longer training, rename the output
suffix first:

```bash
SUFFIX=h100_v2_gru_3ep_v2 sbatch scripts/slurm/train_code2lora_gru_v2.sh
```

---

## 2 · Fill the missing shards in existing eval directories

These three shards just need to be re-run; the rest are already on disk. All
three are SLURM array jobs that resume automatically per-(repo, commit), so
re-launching the **full** array is safe (already-completed shards skip
themselves at output-file-existence check).

### 2.1 sLoRA V2 — CR-test shards 1, 2, 3 of 4

```bash
METHOD=slora \
CKPT=$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000 \
SUFFIX=h100_v2_a24000_sharded \
NUM_SHARDS=4 SUITES="cr_test" \
sbatch --array=1-3 scripts/slurm/eval_baselines_v2_sharded.sh
```

### 2.2 sLoRA V2 — IR-test shard 3 of 4

```bash
METHOD=slora \
CKPT=$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000 \
SUFFIX=h100_v2_a24000_sharded \
NUM_SHARDS=4 SUITES="ir_test" \
sbatch --array=3 scripts/slurm/eval_baselines_v2_sharded.sh
```

### 2.3 Code2LoRA-GRU V2 — CR-test shard 1 of 4

```bash
CKPT=$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/gru_head.best.pt \
SUFFIX=h100_v2_gru_3ep_best_sharded \
NUM_SHARDS=4 SUITES="cr_test" \
sbatch --array=1 scripts/slurm/eval_code2lora_gru_v2_sharded.sh
```

> **Tip:** for any sharded eval below, you can launch a single test shard
> first (`--array=0`) to confirm the output JSON looks sane, then launch
> the full `--array=0-3` (or `0-7`, etc.) once it works.

---

## 3 · New in-distribution evaluations (Table 3)

### 3.1 Code2LoRA-GRU V2 — IR-val + IR-test (8 array tasks)

```bash
CKPT=$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/gru_head.best.pt \
SUFFIX=h100_v2_gru_3ep_best_sharded \
NUM_SHARDS=4 SUITES="ir_val ir_test" \
sbatch --array=0-7 scripts/slurm/eval_code2lora_gru_v2_sharded.sh
```

### 3.2 Code2LoRA-direct V2 — CR-test + IR-test (8 array tasks)

The sharded driver now exists: `evaluation/run_code2lora_static_v2_eval.py`
+ `scripts/slurm/eval_code2lora_static_v2_sharded.sh`. It is the
direct-projection counterpart to the GRU V2 sharded driver — same output
schema (`summary` / `per_commit` / `raw_samples`), same atomic per-snapshot
writes, same resume logic. Each (repo, snapshot) row is scored
independently using `ctx = repo_state_embedding` (no GRU rollout).

```bash
CKPT=$CKPT_DIR/CODE2LORA_STATIC_V2/<run>/head.best.pt \
SUFFIX=h100_v2_static_3ep_best_sharded \
NUM_SHARDS=4 SUITES="cr_test ir_test" \
sbatch --array=0-7 scripts/slurm/eval_code2lora_static_v2_sharded.sh
```

Add `ir_val cr_val` to `SUITES` and bump `--array` accordingly if you also
want the val metrics. Shards write to
`$CKPT_DIR/CODE2LORA_STATIC_EVAL_V2/$SUFFIX/static_v2_<suite>_shard*of*.json`
and are merged the usual way:

```bash
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/CODE2LORA_STATIC_EVAL_V2/$SUFFIX
```

### 3.3 FFT V2 — CR-test + IR-test (8 array tasks)

Run **after** §1.2 finishes training. Pick the highest `checkpoint-*` that
exists at the end of epoch 3.

```bash
METHOD=fft \
CKPT=$CKPT_DIR/FFT_V2/h100_v2/checkpoint-XXX \
SUFFIX=h100_v2_sharded \
NUM_SHARDS=4 SUITES="cr_test ir_test" \
sbatch --array=0-7 scripts/slurm/eval_baselines_v2_sharded.sh
```

---

## 4 · OOD evaluations (Table 4)

The current paper OOD numbers come from a mix of legacy (V1) checkpoints. For
the V2 story they should be re-run against the V2 OOD parquet. Outputs land
under `$CKPT_DIR/BASELINES_V2/<method>_h100_v2_ood_sharded/` (and the GRU
analogue).

```bash
OOD_QNA_DIR="$SCRATCH/REPO_DATASET/commit_parquet_ood"   # uses qna_pairs.parquet
# (Confirm the QnA file your local run_baselines_v2.py expects — see
# DEFAULT_QNA_DIR inside the script. If the schema differs, regenerate it
# from ood_bundle/.)
```

### 4.1 Pretrained V2 OOD

```bash
METHOD=pretrained \
QNA_DIR="$OOD_QNA_DIR" \
SUFFIX=h100_v2_ood_sharded \
NUM_SHARDS=4 SUITES="ood_test" \
sbatch --array=0-3 scripts/slurm/eval_baselines_v2_sharded.sh
```

### 4.2 FFT V2 OOD

```bash
METHOD=fft \
CKPT=$CKPT_DIR/FFT_V2/h100_v2/checkpoint-XXX \
QNA_DIR="$OOD_QNA_DIR" \
SUFFIX=h100_v2_ood_sharded \
NUM_SHARDS=4 SUITES="ood_test" \
sbatch --array=0-3 scripts/slurm/eval_baselines_v2_sharded.sh
```

### 4.3 sLoRA V2 OOD

```bash
METHOD=slora \
CKPT=$CKPT_DIR/SLORA_V2/h100_v2/adapter-24000 \
QNA_DIR="$OOD_QNA_DIR" \
SUFFIX=h100_v2_ood_sharded \
NUM_SHARDS=4 SUITES="ood_test" \
sbatch --array=0-3 scripts/slurm/eval_baselines_v2_sharded.sh
```

### 4.4 Code2LoRA-direct V2 OOD

The sharded driver from §3.2 already exists. Point it at the OOD parquet
(the static driver reads `--snapshots-dir/{commits,qna}/<suite>.parquet`,
so if your OOD bundle uses a different layout you may need to symlink or
adapt the path).

```bash
CKPT=$CKPT_DIR/CODE2LORA_STATIC_V2/<run>/head.best.pt \
SNAPSHOTS_DIR="$OOD_QNA_DIR" \
SUFFIX=h100_v2_static_ood_sharded \
NUM_SHARDS=4 SUITES="ood_test" \
sbatch --array=0-3 scripts/slurm/eval_code2lora_static_v2_sharded.sh
```

### 4.5 Code2LoRA-GRU V2 OOD

```bash
CKPT=$CKPT_DIR/CODE2LORA_GRU/h100_v2_gru_3ep/gru_head.best.pt \
COMMITS_DIR=$SCRATCH/REPO_DATASET/commit_parquet_ood \
QNAS_DIR=$SCRATCH/REPO_DATASET/commit_parquet_ood \
SUFFIX=h100_v2_gru_3ep_best_ood_sharded \
NUM_SHARDS=4 SUITES="ood_test" \
sbatch --array=0-3 scripts/slurm/eval_code2lora_gru_v2_sharded.sh
```

> The GRU eval driver expects per-suite files `commits/<suite>.parquet` and
> `qna/<suite>.parquet`. The OOD parquet is laid out as a single
> `commits.parquet` + `qna_pairs.parquet`. You may need to either (i)
> symlink/copy them into a `commits/ood_test.parquet` + `qna/ood_test.parquet`
> layout, or (ii) extend the driver to accept the flat layout. Easiest:

```bash
mkdir -p $SCRATCH/REPO_DATASET/commit_parquet_ood_hf/{commits,qna}
ln -sf $SCRATCH/REPO_DATASET/commit_parquet_ood/commits.parquet      $SCRATCH/REPO_DATASET/commit_parquet_ood_hf/commits/ood_test.parquet
ln -sf $SCRATCH/REPO_DATASET/commit_parquet_ood/qna_pairs.parquet    $SCRATCH/REPO_DATASET/commit_parquet_ood_hf/qna/ood_test.parquet
# then point COMMITS_DIR / QNAS_DIR at commit_parquet_ood_hf
```

---

## 5 · Merge sharded outputs into final JSONs

After **every** sharded eval array finishes, merge its shards into a single
suite-level JSON the paper can quote. Activate the venv first (the merger
imports numpy):

```bash
source scripts/slurm/common.sh
```

Then for each output dir:

```bash
# Code2LoRA-GRU V2 (CR + IR)
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_best_sharded

# sLoRA V2 (CR + IR)
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/BASELINES_V2/slora_h100_v2_a24000_sharded

# FFT V2 (after §3.3)
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/BASELINES_V2/fft_h100_v2_sharded

# Code2LoRA-direct V2 (after §3.2)
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/CODE2LORA_STATIC_EVAL_V2/h100_v2_static_3ep_best_sharded

# OOD variants — one per directory used in §4.
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/BASELINES_V2/pretrained_h100_v2_ood_sharded
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/BASELINES_V2/fft_h100_v2_ood_sharded
python evaluation/merge_eval_shards.py --auto-detect \
    --input-dir $CKPT_DIR/BASELINES_V2/slora_h100_v2_ood_sharded
# ...etc.
```

The merger writes `baseline_<method>_<suite>.json` (or
`gru_v2_<suite>.json`) — the un-sharded files the paper text references.

---

## 6 · Mapping JSON outputs → paper tables

Once §1–5 are done, the numbers go into:

| Result | JSON file | Table |
|---|---|---|
| Pretrained V2 | `BASELINES_V2/pretrained_h100_v2/baseline_pretrained_{cr,ir}_test.json` | Table 3 |
| FFT V2 | `BASELINES_V2/fft_h100_v2_sharded/baseline_fft_{cr,ir}_test.json` | Table 3 |
| sLoRA V2 | `BASELINES_V2/slora_h100_v2_a24000_sharded/baseline_slora_{cr,ir}_test.json` | Table 3 |
| Code2LoRA-direct V2 | (new dir from §3.2) | Table 3 |
| Code2LoRA-GRU V2 | `CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_best_sharded/gru_v2_{cr,ir}_test.json` | Table 3 |
| OOD V2 (all 5 methods) | `*_h100_v2_ood_sharded/*` (§4) | Table 4 |

Each JSON contains a top-level `summary` with EM (`em_pct`), EditSim
(`edit_similarity`), CodeBLEU (`code_bleu`) and 95 % bootstrap CIs. Drop those
into `RepoPeft_Paper/text/new.tex` (Tables 2, 3, 4).

---

## 7 · Order of operations (suggested)

1. (Already queued) **`train_code2lora_static_v2.sh`** — wait for it.
2. **`train_fft_v2.sh`** — resubmit from checkpoint-64000; chain 2–3 times.
3. While (1) and (2) cook, fill in the **missing shards** (§2.1, §2.2, §2.3) —
   small fast jobs.
4. Submit **Code2LoRA-GRU V2 IR-val/IR-test** (§3.1) — independent of (1)/(2).
5. Submit the **sharded static V2 eval** (§3.2) once
   `train_code2lora_static_v2.sh` finishes.
6. After (2) finishes, submit **FFT V2 CR/IR-test** (§3.3).
7. Submit all **OOD evals** (§4) in parallel — they don't depend on each
   other.
8. **Merge** every output dir with the snippet in §5.
9. Copy summary numbers into `RepoPeft_Paper/text/new.tex` Tables 2/3/4.
10. **Optionally** kick off `Code2LoRA-GRU V2 full retrain` (§1.3) — only
    needed if the reviewer pushes back on the partial-epoch checkpoint.

---

## 8 · Monitoring

```bash
# Live queue:
squeue -u $USER

# Recent logs (newest first):
ls -t slurm_logs/ | head -20

# Tail a running array task:
tail -F slurm_logs/eval_baseline_v2_sh_<JOBID>_<TASKID>.out

# Per-shard JSONs being produced live:
ls -lt $CKPT_DIR/BASELINES_V2/<run>/*.json | head -10
ls -lt $CKPT_DIR/CODE2LORA_GRU_EVAL_V2/<run>/*.json | head -10
```

If a job dies mid-shard, re-launch the **same array element**. The Python
driver writes after every `(repo, commit)`, so resume is automatic — already-
scored items are skipped.

---

## 9 · Known gotchas

- **Wall-time** for FFT V2 and the GRU V2 retrain is at the limit of one
  H100 + 36 h job. Either request longer-time QoS or chain jobs (`--dependency=afterok:<JOBID>`).
- The `--time` field in `train_code2lora_static_v2.sh` is `03:00:00`. That
  was sized for the V2 dataset; do **not** raise it unless you see a
  time-limit cancellation in `slurm_logs/train_c2l_static_v2_*.err`.
- The numpy/`code2lora_core` import path requires the venv at
  `$SCRATCH/venvs/qwen-cu126-py312`. If that venv disappears, recreate it
  with `pip install -r requirements.txt` plus `peft`, `pyarrow`,
  `flash-attn` (cu126) — see `requirements.txt` in the repo root.
- `evaluation/run_baselines_v2.py` currently does **not** accept
  `code2lora_direct` as a `--method`. The sharded static V2 driver
  (§3.2) is the one missing piece of infrastructure for finishing the
  paper.

---

## 10 · Quick sanity-check commands (read-only)

Run these before submitting anything to confirm the environment is healthy:

```bash
source scripts/slurm/common.sh
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python -c "from hypernetwork.code2lora_core import Code2LoRAHead, CommitGRU; print('imports OK')"
python -c "import pyarrow.parquet as pq; print(pq.read_metadata('$SCRATCH/REPO_DATASET/code2lora_snapshots_hf/qna/cr_test.parquet').num_rows, 'cr_test QnA rows')"
sbatch --test-only scripts/slurm/eval_baselines_v2_sharded.sh
```

If all four succeed, you're good to go.
