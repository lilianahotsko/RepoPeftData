# Recreate the same Python environment (Compute Canada)

This matches the environment used for **Code2LoRA-GRU** / RepoPeftData work:
Python **3.12**, **CUDA 12.6**, **Torch 2.9.1**, **Transformers 4.57.6**, PyArrow
via the **`arrow` module** (not pip `pyarrow` wheels), plus the rest from the
Alliance wheel index (`--no-index`).

---

```bash
module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
```
---

```bash
export VENV_DIR=$SCRATCH/venvs/qwen-cu126-py312  
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -V    # should be Python 3.12.x
```

```bash
pip install --no-index --upgrade pip setuptools wheel
```

---

```bash
cd /path/to/RepoPeftData
pip install --no-index -r requirements-frozen-cc-gru.txt
```

```bash
pip install --no-index \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  transformers==4.57.6 tokenizers==0.22.2 \
  accelerate==1.12.0 peft==0.18.1 trl==0.27.0 \
  datasets==4.5.0 huggingface_hub==0.36.0 \
  numpy pandas matplotlib tqdm wandb bitsandbytes sentencepiece protobuf
pip install codebleu rouge-score  # from PyPI if not on --no-index
```

---

##  Wire SLURM scripts to the new venv + repo path

Edit **`scripts/slurm/common.sh`**:

```bash
module purge
module load StdEnv/2023 python/3.12 gcc/12.3 cuda/12.6 arrow
source $SCRATCH/venvs/qwen-cu126-py312/bin/activate 
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

cd $HOME/path/to/RepoPeftData

export SPLITS_DIR="$SCRATCH/REPO_DATASET"
export REPOS_ROOT="$SCRATCH/REPO_DATASET/repositories"
export BASELINES_DIR="$SCRATCH/BASELINES"
export CKPT_DIR="$SCRATCH/TRAINING_CHECKPOINTS"
```

---

sanity check

```bash
source scripts/slurm/common.sh 
python - <<'PY'
import torch, transformers, pyarrow as pa
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("pyarrow", pa.__version__)
PY
```