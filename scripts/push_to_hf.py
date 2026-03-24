#!/usr/bin/env python3
"""Push RepoPeft datasets, checkpoints, and results to HuggingFace Hub.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/push_to_hf.py --all
    python scripts/push_to_hf.py --all --dry-run
    python scripts/push_to_hf.py --datasets --checkpoints --results
    python scripts/push_to_hf.py --all --include-per-repo-lora
    python scripts/push_to_hf.py --all --hf-user myorg --private
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")
DATASET_ROOT = Path(SCRATCH) / "REPO_DATASET"
CKPT_ROOT = Path(SCRATCH) / "TRAINING_CHECKPOINTS"
BASELINES_ROOT = Path(SCRATCH) / "BASELINES"
ORACLE_CACHE = Path(SCRATCH) / "ORACLE_CONTEXT_CACHE_V2"
DEFAULT_USER = "nanigock"
DS_REPO = "RepoPeft-data"
CK_REPO = "RepoPeft-checkpoints"

DS_README = """\
---
license: mit
task_categories: [text-generation]
language: [code]
tags: [code-generation, repository-level, lora, hypernetwork]
pretty_name: RepoPeft Dataset
---
# RepoPeft Dataset
Repository-level code completion dataset for hypernetwork-based LoRA generation.
## Layout
```
splits/main/           # Original splits (train, cr_val, cr_test, ir_val, ir_test + structured)
splits/expanded/       # Expanded training (~609 repos)
oracle_context_cache/  # Pre-computed DRC context (370 MB, 512 repos)
evaluation_results/    # Pre-computed baseline JSONs
```
"""

CK_README = """\
---
license: mit
tags: [code-generation, lora, hypernetwork, repository-level]
---
# RepoPeft Checkpoints
## Layout
```
hypernet/no_oracle/hypernet_best.pt      # Main hypernetwork (~1.3 GB)
hypernet/full_repos/hypernet_best.pt     # Oracle variant (~1.3 GB)
hypernet/scale_{10..500}repos/           # Scaling checkpoints
hypernet_paw/no_oracle/                  # PAW variant (~450 MB)
single_lora/adapter/                     # Single LoRA (PEFT, ~86 MB)
fft/final/                               # Full fine-tuning (~12 GB)
```
"""

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-user", default=DEFAULT_USER)
    p.add_argument("--all", action="store_true")
    p.add_argument("--datasets", action="store_true")
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--results", action="store_true")
    p.add_argument("--include-per-repo-lora", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--private", action="store_true")
    return p.parse_args()

def mk(api, rid, rt, priv, dry):
    if dry: return
    create_repo(rid, repo_type=rt, private=priv, exist_ok=True, token=api.token)
    pfx = "datasets/" if rt == "dataset" else ""
    print(f"  Repo: https://huggingface.co/{pfx}{rid}")

def up(api, src, rid, dst, rt, dry):
    src = Path(src)
    if not src.exists():
        print(f"  SKIP (missing): {src}")
        return
    mb = src.stat().st_size / 1048576 if src.is_file() else sum(
        f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1048576
    tag = "DRY" if dry else "UP"
    print(f"  [{tag}] {dst}  ({mb:.1f} MB)")
    if dry: return
    if src.is_file():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=rid, repo_type=rt)
    else:
        api.upload_folder(folder_path=str(src), path_in_repo=dst,
                          repo_id=rid, repo_type=rt)

def up_readme(api, rid, rt, txt, dry):
    if dry: return
    tmp = Path(f"/tmp/_repopeft_readme_{rt}.md"); tmp.write_text(txt)
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md",
                    repo_id=rid, repo_type=rt)
    print("  README.md uploaded")

def push_ds(api, a):
    rid = f"{a.hf_user}/{DS_REPO}"
    print(f"\n{'='*60}\nDATASET: {rid}\n{'='*60}")
    mk(api, rid, "dataset", a.private, a.dry_run)
    mf = ["train.json","train_structured.json","cr_test.json","cr_test_structured.json",
          "cr_val.json","cr_val_structured.json","ir_test.json","ir_test_structured.json",
          "ir_val.json","ir_val_structured.json"]
    print("\n--- Main splits ---")
    for f in mf: up(api, DATASET_ROOT/f, rid, f"splits/main/{f}", "dataset", a.dry_run)
    ef = ["train.json","cr_test.json","cr_test_structured.json","cr_val.json",
          "cr_val_structured.json","ir_test.json","ir_val.json"]
    print("\n--- Expanded splits ---")
    for f in ef: up(api, DATASET_ROOT/"expanded"/f, rid, f"splits/expanded/{f}", "dataset", a.dry_run)
    print("\n--- Oracle context cache ---")
    if ORACLE_CACHE.exists():
        up(api, ORACLE_CACHE, rid, "oracle_context_cache", "dataset", a.dry_run)
    else:
        print(f"  SKIP (missing): {ORACLE_CACHE}")
    if a.results or a.all:
        print("\n--- Evaluation results ---")
        for f in sorted(BASELINES_ROOT.glob("*.json")):
            up(api, f, rid, f"evaluation_results/{f.name}", "dataset", a.dry_run)
    up_readme(api, rid, "dataset", DS_README, a.dry_run)

def push_ck(api, a):
    rid = f"{a.hf_user}/{CK_REPO}"
    print(f"\n{'='*60}\nCHECKPOINTS: {rid}\n{'='*60}")
    mk(api, rid, "model", a.private, a.dry_run)
    hb = CKPT_ROOT / "HYPERNET"
    print("\n--- Hypernetwork main ---")
    for v,fn in [("no_oracle","hypernet_best.pt"),("full_repos","hypernet_best.pt")]:
        up(api, hb/v/fn, rid, f"hypernet/{v}/{fn}", "model", a.dry_run)
        r = hb/v/"README.md"
        if r.exists(): up(api, r, rid, f"hypernet/{v}/README.md", "model", a.dry_run)
    print("\n--- Hypernetwork scaling ---")
    for d in sorted(d for d in hb.glob("scale_*repos") if d.is_dir() and "_results" not in d.name):
        up(api, d/"hypernet_best.pt", rid, f"hypernet/{d.name}/hypernet_best.pt", "model", a.dry_run)
    print("\n--- Training results ---")
    for d in sorted(hb.glob("*_results")):
        if d.is_dir(): up(api, d, rid, f"hypernet/{d.name}", "model", a.dry_run)
    print("\n--- Hypernetwork PAW ---")
    paw = CKPT_ROOT / "HYPERNET_PAW"
    up(api, paw/"no_oracle"/"lora_mapper_best.pt", rid,
       "hypernet_paw/no_oracle/lora_mapper_best.pt", "model", a.dry_run)
    if (paw/"no_oracle_results").exists():
        up(api, paw/"no_oracle_results", rid, "hypernet_paw/no_oracle_results", "model", a.dry_run)
    print("\n--- Single LoRA ---")
    up(api, CKPT_ROOT/"SINGLE_LORA"/"adapter", rid, "single_lora/adapter", "model", a.dry_run)
    print("\n--- Full Fine-Tuning ---")
    up(api, CKPT_ROOT/"FFT"/"final", rid, "fft/final", "model", a.dry_run)
    if a.include_per_repo_lora:
        print("\n--- Per-Repo LoRA (adapters only) ---")
        prl = CKPT_ROOT / "PER_REPO_LORA"
        for entry in sorted(prl.iterdir()):
            if entry.is_file():
                up(api, entry, rid, f"per_repo_lora/{entry.name}", "model", a.dry_run)
                continue
            for rd in sorted(entry.iterdir()):
                if rd.name.endswith("_results"):
                    up(api, rd, rid, f"per_repo_lora/{entry.name}/{rd.name}", "model", a.dry_run)
                elif rd.is_dir() and (rd/"adapter").exists():
                    up(api, rd/"adapter", rid,
                       f"per_repo_lora/{entry.name}/{rd.name}/adapter", "model", a.dry_run)
    up_readme(api, rid, "model", CK_README, a.dry_run)

def main():
    a = args()
    if not (a.all or a.datasets or a.checkpoints or a.results):
        print("Use --all, --datasets, --checkpoints, and/or --results"); sys.exit(1)
    if a.dry_run:
        print("[DRY RUN] Skipping authentication\n")
        api = None
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: export HF_TOKEN='hf_...'")
            print("Get one at: https://huggingface.co/settings/tokens")
            sys.exit(1)
        api = HfApi(token=token)
        try:
            info = api.whoami()
            username = info.get("name", "???")
            print(f"Authenticated as: {username}\n")
            if a.hf_user == DEFAULT_USER or a.hf_user != username:
                a.hf_user = username
                print(f"  Using HF namespace: {a.hf_user}\n")
        except Exception as e:
            print(f"Auth failed: {e}"); sys.exit(1)
    if a.all or a.datasets: push_ds(api, a)
    if a.all or a.checkpoints: push_ck(api, a)
    if a.results and not (a.all or a.datasets):
        rid = f"{a.hf_user}/{DS_REPO}"
        mk(api, rid, "dataset", a.private, a.dry_run)
        print("\n--- Evaluation results ---")
        for f in sorted(BASELINES_ROOT.glob("*.json")):
            up(api, f, rid, f"evaluation_results/{f.name}", "dataset", a.dry_run)
    print(f"\n{'='*60}\nDONE!")
    if not a.dry_run:
        print(f"  Dataset:     https://huggingface.co/datasets/{a.hf_user}/{DS_REPO}")
        print(f"  Checkpoints: https://huggingface.co/{a.hf_user}/{CK_REPO}")
    print("="*60)

if __name__ == "__main__":
    main()
