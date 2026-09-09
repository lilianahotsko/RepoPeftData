#!/usr/bin/env python3
"""Build the paper-exact dataset artifacts for the code2lora HF org.

Produces, under --out-dir (default $SCRATCH/REPO_DATASET/paper_exact_hf):

    static/qna/{train,cr_val,cr_test,ir_val,ir_test,ood_test}.parquet
        Static-track (V1) QnAs, one row per assertion pair -- exactly the
        62,294 in-distribution + 9,942 OOD QnAs of Table tab:dataset_actual.
    static/repos/{train,cr_val,cr_test}.parquet
        One row per repository with the frozen 2048-d repo embedding used
        by the static-track hypernetwork.
    qna8/{cr_val,cr_test,ir_val,ir_test,ood_test}.parquet
        Evolution-track eval QnAs after the paper's eval-time cap: the
        first <=8 rows per (repo_id, commit_sha) group in parquet row
        order (exactly what --qnas-per-commit-limit 8 scores).
    smartcap/train_paper.parquet
        The 215,129 smart-capped rows with in_repo_split == 'train' that
        provide the Code2LoRA-Evo training loss.
    PAPER_EXACT_README.json
        Row counts + sha256 provenance for everything above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

SCRATCH = os.environ.get("SCRATCH", "/scratch/lhotsko")
DS = Path(SCRATCH) / "REPO_DATASET"

STATIC_JSONS = {
    "train": DS / "train.json",
    "cr_val": DS / "cr_val.json",
    "cr_test": DS / "cr_test.json",
    "ir_val": DS / "ir_val.json",
    "ir_test": DS / "ir_test.json",
    "ood_test": DS / "ood_test.json",
}
# (repos, qnas) exactly as in Table tab:dataset_actual (static track).
STATIC_EXPECTED = {
    "train": (409, 39612),
    "cr_val": (51, 6213),
    "cr_test": (52, 6414),
    "ir_val": (409, 4833),
    "ir_test": (409, 5222),
    "ood_test": (92, 9942),
}

EVAL_QNA_SOURCES = {
    "cr_val": DS / "code2lora_snapshots_hf/qna/cr_val.parquet",
    "cr_test": DS / "code2lora_snapshots_hf/qna/cr_test.parquet",
    "ir_val": DS / "code2lora_snapshots_hf/qna/ir_val.parquet",
    "ir_test": DS / "code2lora_snapshots_hf/qna/ir_test.parquet",
    "ood_test": DS / "commit_parquet_hf_v2_ood/qna/ood_test.parquet",
}
# Capped counts exactly as in Table tab:dataset_actual (evolution track).
EVAL_EXPECTED = {
    "cr_val": 58944,
    "cr_test": 44732,
    "ir_val": 38783,
    "ir_test": 42061,
    "ood_test": 14813,
}

SMARTCAP_SRC = DS / "commit_parquet_hf_smartcap/qna/train.parquet"
SMARTCAP_EXPECTED = 215129


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_static(out_dir: Path, manifest: dict) -> None:
    qna_dir = out_dir / "static" / "qna"
    repos_dir = out_dir / "static" / "repos"
    qna_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    for split, src in STATIC_JSONS.items():
        print(f"[static:{split}] reading {src}", flush=True)
        repos = json.loads(src.read_text())["repositories"]

        rows = defaultdict(list)
        n_pairs = 0
        for repo_id, entry in repos.items():
            for p in entry["qna_pairs"]:
                meta = p.get("metadata", {}) or {}
                rows["repo_id"].append(repo_id)
                rows["prefix"].append(p["prefix"])
                rows["target"].append(p["target"])
                rows["assertion_type"].append(p.get("assertion_type"))
                rows["difficulty"].append(p.get("difficulty"))
                rows["test_file"].append(meta.get("file"))
                rows["test_function"].append(meta.get("test_function"))
                rows["test_class"].append(meta.get("test_class"))
                lineno = meta.get("lineno")
                rows["lineno"].append(int(lineno) if lineno is not None else None)
                ci = p.get("commit_idx", meta.get("commit_idx"))
                rows["commit_idx"].append(int(ci) if ci is not None else None)
                n_pairs += 1

        table = pa.table(
            {
                "repo_id": pa.array(rows["repo_id"], pa.string()),
                "prefix": pa.array(rows["prefix"], pa.large_string()),
                "target": pa.array(rows["target"], pa.large_string()),
                "assertion_type": pa.array(rows["assertion_type"], pa.string()),
                "difficulty": pa.array(rows["difficulty"], pa.string()),
                "test_file": pa.array(rows["test_file"], pa.string()),
                "test_function": pa.array(rows["test_function"], pa.string()),
                "test_class": pa.array(rows["test_class"], pa.string()),
                "lineno": pa.array(rows["lineno"], pa.int32()),
                "commit_idx": pa.array(rows["commit_idx"], pa.int32()),
            }
        )
        exp_repos, exp_qnas = STATIC_EXPECTED[split]
        n_repos = len(repos)
        assert (n_repos, n_pairs) == (exp_repos, exp_qnas), (
            f"static {split}: got ({n_repos}, {n_pairs}), "
            f"expected ({exp_repos}, {exp_qnas})"
        )
        dst = qna_dir / f"{split}.parquet"
        pq.write_table(table, dst, compression="zstd")
        manifest[f"static/qna/{split}"] = {
            "rows": n_pairs, "repos": n_repos, "sha256": _sha256(dst)}
        print(f"[static:{split}] {n_repos} repos, {n_pairs} qnas -> {dst}",
              flush=True)

        # Repo-level embeddings (train / cr_val / cr_test carry the frozen
        # 2048-d snapshot embedding; ir_* reuse the train repos, ood has none).
        if split in ("train", "cr_val", "cr_test"):
            emb_rows = {"repo_id": [], "embedding": []}
            for repo_id, entry in repos.items():
                emb = entry.get("embedding")
                if emb:
                    emb_rows["repo_id"].append(repo_id)
                    emb_rows["embedding"].append(emb)
            etable = pa.table(
                {
                    "repo_id": pa.array(emb_rows["repo_id"], pa.string()),
                    "embedding": pa.array(
                        emb_rows["embedding"], pa.list_(pa.float32(), 2048)),
                }
            )
            edst = repos_dir / f"{split}.parquet"
            pq.write_table(etable, edst, compression="zstd")
            manifest[f"static/repos/{split}"] = {
                "rows": etable.num_rows, "sha256": _sha256(edst)}
            print(f"[static:{split}] {etable.num_rows} repo embeddings -> {edst}",
                  flush=True)


def build_capped_eval(out_dir: Path, manifest: dict) -> None:
    qna8_dir = out_dir / "qna8"
    qna8_dir.mkdir(parents=True, exist_ok=True)
    for suite, src in EVAL_QNA_SOURCES.items():
        print(f"[qna8:{suite}] streaming {src}", flush=True)
        pf = pq.ParquetFile(str(src))
        dst = qna8_dir / f"{suite}.parquet"
        writer = None
        seen: dict = defaultdict(int)
        n_in = 0
        n_out = 0
        # iter_batches preserves file row order, which is what the
        # evaluator's first-<=8-per-commit rule keys on.
        for batch in pf.iter_batches(batch_size=8192):
            n_in += batch.num_rows
            repo_col = batch.column("repo_id").to_pylist()
            sha_col = batch.column("commit_sha").to_pylist()
            mask = []
            for r, s in zip(repo_col, sha_col):
                k = (r, s)
                seen[k] += 1
                mask.append(seen[k] <= 8)
            filtered = pa.Table.from_batches([batch]).filter(
                pa.array(mask, pa.bool_()))
            if writer is None:
                writer = pq.ParquetWriter(dst, filtered.schema,
                                          compression="zstd")
            if filtered.num_rows:
                writer.write_table(filtered)
                n_out += filtered.num_rows
        if writer is not None:
            writer.close()
        exp = EVAL_EXPECTED[suite]
        assert n_out == exp, f"qna8 {suite}: got {n_out}, expected {exp}"
        manifest[f"qna8/{suite}"] = {
            "rows": n_out,
            "commits": len(seen),
            "sha256": _sha256(dst),
        }
        print(f"[qna8:{suite}] {n_in} -> {n_out} rows "
              f"({len(seen)} commits) -> {dst}", flush=True)


def build_smartcap_train(out_dir: Path, manifest: dict) -> None:
    dst_dir = out_dir / "smartcap"
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smartcap] streaming {SMARTCAP_SRC}", flush=True)
    pf = pq.ParquetFile(str(SMARTCAP_SRC))
    dst = dst_dir / "train_paper.parquet"
    writer = None
    n_out = 0
    for batch in pf.iter_batches(batch_size=8192):
        table = pa.Table.from_batches([batch])
        filtered = table.filter(
            pc.equal(table.column("in_repo_split"), "train"))
        if writer is None:
            writer = pq.ParquetWriter(dst, filtered.schema,
                                      compression="zstd")
        if filtered.num_rows:
            writer.write_table(filtered)
            n_out += filtered.num_rows
    if writer is not None:
        writer.close()
    assert n_out == SMARTCAP_EXPECTED, (
        f"smartcap train: got {n_out}, expected {SMARTCAP_EXPECTED}")
    manifest["smartcap/train_paper"] = {"rows": n_out, "sha256": _sha256(dst)}
    print(f"[smartcap] {n_out} rows -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DS / "paper_exact_hf"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}

    build_static(out_dir, manifest)
    build_capped_eval(out_dir, manifest)
    build_smartcap_train(out_dir, manifest)

    readme = out_dir / "PAPER_EXACT_README.json"
    readme.write_text(json.dumps(
        {
            "description": "Paper-exact splits for RepoPeftBench / Code2LoRA "
                           "(Table tab:dataset_actual).",
            "static_sources": {k: str(v) for k, v in STATIC_JSONS.items()},
            "eval_sources": {k: str(v) for k, v in EVAL_QNA_SOURCES.items()},
            "smartcap_source": str(SMARTCAP_SRC),
            "cap_rule": "first <=8 QnAs per (repo_id, commit_sha) in parquet "
                        "row order (evaluator --qnas-per-commit-limit 8)",
            "files": manifest,
        }, indent=2))
    print(f"\nAll checks passed. Manifest -> {readme}", flush=True)


if __name__ == "__main__":
    main()
