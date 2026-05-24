#!/usr/bin/env python3
"""Find QnAs where only Code2LoRA-direct and Code2LoRA-GRU are exact-match correct.

Uses v2 GRU eval shard JSONs (raw_samples) aligned by (repo, commit, test location).
Writes a markdown report with contexts; optionally re-runs inference for predictions.

Usage:
  python scripts/collect_c2l_exclusive_examples.py --suite cr_test --limit 5
  python scripts/collect_c2l_exclusive_examples.py --suite cr_test --run-predictions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HYP = _ROOT / "hypernetwork"
if str(_HYP) not in sys.path:
    sys.path.insert(0, str(_HYP))

from code2lora_core import (  # noqa: E402
    load_commit_rows_for_gru,
    load_qna_rows,
    load_snapshot_rows,
)

QnaKey = Tuple[str, str, str, int, int]  # repo, sha, test_file, lineno, col_offset


def _qna_key(repo: str, sha: str, test_file: str, lineno: int, col_offset: int) -> QnaKey:
    return (repo, sha, test_file, int(lineno), int(col_offset))


def _iter_gru_keys(
    commits_parquet: Path,
    qna_parquet: Path,
    *,
    shard_i: int,
    num_shards: int,
    qnas_per_commit_limit: int = 8,
    in_repo_splits: Optional[List[str]] = None,
) -> Iterator[Tuple[QnaKey, int]]:
    """Same visit order as run_code2lora_gru_v2_eval.py."""
    rows_by_repo = load_commit_rows_for_gru(commits_parquet)
    all_repos = sorted(rows_by_repo.keys())
    if num_shards > 1:
        kept = {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
    else:
        kept = set(all_repos)

    qna_rows = load_qna_rows(qna_parquet)
    qnas_by_key: Dict[Tuple[str, str], List[QnaKey]] = defaultdict(list)
    for qr in qna_rows:
        if in_repo_splits is not None and qr.in_repo_split not in in_repo_splits:
            continue
        qnas_by_key[(qr.repo_id, qr.commit_sha)].append(
            _qna_key(qr.repo_id, qr.commit_sha, qr.test_file,
                     qr.lineno, qr.col_offset)
        )

    for repo_id in sorted(rows_by_repo.keys()):
        if repo_id not in kept:
            continue
        for row in rows_by_repo[repo_id]:
            if in_repo_splits is not None and row.in_repo_split not in in_repo_splits:
                continue
            keys = qnas_by_key.get((row.repo_id, row.commit_sha))
            if not keys:
                continue
            if qnas_per_commit_limit:
                keys = keys[:qnas_per_commit_limit]
            for k in keys:
                yield k, int(row.commit_index)


def _iter_static_keys(
    snapshots_parquet: Path,
    qna_parquet: Path,
    *,
    shard_i: int,
    num_shards: int,
    qnas_per_commit_limit: int = 8,
    in_repo_splits: Optional[List[str]] = None,
) -> Iterator[QnaKey]:
    """Same visit order as run_code2lora_static_v2_eval.py."""
    snap_rows = load_snapshot_rows(
        snapshots_parquet, in_repo_splits=in_repo_splits,
    )
    rows_by_repo: Dict[str, List[Any]] = defaultdict(list)
    for sr in snap_rows:
        rows_by_repo[sr.repo_id].append(sr)
    for repo, rows in rows_by_repo.items():
        rows.sort(key=lambda r: (int(r.commit_index), r.commit_sha))
    all_repos = sorted(rows_by_repo.keys())
    if num_shards > 1:
        kept = {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
    else:
        kept = set(all_repos)

    qna_rows = load_qna_rows(qna_parquet, in_repo_splits=in_repo_splits)
    qnas_by_key: Dict[Tuple[str, str], List[QnaKey]] = defaultdict(list)
    for qr in qna_rows:
        qnas_by_key[(qr.repo_id, qr.commit_sha)].append(
            _qna_key(qr.repo_id, qr.commit_sha, qr.test_file,
                     qr.lineno, qr.col_offset)
        )

    for repo_id in sorted(rows_by_repo.keys()):
        if repo_id not in kept:
            continue
        for row in rows_by_repo[repo_id]:
            keys = qnas_by_key.get((row.repo_id, row.commit_sha))
            if not keys:
                continue
            if qnas_per_commit_limit:
                keys = keys[:qnas_per_commit_limit]
            for k in keys:
                yield k


def _iter_baseline_keys(
    qna_parquet: Path,
    *,
    shard_i: int,
    num_shards: int,
    qnas_per_commit_limit: int = 8,
    in_repo_splits: Optional[List[str]] = None,
) -> Iterator[QnaKey]:
    """Same visit order as run_baselines_v2.py score_suite."""
    rows = load_qna_rows(qna_parquet)
    groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"commit_index": -1, "keys": []}
    )
    for qr in rows:
        if in_repo_splits is not None and qr.in_repo_split not in in_repo_splits:
            continue
        key = (qr.repo_id, qr.commit_sha)
        g = groups[key]
        g["commit_index"] = int(qr.commit_index)
        g["keys"].append(
            _qna_key(qr.repo_id, qr.commit_sha, qr.test_file,
                     qr.lineno, qr.col_offset)
        )
    all_repos = sorted({r for (r, _) in groups.keys()})
    if num_shards > 1:
        kept = {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
    else:
        kept = set(all_repos)
    group_keys = sorted(
        groups.keys(),
        key=lambda k: (k[0], groups[k]["commit_index"]),
    )
    for repo_id, sha in group_keys:
        if repo_id not in kept:
            continue
        keys = groups[(repo_id, sha)]["keys"]
        if qnas_per_commit_limit:
            keys = keys[:qnas_per_commit_limit]
        for k in keys:
            yield k


def _load_shard_em(path: Path) -> List[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(x) for x in data.get("raw_samples", {}).get("exact_match", [])]


def _em_by_key_from_shards(
    shard_paths: List[Path],
    iter_keys_fn,
) -> Dict[QnaKey, int]:
    out: Dict[QnaKey, int] = {}
    for sp in sorted(shard_paths):
        data = json.loads(sp.read_text(encoding="utf-8"))
        shard_i = int(data.get("shard_i", 0))
        num_shards = int(data.get("num_shards", 1))
        ems = _load_shard_em(sp)
        keys = list(iter_keys_fn(shard_i=shard_i, num_shards=num_shards))
        if len(keys) != len(ems):
            raise RuntimeError(
                f"key/em length mismatch in {sp}: {len(keys)} keys vs {len(ems)} ems"
            )
        for k, em in zip(keys, ems):
            out[k] = em
    return out


def _glob_shards(run_dir: Path, suite: str, prefix: str) -> List[Path]:
    return sorted(run_dir.glob(f"{prefix}_{suite}_shard*of*.json"))


def _repo_commit_stats(
    repo_id: str,
    sha: str,
    repos_root: Path,
    commits_parquet: Path,
) -> Dict[str, Any]:
    """Files at commit + max commit index + rough repo token count."""
    author, name = repo_id.split("/", 1)
    repo_dir = repos_root / author / name
    n_files = 0
    total_chars = 0
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", sha],
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        paths = out.decode("utf-8", errors="ignore").splitlines()
        py_paths = [p for p in paths if p.endswith(".py")]
        n_files = len(py_paths)
        for p in py_paths[:200]:  # cap for speed
            try:
                blob = subprocess.check_output(
                    ["git", "-C", str(repo_dir), "show", f"{sha}:{p}"],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                total_chars += len(blob)
            except Exception:
                pass
        if len(py_paths) > 200:
            total_chars = int(total_chars * len(py_paths) / 200)
    except Exception:
        pass

    max_idx = 0
    commit_index = 0
    rows_by_repo = load_commit_rows_for_gru(commits_parquet)
    for row in rows_by_repo.get(repo_id, []):
        max_idx = max(max_idx, int(row.commit_index))
        if row.commit_sha == sha:
            commit_index = int(row.commit_index)
    pct = (100.0 * commit_index / max_idx) if max_idx > 0 else 0.0
    return {
        "n_py_files": n_files,
        "approx_repo_chars": total_chars,
        "commit_index": commit_index,
        "max_commit_index": max_idx,
        "commit_pct": pct,
    }


def _load_drc(repo_id: str, sha: str, qna: Dict[str, Any], drc_dir: Path) -> str:
    from evaluation.run_baselines_v2 import (
        _drc_key_for_qna,
        _load_drc_commit_contexts,
    )
    from evaluation.compress_context import compress_oracle_context
    from transformers import AutoTokenizer

    ctxs = _load_drc_commit_contexts(drc_dir, repo_id, sha)
    raw = ""
    if ctxs:
        entry = ctxs.get(_drc_key_for_qna(qna), {})
        raw = entry.get("extracted_code", "") if isinstance(entry, dict) else ""
    if not raw:
        return "(no DRC context for this QnA)"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    return compress_oracle_context(raw, qna["prefix"], tok, max_tokens=4096)


def _load_rag_context(
    repo_id: str,
    sha: str,
    prefix: str,
    rag_dir: Path,
    *,
    top_k: int = 3,
) -> str:
    from evaluation.run_baselines_v2 import (
        _embed_rag_queries,
        _load_rag_commit_index,
        _rag_retrieve_and_compress,
    )
    from transformers import AutoModel, AutoTokenizer
    import torch

    index = _load_rag_commit_index(rag_dir, repo_id, sha)
    if not index.get("chunks"):
        return "(no RAG index for this commit)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    etok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    emodel = AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    qe = _embed_rag_queries(
        [prefix[-2000:]], emodel, etok, torch.device(device),
    )[0]
    prompt = _rag_retrieve_and_compress(
        prefix, qe.unsqueeze(0), index, top_k=top_k, tokenizer=tok,
        rag_max_context_tokens=1536, rag_hybrid=True,
    )
    # Return only the retrieved prefix (before original prefix duplicated)
    if prompt == prefix:
        return "(RAG: no chunks retrieved)"
    if prompt.endswith(prefix):
        return prompt[: -len(prefix)].rstrip()
    return prompt


@dataclass
class Example:
    key: QnaKey
    commit_index: int
    em: Dict[str, int] = field(default_factory=dict)
    qna: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    drc_context: str = ""
    rag_context: str = ""
    predictions: Dict[str, str] = field(default_factory=dict)


class _EvalContext:
    """Cached parquet state so shard alignment does not reload commits each time."""

    def __init__(
        self,
        suite: str,
        scratch: Path,
        qna_parquet: Path,
        commits_parquet: Path,
        snapshots_parquet: Path,
    ):
        self.qna_parquet = qna_parquet
        self.rows_by_repo = load_commit_rows_for_gru(commits_parquet)
        self._qna_keys_by_commit: Dict[Tuple[str, str], List[QnaKey]] = defaultdict(list)
        self.qna_by_key: Dict[QnaKey, Dict[str, Any]] = {}
        self._commit_index: Dict[Tuple[str, str], int] = {}
        import pyarrow.parquet as pq
        tbl = pq.read_table(
            str(qna_parquet),
            columns=[
                "repo_id", "commit_sha", "test_file", "lineno", "col_offset",
            ],
        )
        for row in tbl.to_pylist():
            k = _qna_key(
                row["repo_id"], row["commit_sha"], row["test_file"],
                row["lineno"], row["col_offset"],
            )
            self._qna_keys_by_commit[(row["repo_id"], row["commit_sha"])].append(k)
        for repo_id, rows in self.rows_by_repo.items():
            for row in rows:
                self._commit_index[(repo_id, row.commit_sha)] = int(row.commit_index)
        self._snap_rows_by_repo: Optional[Dict[str, List[Any]]] = None
        self.snapshots_parquet = snapshots_parquet

    def gru_keys(self, shard_i: int, num_shards: int, limit: int = 8) -> List[QnaKey]:
        all_repos = sorted(self.rows_by_repo.keys())
        kept = (
            {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
            if num_shards > 1 else set(all_repos)
        )
        out: List[QnaKey] = []
        for repo_id in sorted(self.rows_by_repo.keys()):
            if repo_id not in kept:
                continue
            for row in self.rows_by_repo[repo_id]:
                keys = self._qna_keys_by_commit.get((repo_id, row.commit_sha))
                if not keys:
                    continue
                if limit:
                    keys = keys[:limit]
                out.extend(keys)
        return out

    def baseline_keys(self, shard_i: int, num_shards: int, limit: int = 8) -> List[QnaKey]:
        groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {"commit_index": -1, "keys": []}
        )
        for (repo, sha), keys in self._qna_keys_by_commit.items():
            groups[(repo, sha)]["keys"] = keys[:limit] if limit else keys
            groups[(repo, sha)]["commit_index"] = self._commit_index.get(
                (repo, sha), -1
            )
        all_repos = sorted({r for (r, _) in groups.keys()})
        kept = (
            {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
            if num_shards > 1 else set(all_repos)
        )
        out: List[QnaKey] = []
        for (repo, sha), g in sorted(
            groups.items(), key=lambda x: (x[0][0], x[1]["commit_index"])
        ):
            if repo not in kept:
                continue
            out.extend(g["keys"])
        return out

    def static_keys(self, shard_i: int, num_shards: int, limit: int = 8) -> List[QnaKey]:
        if self._snap_rows_by_repo is None:
            snap_rows = load_snapshot_rows(self.snapshots_parquet, in_repo_splits=None)
            self._snap_rows_by_repo = defaultdict(list)
            for sr in snap_rows:
                self._snap_rows_by_repo[sr.repo_id].append(sr)
            for repo, rows in self._snap_rows_by_repo.items():
                rows.sort(key=lambda r: (int(r.commit_index), r.commit_sha))
        all_repos = sorted(self._snap_rows_by_repo.keys())
        kept = (
            {r for i, r in enumerate(all_repos) if i % num_shards == shard_i}
            if num_shards > 1 else set(all_repos)
        )
        out: List[QnaKey] = []
        for repo_id in sorted(self._snap_rows_by_repo.keys()):
            if repo_id not in kept:
                continue
            for row in self._snap_rows_by_repo[repo_id]:
                keys = self._qna_keys_by_commit.get((row.repo_id, row.commit_sha))
                if not keys:
                    continue
                if limit:
                    keys = keys[:limit]
                out.extend(keys)
        return out


def _merge_shard_ems(
    shard_paths: List[Path],
    keys_fn,
) -> Dict[QnaKey, int]:
    out: Dict[QnaKey, int] = {}
    for sp in sorted(shard_paths):
        data = json.loads(sp.read_text(encoding="utf-8"))
        shard_i = int(data["shard_i"])
        num_shards = int(data["num_shards"])
        keys = keys_fn(shard_i, num_shards)
        ems = _load_shard_em(sp)
        if len(keys) != len(ems):
            raise RuntimeError(f"{sp}: {len(keys)} keys vs {len(ems)} ems")
        for k, em in zip(keys, ems):
            out[k] = em
    return out


def find_exclusive(suite: str, scratch: Path) -> List[Example]:
    ckpt = scratch / "TRAINING_CHECKPOINTS"
    qna_parquet = scratch / "REPO_DATASET/code2lora_snapshots_hf/qna" / f"{suite}.parquet"
    commits_parquet = (
        scratch / "REPO_DATASET/commit_parquet_hf_v2/commits" / f"{suite}.parquet"
    )
    snapshots_parquet = (
        scratch / "REPO_DATASET/code2lora_snapshots_hf/commits" / f"{suite}.parquet"
    )
    print("Loading QnA + commit parquets (once)...", flush=True)
    ctx = _EvalContext(suite, scratch, qna_parquet, commits_parquet, snapshots_parquet)

    print("Aligning GRU shard EM...", flush=True)
    em_gru = _merge_shard_ems(
        _glob_shards(ckpt / "CODE2LORA_GRU_EVAL_V2/h100_v2_gru_3ep_best_sharded",
                     suite, "gru_v2"),
        ctx.gru_keys,
    )
    print("Aligning static shard EM...", flush=True)
    em_static = _merge_shard_ems(
        _glob_shards(ckpt / "CODE2LORA_STATIC_EVAL_V2/h100_v2_static_3ep_run5_ep2",
                     suite, "static_v2"),
        ctx.static_keys,
    )
    print("Aligning baseline shard EMs...", flush=True)
    em_drc = _merge_shard_ems(
        _glob_shards(ckpt / "BASELINES_V2/drc_h100_v2_sharded", suite, "baseline_drc"),
        ctx.baseline_keys,
    )
    em_rag = _merge_shard_ems(
        _glob_shards(ckpt / "BASELINES_V2/rag_h100_v2_sharded", suite, "baseline_rag"),
        ctx.baseline_keys,
    )
    em_slora = _merge_shard_ems(
        _glob_shards(ckpt / "BASELINES_V2/slora_h100_v2_a24000_sharded",
                     suite, "baseline_slora"),
        ctx.baseline_keys,
    )
    em_pre = _merge_shard_ems(
        _glob_shards(ckpt / "BASELINES_V2/pretrained_h100_v2_prefix256_sharded",
                     suite, "baseline_pretrained"),
        ctx.baseline_keys,
    )

    candidates: List[Example] = []
    for k, em_g in em_gru.items():
        if em_g != 1 or em_static.get(k, 0) != 1:
            continue
        others = {
            "pretrained": em_pre.get(k, -1),
            "rag": em_rag.get(k, -1),
            "drc": em_drc.get(k, -1),
            "slora": em_slora.get(k, -1),
        }
        if any(v < 0 for v in others.values()):
            continue
        if any(v != 0 for v in others.values()):
            continue
        repo, sha = k[0], k[1]
        candidates.append(Example(
            key=k,
            commit_index=ctx._commit_index.get((repo, sha), 0),
            em={"code2lora_gru": 1, "code2lora_static": 1, **{m: 0 for m in others}},
            qna=ctx.qna_by_key.get(k, {}),
        ))
    return candidates


def _run_predictions(ex: Example, scratch: Path) -> Dict[str, str]:
    """Run inference for one QnA across methods (GPU)."""
    import torch
    from evaluation.run_baselines_v2 import (
        _prepare_prefix_ids,
        _generate_batch,
        load_model_for_method,
    )
    from evaluation.metrics import compute_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds: Dict[str, str] = {}
    prefix = ex.qna["prefix"]
    target = ex.qna["target"]

    # Pretrained
    tok, model = load_model_for_method("pretrained", "Qwen/Qwen2.5-Coder-1.5B",
                                       None, device=device)
    bos = tok.bos_token_id or tok.eos_token_id
    inp = [_prepare_prefix_ids(tok, prefix, 4096, bos)]
    preds["pretrained"] = _generate_batch(model, tok, device, inp, max_new_tokens=64)[0]

    # RAG / DRC via baseline loader
    rag_dir = scratch / "RAG_CHUNK_CACHE_COMMITS"
    drc_dir = scratch / "ORACLE_CONTEXT_CACHE_COMMITS"
    rag_prompt = _load_rag_context(ex.key[0], ex.key[1], prefix, rag_dir)
    drc_raw = _load_drc(ex.key[0], ex.key[1], ex.qna, drc_dir)
    for method, prompt in [("rag", rag_prompt + "\n\n\n" + prefix if rag_prompt.startswith("#") else prefix),
                           ("drc", drc_raw + "\n\n\n" + prefix if drc_raw and not drc_raw.startswith("(") else prefix)]:
        tok, model = load_model_for_method("pretrained", "Qwen/Qwen2.5-Coder-1.5B",
                                           None, device=device)
        bos = tok.bos_token_id or tok.eos_token_id
        inp = [_prepare_prefix_ids(tok, prompt, 4096, bos)]
        preds[method] = _generate_batch(model, tok, device, inp, max_new_tokens=64)[0]

    slora_ckpt = scratch / "BASELINES/slora_r16_v2/adapter"
    tok, model = load_model_for_method("slora", "Qwen/Qwen2.5-Coder-1.5B",
                                       slora_ckpt if slora_ckpt.exists() else None,
                                       device=device)
    bos = tok.bos_token_id or tok.eos_token_id
    inp = [_prepare_prefix_ids(tok, prefix, 4096, bos)]
    preds["slora"] = _generate_batch(model, tok, device, inp, max_new_tokens=64)[0]

  # Code2LoRA static + GRU need dedicated loaders — skip full re-run; use EM labels
    preds["gold_target"] = target
    return preds


def write_markdown(
    examples: List[Example],
    out_path: Path,
    suite: str,
) -> None:
    lines = [
        "# Code2LoRA-exclusive completion examples",
        "",
        f"Suite: **{suite}** (commit-derived prefixes, v2 GRU eval protocol).",
        "",
        "Each row is a QnA where **only** Code2LoRA-direct (static projection) and ",
        "Code2LoRA-GRU achieved exact match; all other listed baselines failed.",
        "",
        "Pretrained/RAG/DRC/sLoRA correctness comes from saved eval shards ",
        "(pretrained from the prefix-256 matched-budget ablation shards).",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        repo, sha, tf, ln, col = ex.key
        st = ex.stats
        lines += [
            f"## Example {i}: `{repo}`",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Repository** | `{repo}` |",
            f"| **Commit SHA** | `{sha[:12]}…` |",
            f"| **Commit position** | {st.get('commit_pct', 0):.1f}% of repo timeline "
            f"(index {st.get('commit_index')}/{st.get('max_commit_index')}) |",
            f"| **Python files at commit** | {st.get('n_py_files', '?')} |",
            f"| **Approx. repo source size** | {st.get('approx_repo_chars', 0):,} chars "
            f"(.py blobs, estimated) |",
            f"| **Test location** | `{tf}:{ln}:{col}` |",
            "",
            "### Exact match (1=correct)",
            "",
        ]
        for m, v in sorted(ex.em.items()):
            lines.append(f"- **{m}**: {v}")
        lines += [
            "",
            "### Gold target",
            "",
            "```python",
            ex.qna.get("target", ""),
            "```",
            "",
            "### QnA prefix (model input)",
            "",
            "```python",
            ex.qna.get("prefix", "")[:12000],
            "```",
            "",
            "### DRC context (compressed)",
            "",
            "```python",
            ex.drc_context[:8000],
            "```",
            "",
            "### RAG context (retrieved + compressed, k=3)",
            "",
            "```python",
            ex.rag_context[:8000],
            "```",
            "",
            "### Predictions (exact-match outcome)",
            "",
            "Eval shards store per-QnA EM bits, not decoded strings. "
            "✓ = model completion matched the gold target after normalization.",
            "",
            "| Method | Exact match |",
            "|--------|-------------|",
        ]
        order = [
            "pretrained", "rag", "drc", "slora",
            "code2lora_static", "code2lora_gru",
        ]
        for m in order:
            v = ex.em.get(m, -1)
            mark = "✓" if v == 1 else ("✗" if v == 0 else "?")
            lines += [f"| {m} | {mark} |"]
        lines += ["", "**Gold target** (above) is the assertion literal to complete.", ""]
        if ex.predictions:
            lines += ["", "#### Re-generated completions (`--run-predictions`)", ""]
            for m, pred in ex.predictions.items():
                lines += [f"**{m}**", "", "```python", pred, "```", ""]
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cr_test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=_ROOT / "RepoPeft_Paper/qualitative_c2l_exclusive_examples.md")
    ap.add_argument("--run-predictions", action="store_true")
    args = ap.parse_args()

    scratch = Path(os.environ.get("SCRATCH", "/scratch/lhotsko"))
    repos_root = scratch / "REPO_DATASET/repositories"
    commits_parquet = (
        scratch / "REPO_DATASET/commit_parquet_hf_v2/commits" / f"{args.suite}.parquet"
    )
    drc_dir = scratch / "ORACLE_CONTEXT_CACHE_COMMITS"
    rag_dir = scratch / "RAG_CHUNK_CACHE_COMMITS"

    print("Scanning for Code2LoRA-exclusive QnAs...", flush=True)
    candidates = find_exclusive(args.suite, scratch)
    print(f"Found {len(candidates)} candidates", flush=True)

    # One example per repo
    by_repo: Dict[str, Example] = {}
    for ex in candidates:
        repo = ex.key[0]
        if repo not in by_repo:
            by_repo[repo] = ex
    selected = list(by_repo.values())[: args.limit]
    print(f"Selected {len(selected)} repos", flush=True)

    # Load prefix/target only for selected QnAs (row-group filters, low memory).
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    qna_parquet = scratch / "REPO_DATASET/code2lora_snapshots_hf/qna" / f"{args.suite}.parquet"
    cols = [
        "repo_id", "commit_sha", "test_file", "lineno", "col_offset",
        "prefix", "target", "assertion_event_id",
    ]
    for ex in selected:
        repo, sha, tf, ln, col = ex.key
        filt = (
            (pc.field("repo_id") == repo)
            & (pc.field("commit_sha") == sha)
            & (pc.field("test_file") == tf)
            & (pc.field("lineno") == ln)
            & (pc.field("col_offset") == col)
        )
        tbl = pq.read_table(str(qna_parquet), columns=cols, filters=filt)
        if tbl.num_rows == 0:
            continue
        ex.qna = tbl.to_pylist()[0]

    for ex in selected:
        ex.stats = _repo_commit_stats(
            ex.key[0], ex.key[1], repos_root, commits_parquet,
        )
        ex.stats["commit_index"] = ex.commit_index
        if not ex.qna:
            continue
        print(f"  contexts: {ex.key[0]} @ {ex.key[1][:8]}", flush=True)
        ex.drc_context = _load_drc(ex.key[0], ex.key[1], ex.qna, drc_dir)
        ex.rag_context = _load_rag_context(ex.key[0], ex.key[1], ex.qna["prefix"], rag_dir)
        if args.run_predictions:
            ex.predictions = _run_predictions(ex, scratch)

    write_markdown(selected, args.output, args.suite)


if __name__ == "__main__":
    main()
