#!/usr/bin/env python3
"""Shared building blocks for the v2 Code2LoRA family of trainers.

This is the **single source of truth** for the LoRA-generation head, the LoRA
module that wraps a target `nn.Linear`, and the parquet loaders that read
precomputed embeddings from the v2 datasets:

* `repopeft-gru-commits-v2`           -- per-commit `diff_embedding` (2048-d)
                                         + `repo_state_embedding` (2048-d).
* `repopeft-code2lora-snapshots`       -- per-snapshot `repo_state_embedding`
                                         (2048-d) + QnAs.

Both ``Code2LoRA-direct`` (static, see ``train_code2lora_static_v2.py``) and
``Code2LoRA-GRU<sub>commit</sub>`` (sequential, see
``train_code2lora_gru_v2.py``) consume the **same** ``Code2LoRAHead``:

    head(ctx) -> {"A": {module_type: [B, rank, in_f]},
                  "B": {module_type: [B, out_f, rank]}}

The head outputs ONE (A, B) pair per LoRA module *type* (e.g. q_proj, v_proj,
gate_proj...) -- shared across all transformer layers. This matches the
original "regular Code2LoRA" architecture (`hypernetwork_sampled.py:Hypernetwork`).

The two trainers differ ONLY in how ``ctx`` is computed:

* Static: ``ctx = repo_state_embedding``                                (2048-d)
* GRU   : ``ctx = h_T = GRU(diff_embedding_1, ..., diff_embedding_T,
                            h_0 = init(repo_state_embedding @ commit_0))``
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA module + injection (port of hypernetwork_sampled.py with light cleanup)
# ---------------------------------------------------------------------------

class LoRA(nn.Module):
    """Wraps an ``nn.Linear`` with an additive low-rank update.

    Forward: ``y = base(x) + scaling * (x @ A^T) @ B^T``, where the per-batch
    A: ``[rank, in_features]`` and B: ``[out_features, rank]`` come from an
    external hypernet via :meth:`set_lora_weights`.

    IMPORTANT autograd contract (matches ``hypernetwork_sampled.py:LoRA``):
    A and B are kept as **plain attributes**, not buffers, and are stored
    **without detaching**, so the LM loss's backward graph flows through
    them straight into the hypernet parameters that produced them. The
    base ``nn.Linear`` is frozen and its forward sees a detached copy of
    the input to avoid building an autograd graph through the LLM weights
    (saves a lot of memory).
    """

    def __init__(self, base: nn.Linear, in_features: int, out_features: int,
                 rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(alpha) / float(max(1, rank))
        # Plain attributes -- NOT registered buffers -- so assignment preserves
        # grad_fn for tensors coming from the hypernet.
        self.A: Optional[torch.Tensor] = None  # [rank, in_features]
        self.B: Optional[torch.Tensor] = None  # [out_features, rank]

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor) -> None:
        # No .detach() -- we WANT gradients to flow back to the hypernet.
        # No device cast either (head outputs land on the same device);
        # leave dtype as-is and let the forward path upcast to fp32 if
        # the base is bf16.
        self.A = A
        self.B = B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is None or self.B is None:
            return y
        # Compute the LoRA delta in fp32 for numerical headroom; detach x
        # so we don't build an autograd graph through the (frozen) base.
        x_f32 = x.detach().to(torch.float32)
        A = self.A.to(torch.float32)
        B = self.B.to(torch.float32)
        delta = F.linear(F.linear(x_f32, A), B) * self.scaling
        return y + delta.to(dtype=y.dtype)


@dataclass
class ModuleSpec:
    full_name: str   # e.g. 'model.layers.5.self_attn.q_proj'
    layer_idx: int
    type: str        # e.g. 'q_proj'
    in_features: int
    out_features: int


def get_module_specs(model: nn.Module, target_module_types: List[str]
                     ) -> List[ModuleSpec]:
    """Discover every nn.Linear in ``model`` whose name contains one of
    ``target_module_types`` and return one :class:`ModuleSpec` per match,
    sorted by (layer_idx, full_name).
    """
    specs: List[ModuleSpec] = []
    pat = re.compile(r"\bmodel\.layers\.(\d+)\.")
    for name, m in model.named_modules():
        match_type = next(
            (t for t in target_module_types if t in name), None
        )
        if match_type is None:
            continue
        if not isinstance(m, nn.Linear):
            continue
        m_layer = pat.search(name)
        layer_idx = int(m_layer.group(1)) if m_layer else -1
        specs.append(ModuleSpec(
            full_name=name,
            layer_idx=layer_idx,
            type=match_type,
            in_features=int(m.in_features),
            out_features=int(m.out_features),
        ))
    specs.sort(key=lambda s: (s.layer_idx, s.full_name))
    return specs


def replace_with_lora(model: nn.Module, specs: List[ModuleSpec],
                      rank: int, alpha: float) -> None:
    """Replace each target ``nn.Linear`` in ``model`` with a :class:`LoRA`
    wrapper. Idempotent."""
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for sp in specs:
        parent_name, attr = sp.full_name.rsplit(".", 1)
        orig = getattr(named[parent_name], attr)
        if isinstance(orig, LoRA):
            continue
        assert isinstance(orig, nn.Linear), \
            f"{sp.full_name} is not nn.Linear (got {type(orig)})"
        wrapped = LoRA(orig, sp.in_features, sp.out_features,
                       rank, alpha).to(device=device, dtype=dtype)
        setattr(named[parent_name], attr, wrapped)


def inject_lora_weights(model: nn.Module, specs: List[ModuleSpec],
                        head_out: Dict[str, Dict[str, torch.Tensor]],
                        batch_index: int = 0) -> None:
    """Push ``head_out["A"][type]`` and ``head_out["B"][type]`` into the
    wrapper :class:`LoRA` modules for every spec sharing that type."""
    A_by_type = head_out["A"]
    B_by_type = head_out["B"]
    named = dict(model.named_modules())
    for sp in specs:
        named[sp.full_name].set_lora_weights(
            A_by_type[sp.type][batch_index],
            B_by_type[sp.type][batch_index],
        )


def discover_module_types_and_dims(specs: List[ModuleSpec]
                                   ) -> Dict[str, Tuple[int, int]]:
    """Return {type_name: (in_features, out_features)} -- one entry per
    target module type. Assumes all instances of the same type share dims."""
    type_dims: Dict[str, Tuple[int, int]] = {}
    for sp in specs:
        if sp.type in type_dims:
            assert type_dims[sp.type] == (sp.in_features, sp.out_features), \
                f"type {sp.type} appears with inconsistent dims"
            continue
        type_dims[sp.type] = (sp.in_features, sp.out_features)
    return type_dims


# ---------------------------------------------------------------------------
# Shared LoRA generation head (= regular Code2LoRA's Hypernetwork)
# ---------------------------------------------------------------------------

class Code2LoRAHead(nn.Module):
    """The shared LoRA-generation head used by both Code2LoRA-direct
    (static) and Code2LoRA-GRU<sub>commit</sub>.

    Input  : ctx ``[B, input_dim]`` -- a single repo / repo-state context vector.
    Output : ``{"A": {type: [B, rank, in_f]}, "B": {type: [B, out_f, rank]}}``,
             one (A, B) pair per LoRA module *type*, shared across all
             transformer layers (this is the "regular Code2LoRA" head;
             not the PAW-style per-(layer, module) head).

    Args:
        input_dim   : Context-vector dim (typically 2048).
        type_dims   : ``{type: (in_features, out_features)}`` for each LoRA
                      module type (q_proj, v_proj, gate_proj, ...).
        hidden_dim  : Trunk hidden dimension.
        rank        : LoRA rank ``r``.
        init_log_scale : Initial log-scale for tanh squashing. -3.5 gives
                         output magnitudes ~0.03 at init -> tiny LoRA delta.
    """

    def __init__(
        self,
        input_dim: int,
        type_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int = 1024,
        rank: int = 16,
        init_log_scale: float = -3.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.type_dims = dict(type_dims)
        self.types = sorted(type_dims.keys())

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.heads_A = nn.ModuleDict({
            t: nn.Linear(hidden_dim, rank * type_dims[t][0])
            for t in self.types
        })
        self.heads_B = nn.ModuleDict({
            t: nn.Linear(hidden_dim, type_dims[t][1] * rank)
            for t in self.types
        })
        self.log_scale_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })
        self.log_scale_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })

    def forward(self, ctx: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        # ctx: [B, input_dim] or [B, K, input_dim] (the K case max-pools).
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values
        h = self.trunk(ctx.float())
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)

        A_out: Dict[str, torch.Tensor] = {}
        B_out: Dict[str, torch.Tensor] = {}
        for t in self.types:
            in_f, out_f = self.type_dims[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)
            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)
            A_out[t] = torch.tanh(A_raw) * scale_A
            B_out[t] = torch.tanh(B_raw) * scale_B
        return {"A": A_out, "B": B_out}

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "types": self.types,
            "type_dims": {t: list(v) for t, v in self.type_dims.items()},
        }


# ---------------------------------------------------------------------------
# Optional: tiny GRU that sits on top of Code2LoRAHead for the GRU variant
# ---------------------------------------------------------------------------

class CommitGRU(nn.Module):
    """One-layer GRU that ingests per-commit diff embeddings.

    h_0 comes from a learnable projection of the initial repo-state embedding
    (``repo_state_embedding @ commit_0``); h_T is fed straight into
    :class:`Code2LoRAHead` -- so the GRU variant *is* the static variant with
    an extra recurrence stage on top of the same head.
    """

    def __init__(
        self,
        diff_input_dim: int = 2048,
        repo_state_dim: int = 2048,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.diff_input_dim = diff_input_dim
        self.repo_state_dim = repo_state_dim
        self.hidden_dim = hidden_dim

        self.diff_proj = nn.Sequential(
            nn.Linear(diff_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.repo_init_proj = nn.Sequential(
            nn.Linear(repo_state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def init_hidden(self, repo_state_emb_0: torch.Tensor) -> torch.Tensor:
        # repo_state_emb_0: [B, repo_state_dim] -> h_0: [1, B, hidden_dim]
        h0 = self.repo_init_proj(repo_state_emb_0.float())
        return h0.unsqueeze(0)

    def step(self, diff_emb: torch.Tensor,
             h_prev: torch.Tensor) -> torch.Tensor:
        # diff_emb: [B, diff_input_dim] ; h_prev: [1, B, hidden_dim]
        x = self.diff_proj(diff_emb.float()).unsqueeze(1)  # [B, 1, H]
        _, h_new = self.gru(x, h_prev)
        return h_new

    def forward(self, diff_embs: torch.Tensor,
                repo_state_emb_0: torch.Tensor,
                lengths: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Sequential rollout used at eval time when we want h_T from a full
        repo history in one go.

        Args:
            diff_embs:        [B, T, diff_input_dim]
            repo_state_emb_0: [B, repo_state_dim]
            lengths:          [B] sequence lengths (defaults to T).

        Returns:
            h_T: [B, hidden_dim]
        """
        h = self.init_hidden(repo_state_emb_0)
        x = self.diff_proj(diff_embs.float())
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed, h)
        else:
            _, h_n = self.gru(x, h)
        return self.output_norm(h_n[-1])  # [B, hidden_dim]


# ---------------------------------------------------------------------------
# Parquet loaders for the v2 datasets
# ---------------------------------------------------------------------------

def _list_to_f32_array(col, dim: int) -> np.ndarray:
    """``pa.array([list<f16>, ...])`` -> np.ndarray[N, dim] float32."""
    out = np.zeros((len(col), dim), dtype=np.float32)
    for i, v in enumerate(col.to_pylist()):
        # v is a list of length `dim` of (float16-encoded) floats. python lists
        # promote to float64 so we just convert back to float32.
        out[i] = v
    return out


@dataclass
class SnapshotRow:
    repo_id: str
    commit_sha: str
    commit_index: int
    in_repo_split: str
    cross_repo_split: str
    repo_state_embedding: np.ndarray  # fp32 [2048]


@dataclass
class CommitRow:
    repo_id: str
    commit_sha: str
    commit_index: int
    in_repo_split: str
    cross_repo_split: str
    diff_embedding: np.ndarray         # fp32 [2048]
    repo_state_embedding: np.ndarray   # fp32 [2048]


# ---------------------------------------------------------------------------
# Per-step GRU input selection (ablation: diff vs. repo-state vs. concat)
# ---------------------------------------------------------------------------

PER_STEP_INPUT_MODES = ("diff", "repo_state", "concat")


def per_step_input_dim(diff_dim: int, repo_dim: int, mode: str) -> int:
    """Dimensionality of the per-step GRU input for the given ablation mode."""
    if mode == "diff":
        return diff_dim
    if mode == "repo_state":
        return repo_dim
    if mode == "concat":
        return diff_dim + repo_dim
    raise ValueError(f"unknown per_step_input mode: {mode!r}")


def make_per_step_input(row: "CommitRow", mode: str) -> np.ndarray:
    """Return the per-step GRU input vector for ``row`` under ``mode``.

    * ``diff``       -> per-commit diff_embedding (default / original v2 GRU).
    * ``repo_state`` -> per-commit whole-repo repo_state_embedding.
    * ``concat``     -> [diff_embedding ; repo_state_embedding].

    NB: ``h_0`` is *always* initialized from ``rows[0].repo_state_embedding``
    regardless of mode -- only the per-step recurrent input changes.
    """
    if mode == "diff":
        return row.diff_embedding
    if mode == "repo_state":
        return row.repo_state_embedding
    if mode == "concat":
        return np.concatenate([row.diff_embedding, row.repo_state_embedding])
    raise ValueError(f"unknown per_step_input mode: {mode!r}")


def load_snapshot_rows(parquet_path: Path,
                       in_repo_splits: Optional[List[str]] = None,
                       repo_ids: Optional[List[str]] = None,
                       embedding_col: str = "repo_state_embedding",
                       ) -> List[SnapshotRow]:
    """Read a snapshot parquet (commits/{train,ir_val,ir_test,cr_val,cr_test}.parquet
    from the Code2LoRA-snapshots dataset) and return a flat list of rows
    with the requested embedding column materialized as float32."""
    needed = [
        "repo_id", "commit_sha", "commit_index", "in_repo_split",
        "cross_repo_split", embedding_col,
    ]
    ds = pads.dataset(str(parquet_path), format="parquet")
    filters: List[Any] = []
    if in_repo_splits:
        if len(in_repo_splits) == 1:
            filters.append(pc.equal(pads.field("in_repo_split"),
                                    pa.scalar(in_repo_splits[0])))
        else:
            filters.append(pc.is_in(
                pads.field("in_repo_split"),
                value_set=pa.array(in_repo_splits, type=pa.string()),
            ))
    if repo_ids:
        filters.append(pc.is_in(
            pads.field("repo_id"),
            value_set=pa.array(repo_ids, type=pa.string()),
        ))
    flt = None
    for f in filters:
        flt = f if flt is None else (flt & f)
    table = ds.to_table(columns=needed, filter=flt)
    n = table.num_rows
    if n == 0:
        return []
    dim = len(table.column(embedding_col)[0].as_py())
    embs = _list_to_f32_array(table.column(embedding_col), dim)
    rows: List[SnapshotRow] = []
    repo_col = table.column("repo_id").to_pylist()
    sha_col = table.column("commit_sha").to_pylist()
    idx_col = table.column("commit_index").to_pylist()
    irs_col = table.column("in_repo_split").to_pylist()
    crs_col = table.column("cross_repo_split").to_pylist()
    for i in range(n):
        rows.append(SnapshotRow(
            repo_id=repo_col[i], commit_sha=sha_col[i],
            commit_index=int(idx_col[i]),
            in_repo_split=irs_col[i] or "",
            cross_repo_split=crs_col[i] or "",
            repo_state_embedding=embs[i],
        ))
    return rows


def load_commit_rows_for_gru(parquet_path: Path,
                             repo_ids: Optional[List[str]] = None,
                             ) -> Dict[str, List[CommitRow]]:
    """Read a v2 commits parquet (commit_parquet_hf_v2/commits/*.parquet) and
    group rows by repo_id, sorted by commit_index. Used by the GRU trainer
    which needs the full chronological sequence per repo.

    Returns ``{repo_id: [CommitRow sorted by commit_index, ...]}``.
    """
    needed = [
        "repo_id", "commit_sha", "commit_index", "in_repo_split",
        "cross_repo_split", "diff_embedding", "repo_state_embedding",
    ]
    ds = pads.dataset(str(parquet_path), format="parquet")
    flt = None
    if repo_ids:
        flt = pc.is_in(pads.field("repo_id"),
                       value_set=pa.array(repo_ids, type=pa.string()))
    table = ds.to_table(columns=needed, filter=flt)
    n = table.num_rows
    if n == 0:
        return {}
    diff_dim = len(table.column("diff_embedding")[0].as_py())
    repo_dim = len(table.column("repo_state_embedding")[0].as_py())
    diff_arr = _list_to_f32_array(table.column("diff_embedding"), diff_dim)
    repo_arr = _list_to_f32_array(table.column("repo_state_embedding"), repo_dim)
    repo_col = table.column("repo_id").to_pylist()
    sha_col = table.column("commit_sha").to_pylist()
    idx_col = table.column("commit_index").to_pylist()
    irs_col = table.column("in_repo_split").to_pylist()
    crs_col = table.column("cross_repo_split").to_pylist()

    by_repo: Dict[str, List[CommitRow]] = {}
    for i in range(n):
        r = repo_col[i]
        by_repo.setdefault(r, []).append(CommitRow(
            repo_id=r, commit_sha=sha_col[i],
            commit_index=int(idx_col[i]),
            in_repo_split=irs_col[i] or "",
            cross_repo_split=crs_col[i] or "",
            diff_embedding=diff_arr[i],
            repo_state_embedding=repo_arr[i],
        ))
    for rows in by_repo.values():
        rows.sort(key=lambda r: r.commit_index)
    return by_repo


# ---------------------------------------------------------------------------
# QnA parquet loader
# ---------------------------------------------------------------------------

@dataclass
class QnaRow:
    repo_id: str
    commit_sha: str
    commit_index: int
    in_repo_split: str
    cross_repo_split: str
    test_file: str
    test_function: str
    prefix: str
    target: str
    lineno: int = 0
    col_offset: int = 0
    assertion_event_id: str = ""


def load_qna_rows(parquet_path: Path,
                  in_repo_splits: Optional[List[str]] = None,
                  repo_ids: Optional[List[str]] = None,
                  commit_keys: Optional[List[Tuple[str, str]]] = None,
                  ) -> List[QnaRow]:
    """Read QnAs and optionally filter by (in_repo_split, repo_id) or by
    a list of (repo_id, commit_sha) keys (for per-commit eval suites).
    """
    ds = pads.dataset(str(parquet_path), format="parquet")
    needed = [c for c in (
        "repo_id", "commit_sha", "commit_index", "in_repo_split",
        "cross_repo_split", "test_file", "test_function", "prefix", "target",
        "lineno", "col_offset", "assertion_event_id",
    ) if c in ds.schema.names]
    filters: List[Any] = []
    if in_repo_splits:
        filters.append(pc.is_in(
            pads.field("in_repo_split"),
            value_set=pa.array(in_repo_splits, type=pa.string()),
        ))
    if repo_ids:
        filters.append(pc.is_in(
            pads.field("repo_id"),
            value_set=pa.array(repo_ids, type=pa.string()),
        ))
    flt = None
    for f in filters:
        flt = f if flt is None else (flt & f)
    table = ds.to_table(columns=needed, filter=flt)
    rows: List[QnaRow] = []
    cols = set(table.column_names)
    repo_col = table.column("repo_id").to_pylist()
    sha_col = table.column("commit_sha").to_pylist()
    idx_col = table.column("commit_index").to_pylist() if "commit_index" in cols else [-1] * table.num_rows
    irs_col = table.column("in_repo_split").to_pylist() if "in_repo_split" in cols else [""] * table.num_rows
    crs_col = table.column("cross_repo_split").to_pylist() if "cross_repo_split" in cols else [""] * table.num_rows
    tf_col = table.column("test_file").to_pylist() if "test_file" in cols else [""] * table.num_rows
    tfun_col = table.column("test_function").to_pylist() if "test_function" in cols else [""] * table.num_rows
    prefix_col = table.column("prefix").to_pylist()
    target_col = table.column("target").to_pylist()
    lineno_col = table.column("lineno").to_pylist() if "lineno" in cols else [0] * table.num_rows
    col_off_col = table.column("col_offset").to_pylist() if "col_offset" in cols else [0] * table.num_rows
    eid_col = table.column("assertion_event_id").to_pylist() if "assertion_event_id" in cols else [""] * table.num_rows
    keep_set = set(commit_keys) if commit_keys else None
    for i in range(table.num_rows):
        key = (repo_col[i], sha_col[i])
        if keep_set is not None and key not in keep_set:
            continue
        rows.append(QnaRow(
            repo_id=repo_col[i], commit_sha=sha_col[i],
            commit_index=int(idx_col[i] if idx_col[i] is not None else -1),
            in_repo_split=irs_col[i] or "",
            cross_repo_split=crs_col[i] or "",
            test_file=tf_col[i] or "",
            test_function=tfun_col[i] or "",
            prefix=prefix_col[i] or "",
            target=target_col[i] or "",
            lineno=int(lineno_col[i]) if lineno_col[i] is not None else 0,
            col_offset=int(col_off_col[i]) if col_off_col[i] is not None else 0,
            assertion_event_id=eid_col[i] or "",
        ))
    return rows


__all__ = [
    "LoRA",
    "ModuleSpec",
    "get_module_specs",
    "replace_with_lora",
    "inject_lora_weights",
    "discover_module_types_and_dims",
    "Code2LoRAHead",
    "CommitGRU",
    "SnapshotRow",
    "CommitRow",
    "QnaRow",
    "load_snapshot_rows",
    "load_commit_rows_for_gru",
    "load_qna_rows",
]
