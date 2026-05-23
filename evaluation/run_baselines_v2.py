#!/usr/bin/env python3
"""Unified v2 baseline evaluator -- scores **pretrained**, **FFT**,
**SLoRA**, **RAG**, **DRC**, **Text2LoRA**, and **Doc2LoRA** on exactly
the same (repo, commit, qna) triples as the v2 Code2LoRA trainers
(``hypernetwork/train_code2lora_{static,gru}_v2.py``).

The evaluator pulls QnAs from a v2 parquet suite -- by default the
*Dataset A* (GRU-aligned) parquets at
``$SCRATCH/REPO_DATASET/commit_parquet_hf_v2/qna/<suite>.parquet`` -- and
walks (repo_id, commit_sha) groups, scoring every QnA at every commit.
For each suite it dumps a per-commit JSON suitable for decay-curve plotting:

    {
        "summary": {"suite": ..., "n_qnas": ..., "exact_match": ..., ...,
                    "exact_match_ci": [lo, hi], ...},
        "per_commit": [
            {"repo_id": ..., "commit_sha": ..., "commit_index": ...,
             "n_qnas": ..., "exact_match": ..., "edit_similarity": ...,
             "code_bleu": ...},
            ...
        ]
    }

Both v2 Code2LoRA trainers emit the same shape so plots overlay cleanly.

Methods
-------
* ``pretrained``  No adaptation -- raw Qwen2.5-Coder-1.5B.
* ``fft``         Full fine-tune; ``--ckpt`` must be a HF model directory
                  (output of ``baselines/finetuned/train_fft_v2.py``).
* ``slora``       Single LoRA shared across repos; ``--ckpt`` must contain
                  a PEFT adapter (output of
                  ``baselines/single_lora/train_slora_v2.py``).
* ``rag``         Pretrained + top-k retrieval over a per-commit chunk
                  index (built by
                  ``evaluation/build_rag_cache_per_commit.py``).
                  Requires ``--rag-cache-dir`` and optionally
                  ``--rag-top-k`` (default 3) and
                  ``--rag-embed-model-name``.
* ``drc``         Pretrained + dependency-resolved context prepended to
                  the prefix (precomputed by
                  ``evaluation/build_drc_cache_per_commit.py``).
                  Requires ``--drc-cache-dir`` and optionally
                  ``--drc-max-tokens`` (default 4096; adaptive
                  compression via ``evaluation.compress_context``).
* ``text2lora``   Code-conditioned Text2LoRA hypernet (Code-SFT v2
                  variant). Generates per-repo LoRA weights from a
                  precomputed 2048-d code embedding (the v2 anchor-
                  commit ``repo_state_embedding``) and applies them
                  through the PEFT wrapper before scoring. Requires
                  ``--text2lora-hypermod-dir`` (run dir produced by
                  ``baselines/text2lora/train_code_sft.py``) and
                  ``--text2lora-code-emb-path`` (``code_embeddings_v2.pt``
                  from ``baselines/text2lora/extract_code_embeddings_v2.py``).
                  Repos without an embedding fall back to pretrained.
* ``doc2lora``    Doc-to-LoRA (Sakana D2L). Internalizes the per-
                  ``(repo, commit_sha)`` DRC context into LoRA weights
                  via ``ModulatedPretrainedModel.internalize`` and
                  generates with ``model.base_model.generate``. Requires
                  ``--doc2lora-ckpt`` (a ``pytorch_model.bin`` produced
                  by ``baselines/doc2lora/train_doc2lora_v2.sh``) and
                  ``--doc2lora-drc-cache-dir`` (the same per-commit DRC
                  cache the DRC baseline uses). Commits without a DRC
                  cache fall back to the un-adapted base model.

Truncation policy matches the v2 Code2LoRA trainers: **left-truncate, left-pad**
so the assertion-adjacent code is preserved.

Usage::

    sbatch scripts/slurm/eval_baselines_v2.sh   # default = pretrained, all 4 suites
    METHOD=fft CKPT=$CKPT_DIR/FFT_V2/checkpoint-final sbatch ...
    METHOD=slora CKPT=$CKPT_DIR/SLORA_V2/adapter sbatch ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HYP = _ROOT / "hypernetwork"
if str(_HYP) not in sys.path:
    sys.path.insert(0, str(_HYP))

from code2lora_core import load_qna_rows  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate_metrics_with_ci,
    compute_metrics,
    format_ci,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_QNA_DIR = "/scratch/lhotsko/REPO_DATASET/code2lora_snapshots_hf/qna"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _slug_repo(repo_name: str) -> str:
    """Convert ``foo/bar`` -> ``foo__bar`` (matches the slug used by
    ``baselines/text2lora/extract_code_embeddings_v2.py`` and the v1
    ``extract_code_embeddings.py``)."""
    return repo_name.replace("/", "__")


def _load_text2lora_bundle(hypermod_dir: Path, text2lora_dir: Path,
                           code_emb_path: Path, device: torch.device,
                           dtype: torch.dtype):
    """Build (tokenizer, peft_model, hypermod, layer_indices, code_embs)
    for the Text2LoRA Code-SFT baseline.

    Mirrors the manual checkpoint-load path in
    ``baselines/text2lora/evaluate_text2lora_code.py`` (the canonical
    ``load_hypermod_checkpoint`` infers ``task_emb_size`` from the
    embedding LLM, but here the code embedding *is* the task embedding,
    so we override it from the .pt artifact).
    """
    # Local imports -- only the text2lora method needs them.
    import argparse as _argparse
    import yaml as _yaml
    from peft import PeftConfig as _PeftConfig
    from peft import get_peft_config as _get_peft_config
    from hyper_llm_modulator.hyper_modulator import (
        create_hypermod as _create_hypermod,
    )
    from hyper_llm_modulator.utils import get_layers as _get_layers
    from hyper_llm_modulator.utils.model_loading import (
        get_model_and_tokenizer as _get_model_and_tokenizer,
    )

    hypermod_dir = hypermod_dir.expanduser().resolve()
    text2lora_dir = text2lora_dir.expanduser().resolve()
    code_emb_path = code_emb_path.expanduser().resolve()

    print(f"[load] Text2LoRA: hypermod_dir={hypermod_dir}", flush=True)
    print(f"[load]            code_emb_path={code_emb_path}", flush=True)
    print(f"[load]            text2lora_dir={text2lora_dir}", flush=True)

    # ---- Code embeddings (one vec per repo) ----
    code_embs = torch.load(
        str(code_emb_path), map_location="cpu", weights_only=True,
    )
    task_emb_size = int(next(iter(code_embs.values())).shape[-1])
    print(f"[load] code_embs: {len(code_embs)} repos, dim={task_emb_size}",
          flush=True)

    # ---- Re-build model + PEFT wrapper exactly the way training did ----
    # The hypernet's adapter_config.json + args.yaml were written next to
    # ``hypermod.pt`` by ``save_hypermod_checkpoint``.
    ckpt_path = hypermod_dir / "hypermod.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"text2lora hypermod.pt not found: {ckpt_path}")
    hargs = _argparse.Namespace(
        **_yaml.safe_load((hypermod_dir / "args.yaml").read_text())
    )
    peft_config = _get_peft_config(
        _PeftConfig.from_json_file(str(hypermod_dir / "adapter_config.json"))
    )
    state_dict = torch.load(str(ckpt_path), map_location=device)

    # Some chat-template path lookups inside get_model_and_tokenizer
    # resolve relative to text2lora/; cd in/out around the call.
    orig_cwd = os.getcwd()
    os.chdir(str(text2lora_dir))
    try:
        model, tokenizer = _get_model_and_tokenizer(
            hargs.model_dir,
            train=False,
            requires_grad=False,
            peft_config=peft_config,
            model_kwargs={
                "output_hidden_states": True,
                "output_attentions": False,
                "torch_dtype": dtype,
            },
            device=device,
        )
    finally:
        os.chdir(orig_cwd)

    layer_indices = torch.tensor(
        list(range(len(_get_layers(model)))),
        dtype=torch.long, device=device,
    )

    hypermod = _create_hypermod(
        hargs, peft_config.peft_type.lower(), device, model,
        layer_indices, task_emb_size, from_scratch=False,
    )
    info = hypermod.load_state_dict(state_dict, strict=False)
    print(f"[load] hypermod state_dict: missing={len(info.missing_keys)}, "
          f"unexpected={len(info.unexpected_keys)}", flush=True)
    hypermod.eval()
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False
    for p in hypermod.parameters():
        p.requires_grad = False

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, hypermod, layer_indices, code_embs


# ---------------------------------------------------------------------------
# Doc-to-LoRA bundle: load ModulatedPretrainedModel + per-commit internalize
# ---------------------------------------------------------------------------

def _resolve_doc2lora_ckpt(ckpt: Path) -> Path:
    """Accept any of: ``pytorch_model.bin`` file, ``checkpoint-XXXX`` dir,
    or run-dir (containing ``checkpoint-*`` sub-dirs). Return the
    resolved file path."""
    p = ckpt.expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "pytorch_model.bin"
        if cand.exists():
            return cand
        sub = sorted(p.glob("checkpoint-*"),
                     key=lambda d: int(d.name.split("-")[-1])
                     if d.name.split("-")[-1].isdigit() else -1)
        if sub:
            cand = sub[-1] / "pytorch_model.bin"
            if cand.exists():
                return cand
    raise SystemExit(f"could not resolve doc2lora ckpt at {ckpt}; "
                     f"expected a pytorch_model.bin or a checkpoint dir.")


def _load_doc2lora_bundle(ckpt: Path, drc_cache_dir: Path,
                          max_ctx_tokens: int, device: torch.device,
                          dtype: torch.dtype):
    """Build (tokenizer, ModulatedPretrainedModel) for the D2L baseline,
    plus a context tokenizer and the per-commit DRC cache directory
    stashed on the model for later use by ``score_suite``.
    """
    # Local imports -- only the doc2lora method needs them.
    from ctx_to_lora.model_loading import get_tokenizer as _get_tok
    from ctx_to_lora.modeling.hypernet import (
        ModulatedPretrainedModel as _ModulatedPretrainedModel,
    )

    ckpt_path = _resolve_doc2lora_ckpt(ckpt)
    drc_cache_dir = drc_cache_dir.expanduser().resolve()
    print(f"[load] Doc2LoRA: ckpt={ckpt_path}", flush=True)
    print(f"[load]           drc_cache_dir={drc_cache_dir}", flush=True)
    print(f"[load]           max_ctx_tokens={max_ctx_tokens}", flush=True)

    state_dict = torch.load(str(ckpt_path), map_location=device,
                            weights_only=False)
    model = _ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False,
    )
    model.eval()
    model.to(device)

    base_name = model.base_model.config.name_or_path
    tokenizer = _get_tok(base_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
    if ctx_name is None:
        ctx_name = base_name
    ctx_tokenizer = _get_tok(ctx_name)

    print(f"[load]           base_model={base_name}", flush=True)
    print(f"[load]           ctx_encoder={ctx_name}", flush=True)

    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model, ctx_tokenizer


def _d2l_build_commit_doc(contexts: Dict[str, Dict[str, Any]],
                          ctx_tokenizer, max_ctx_tokens: int) -> str:
    """Concatenate unique ``extracted_code`` snippets from this commit's
    DRC cache into a single document string and truncate to
    ``max_ctx_tokens`` via the context encoder's tokenizer. Mirrors the
    document-construction logic used by
    ``baselines/doc2lora/generate_teacher_logprobs_v2.py`` so train and
    eval see the same per-commit doc shape.
    """
    snippets: List[str] = []
    seen: set = set()
    for v in contexts.values():
        if not isinstance(v, dict):
            continue
        code = v.get("extracted_code") or ""
        if not code or code in seen:
            continue
        seen.add(code)
        snippets.append(code)
    snippets.sort()
    doc_text = "\n\n".join(snippets)
    if not doc_text:
        return ""
    ids = ctx_tokenizer.encode(doc_text, add_special_tokens=False)
    if len(ids) > max_ctx_tokens:
        ids = ids[:max_ctx_tokens]
        doc_text = ctx_tokenizer.decode(ids, skip_special_tokens=True)
    return doc_text


@torch.no_grad()
def _doc2lora_internalize_commit(model, drc_cache_dir: Path, repo_id: str,
                                 commit_sha: str, ctx_tokenizer,
                                 max_ctx_tokens: int) -> bool:
    """Load this commit's DRC, build the doc, ``model.reset() + internalize``.
    Returns True on success, False if no DRC text was found (caller will
    keep the previous-commit LoRA or fall back to the base model)."""
    contexts = _load_drc_commit_contexts(drc_cache_dir, repo_id, commit_sha)
    if not contexts:
        return False
    doc_text = _d2l_build_commit_doc(contexts, ctx_tokenizer, max_ctx_tokens)
    if not doc_text:
        return False
    model.reset()
    model.internalize(doc_text)
    return True


@torch.no_grad()
def _text2lora_inject_repo(model, hypermod, layer_indices, code_embs,
                           repo_id: str, device: torch.device) -> bool:
    """Generate this repo's LoRA from its code embedding and apply it
    in-place to ``model``. Returns ``True`` if a repo-specific adapter
    was installed, ``False`` if no embedding was found (caller can fall
    back to a zero-LoRA baseline)."""
    from peft.utils import set_peft_model_state_dict as _set_peft_sd

    slug = _slug_repo(repo_id)
    emb = code_embs.get(slug)
    if emb is None:
        return False
    task_emb = emb.to(device=device, dtype=torch.float32)
    if task_emb.dim() == 1:
        task_emb = task_emb.unsqueeze(0)
    encoder_out = hypermod.task_encoder(task_emb)
    encoded = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded)
    _set_peft_sd(model, lora_sd)
    return True


def load_model_for_method(method: str, base_model_name: str, ckpt: Optional[Path],
                          device: torch.device,
                          dtype: torch.dtype = torch.bfloat16,
                          *,
                          text2lora_hypermod_dir: Optional[Path] = None,
                          text2lora_code_emb_path: Optional[Path] = None,
                          text2lora_dir: Optional[Path] = None,
                          doc2lora_ckpt: Optional[Path] = None,
                          doc2lora_drc_cache_dir: Optional[Path] = None,
                          doc2lora_max_ctx_tokens: int = 4096):
    """Return (tokenizer, model[, extras]) ready for inference, all params frozen.

    ``rag`` and ``drc`` use the pretrained backbone with no adapter -- only
    the *input* changes (retrieved chunks or DRC-extracted code is prepended
    to the prefix). ``--ckpt`` is ignored for those methods.

    For ``text2lora`` we additionally stash the hypermod, ``layer_indices``,
    and ``code_embs`` dict as private attributes on the returned model so
    ``score_suite`` can call ``hypermod.gen_lora`` per-repo without a
    second loader-style API.

    For ``doc2lora`` we stash ``_d2l_drc_cache_dir``, ``_d2l_max_ctx_tokens``,
    ``_d2l_ctx_tokenizer`` on the ``ModulatedPretrainedModel`` so the
    score loop can call ``model.reset() + internalize(per-commit-DRC)``
    once per (repo, commit_sha) group.
    """
    if method == "doc2lora":
        if doc2lora_ckpt is None or doc2lora_drc_cache_dir is None:
            raise SystemExit("--doc2lora-ckpt and --doc2lora-drc-cache-dir "
                             "are required for --method doc2lora")
        tok, model, ctx_tokenizer = _load_doc2lora_bundle(
            doc2lora_ckpt, doc2lora_drc_cache_dir,
            int(doc2lora_max_ctx_tokens), device=device, dtype=dtype,
        )
        model._d2l_mode = True
        model._d2l_drc_cache_dir = doc2lora_drc_cache_dir.expanduser().resolve()
        model._d2l_max_ctx_tokens = int(doc2lora_max_ctx_tokens)
        model._d2l_ctx_tokenizer = ctx_tokenizer
        return tok, model

    if method == "text2lora":
        if text2lora_hypermod_dir is None or text2lora_code_emb_path is None:
            raise SystemExit("--text2lora-hypermod-dir and "
                             "--text2lora-code-emb-path are required for "
                             "--method text2lora")
        if text2lora_dir is None:
            text2lora_dir = Path(_ROOT) / "text2lora"
        tok, model, hypermod, layer_indices, code_embs = _load_text2lora_bundle(
            text2lora_hypermod_dir, text2lora_dir, text2lora_code_emb_path,
            device=device, dtype=dtype,
        )
        # Stash text2lora state on the model so score_suite can pick it up.
        model._t2l_hypermod = hypermod
        model._t2l_layer_indices = layer_indices
        model._t2l_code_embs = code_embs
        model._t2l_device = device
        return tok, model

    # ``rag`` and ``drc`` use the pretrained backbone with no adapter --
    # only the *input* changes (retrieved chunks or DRC-extracted code is
    # prepended to the prefix). Fall through to the pretrained loader.
    load_method = "pretrained" if method in ("rag", "drc") else method
    if load_method == "fft":
        if ckpt is None:
            raise SystemExit("--ckpt is required for method=fft")
        print(f"[load] FFT -> AutoModelForCausalLM.from_pretrained({ckpt})",
              flush=True)
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    elif load_method == "slora":
        if ckpt is None:
            raise SystemExit("--ckpt is required for method=slora")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise SystemExit("peft must be installed for method=slora") from e
        print(f"[load] SLoRA: base={base_model_name}, adapter={ckpt}",
              flush=True)
        tok = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        if hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
    else:  # pretrained
        print(f"[load] Pretrained {base_model_name}", flush=True)
        tok = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


# ---------------------------------------------------------------------------
# Tokenization (left-truncate, left-pad) -- mirrors v2 Code2LoRA trainers
# ---------------------------------------------------------------------------

def _prepare_prefix_ids(tokenizer, prefix: str, max_input_tokens: int,
                        bos_id: Optional[int]) -> List[int]:
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if bos_id is not None:
        ids = [bos_id] + ids
    if len(ids) > max_input_tokens:
        ids = ids[-max_input_tokens:]
    return ids


def _get_bos_id(tok) -> Optional[int]:
    bid = getattr(tok, "bos_token_id", None)
    if bid is not None:
        return int(bid)
    eid = getattr(tok, "eos_token_id", None)
    if eid is not None:
        return int(eid)
    return None


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_batch(model, tokenizer, device, input_ids_list: List[List[int]],
                    max_new_tokens: int) -> List[str]:
    """Left-pad a batch, generate, decode only the new tokens per sample.

    For Doc-to-LoRA's ``ModulatedPretrainedModel`` we route through
    ``model.base_model.generate`` because the outer model wraps the base
    with its own ``.generate`` API (which expects context tokens, not
    plain text inputs).
    """
    L = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    bs = len(input_ids_list)
    input_ids = torch.full((bs, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, L), dtype=torch.long, device=device)
    lens = []
    for i, ids in enumerate(input_ids_list):
        n = len(ids)
        input_ids[i, L - n:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, L - n:] = 1
        lens.append(n)
    gen_target = (model.base_model
                  if getattr(model, "_d2l_mode", False)
                  else model)
    out = gen_target.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    # Strip the prefix off each row; whatever's left is the prediction.
    decoded: List[str] = []
    for i in range(bs):
        gen = out[i, L:].tolist()
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True))
    return decoded


# ---------------------------------------------------------------------------
# RAG: per-commit chunk index loader + retrieval
# ---------------------------------------------------------------------------

def _rag_cache_path(cache_dir: Path, repo_id: str, sha: str) -> Path:
    safe = repo_id.replace("/", "__")
    return cache_dir / f"{safe}__{sha}.pt"


def _load_rag_commit_index(cache_dir: Path, repo_id: str,
                           sha: str) -> Dict[str, Any]:
    """Return ``{"chunks": [...], "embeddings": float32[N, D] | None}``.
    Missing/empty cache => no retrieval; the QnA falls back to plain prefix."""
    path = _rag_cache_path(cache_dir, repo_id, sha)
    if not path.exists():
        return {"chunks": [], "embeddings": None}
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [rag] failed to load {path}: {type(e).__name__}: {e}",
              flush=True)
        return {"chunks": [], "embeddings": None}
    embs = data.get("embeddings")
    if embs is not None and not isinstance(embs, torch.Tensor):
        embs = torch.as_tensor(embs)
    if embs is not None:
        embs = embs.float()
    return {"chunks": list(data.get("chunks") or []), "embeddings": embs}


@torch.inference_mode()
def _embed_rag_queries(texts: List[str], embed_model, embed_tokenizer,
                       device, max_length: int = 512) -> torch.Tensor:
    """Attention-mean pooled, L2-normalised query embeddings (CPU)."""
    enc = embed_tokenizer(
        texts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = embed_model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
    return pooled.cpu()


def _rag_retrieve_topk(query_emb: torch.Tensor, index: Dict[str, Any],
                       top_k: int) -> List[str]:
    if not index["chunks"] or index["embeddings"] is None:
        return []
    embs = index["embeddings"]
    sims = (query_emb @ embs.T).squeeze(0)
    k = min(top_k, len(index["chunks"]))
    _, top_idx = sims.topk(k)
    return [index["chunks"][i] for i in top_idx.tolist()]


def _format_rag_prompt(prefix: str, retrieved_chunks: List[str]) -> str:
    """Plain code-completion prompt: retrieved chunks then prefix."""
    if not retrieved_chunks:
        return prefix
    return "\n\n\n".join(retrieved_chunks) + "\n\n\n" + prefix


# ---------------------------------------------------------------------------
# DRC: per-commit context loader
# ---------------------------------------------------------------------------

def _drc_cache_path(cache_dir: Path, repo_id: str, sha: str) -> Path:
    safe = repo_id.replace("/", "__")
    return cache_dir / f"{safe}__{sha}.json"


def _load_drc_commit_contexts(cache_dir: Path, repo_id: str,
                              sha: str) -> Dict[str, Dict[str, Any]]:
    """Return the per-(test_file, lineno, col_offset)-keyed context map for
    this commit, or {} if missing."""
    path = _drc_cache_path(cache_dir, repo_id, sha)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [drc] failed to load {path}: {type(e).__name__}: {e}",
              flush=True)
        return {}
    return data.get("contexts") or {}


def _drc_key_for_qna(q: Dict[str, Any]) -> str:
    if q.get("assertion_event_id"):
        return q["assertion_event_id"]
    return f"{q.get('test_file', '')}::{int(q.get('lineno', 0))}" \
           f"::{int(q.get('col_offset', 0))}"


def _format_drc_prompt(prefix: str, drc_code: str) -> str:
    if not drc_code:
        return prefix
    return drc_code + "\n\n\n" + prefix


def score_suite(
    *,
    qna_path: Path,
    suite_name: str,
    model,
    tokenizer,
    device: torch.device,
    max_input_tokens: int = 4096,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    repo_limit: int = 0,
    qnas_per_commit_limit: int = 0,
    out_path: Path,
    bootstrap: int = 5000,
    shard_i: int = 0,
    num_shards: int = 1,
    # ---- Prefix-length cap (ablation) ------------------------------------
    # When > 0, each QnA's prefix is left-truncated to its last N tokens
    # **before** any RAG query / DRC compression / prompt assembly. Set to
    # ~256 to reproduce the static-snapshot prefix budget on the
    # commit-derived suites; default 0 keeps the QnA's native prefix.
    prefix_max_tokens: int = 0,
    # ---- Context-injection options (rag / drc) ---------------------------
    context_method: str = "none",   # "none" | "rag" | "drc"
    rag_cache_dir: Optional[Path] = None,
    rag_top_k: int = 3,
    rag_embed_model=None,
    rag_embed_tokenizer=None,
    rag_query_chars: int = 2000,
    drc_cache_dir: Optional[Path] = None,
    drc_max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Walk one v2 QnA parquet, group by (repo, commit), generate, score.

    Sharding (optional): when ``num_shards > 1``, only repos whose
    ``sorted_index % num_shards == shard_i`` are scored. The output JSON path
    is suffixed with ``_shard{i}of{n}`` so multiple shards can be combined
    later via ``evaluation/merge_eval_shards.py``.

    Resumability: results are written to disk **after every scored
    (repo, commit) group** -- so a SLURM timeout never loses more than the
    one commit currently in flight.
    """
    print(f"\n[suite {suite_name}] loading {qna_path} ...", flush=True)
    rows = load_qna_rows(qna_path)
    # Group by (repo_id, commit_sha) preserving commit_index.
    groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"commit_index": -1, "pairs": []}
    )
    for qr in rows:
        key = (qr.repo_id, qr.commit_sha)
        g = groups[key]
        g["commit_index"] = int(qr.commit_index)
        g["pairs"].append({
            "prefix": qr.prefix,
            "target": qr.target,
            "test_file": qr.test_file,
            "lineno": int(qr.lineno),
            "col_offset": int(qr.col_offset),
            "assertion_event_id": qr.assertion_event_id,
        })
    print(f"[suite {suite_name}] {len(rows):,} qnas across "
          f"{len(groups):,} (repo, commit) groups", flush=True)

    if repo_limit:
        repos_kept = sorted({r for (r, _) in groups.keys()})[:repo_limit]
        groups = {k: v for k, v in groups.items() if k[0] in repos_kept}
        print(f"[suite {suite_name}] limited to {repo_limit} repos -> "
              f"{len(groups)} (repo, commit) groups", flush=True)

    # ---- Sharding by repo (deterministic round-robin on sorted repo ids) ----
    all_repos = sorted({r for (r, _) in groups.keys()})
    if num_shards > 1:
        kept_repos = {r for k, r in enumerate(all_repos) if k % num_shards == shard_i}
        groups = {k: v for k, v in groups.items() if k[0] in kept_repos}
        print(f"[suite {suite_name}] shard {shard_i+1}/{num_shards}: "
              f"{len(kept_repos)} of {len(all_repos)} repos "
              f"-> {len(groups)} (repo, commit) groups", flush=True)
    else:
        kept_repos = set(all_repos)

    bos_id = _get_bos_id(tokenizer)

    # Sort groups by (repo, commit_index) for deterministic per-commit output.
    group_keys = sorted(groups.keys(),
                        key=lambda k: (k[0], groups[k]["commit_index"]))

    # ---- Resume from existing on-disk shard file if any ----
    per_commit_records: List[Dict[str, Any]] = []
    all_samples: List[Tuple[float, float, float]] = []
    done_keys: set = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if not prev.get("finalized"):
                per_commit_records = list(prev.get("per_commit", []))
                done_keys = {(r["repo_id"], r["commit_sha"])
                             for r in per_commit_records}
                # Reconstruct raw samples (em,ed,cb) from per-commit averages
                # weighted by n_qnas. This is an APPROXIMATION used only for
                # CI bootstrapping at the SHARD level -- merge_eval_shards.py
                # will recombine using the persisted raw_samples for correct
                # CIs when those are present.
                for r in per_commit_records:
                    n = int(r["n_qnas"])
                    em = float(r["exact_match"])
                    ed = float(r["edit_similarity"])
                    cb = float(r["code_bleu"])
                    for _ in range(n):
                        all_samples.append((em, ed, cb))
                print(f"[suite {suite_name}] resuming: {len(done_keys)} "
                      f"already-scored (repo, commit) groups on disk",
                      flush=True)
        except Exception as e:
            print(f"[suite {suite_name}] could not parse existing "
                  f"{out_path}: {e}; starting fresh", flush=True)

    t0 = time.time()
    n_done = len(done_keys)
    n_done_in_run = 0
    # Text2LoRA: re-inject the per-repo LoRA only when the repo changes.
    current_t2l_repo: Optional[str] = None
    t2l_state = getattr(model, "_t2l_hypermod", None)
    t2l_missing_repos: set = set()
    # Doc-to-LoRA: re-internalize the per-commit DRC at every new commit.
    d2l_mode = bool(getattr(model, "_d2l_mode", False))
    d2l_missing_commits: int = 0
    d2l_internalized_commits: int = 0
    for (repo_id, commit_sha) in group_keys:
        if (repo_id, commit_sha) in done_keys:
            continue
        g = groups[(repo_id, commit_sha)]
        pairs = g["pairs"]
        if qnas_per_commit_limit and len(pairs) > qnas_per_commit_limit:
            pairs = pairs[:qnas_per_commit_limit]

        # ---- Optional prefix-length cap (matches static-snapshot budget) -
        # Re-truncate each QnA's prefix to its last N tokens **before**
        # RAG query, DRC compression, and prompt assembly run -- so all
        # downstream stages see the same shortened prefix the static-suite
        # baseline saw at the corresponding step.
        if prefix_max_tokens and prefix_max_tokens > 0:
            for p in pairs:
                ids = tokenizer.encode(p["prefix"], add_special_tokens=False)
                if len(ids) > prefix_max_tokens:
                    ids = ids[-prefix_max_tokens:]
                    p["prefix"] = tokenizer.decode(ids, skip_special_tokens=True)

        # ---- Text2LoRA: regenerate the LoRA adapter when the repo changes
        if t2l_state is not None and repo_id != current_t2l_repo:
            ok = _text2lora_inject_repo(
                model,
                model._t2l_hypermod,
                model._t2l_layer_indices,
                model._t2l_code_embs,
                repo_id,
                model._t2l_device,
            )
            if not ok and repo_id not in t2l_missing_repos:
                t2l_missing_repos.add(repo_id)
                print(f"  [text2lora] WARN: no code embedding for "
                      f"{repo_id}; falling back to last loaded LoRA "
                      f"(or pretrained for the first repo).", flush=True)
            current_t2l_repo = repo_id

        # ---- Doc-to-LoRA: re-internalize this commit's DRC document.
        if d2l_mode:
            ok = _doc2lora_internalize_commit(
                model,
                model._d2l_drc_cache_dir,
                repo_id,
                commit_sha,
                model._d2l_ctx_tokenizer,
                model._d2l_max_ctx_tokens,
            )
            if ok:
                d2l_internalized_commits += 1
            else:
                d2l_missing_commits += 1
                # No DRC -> revert to un-adapted base model for this commit
                # to avoid bleeding the previous commit's LoRA across.
                model.reset()

        # ---- Lazy per-commit context loads (RAG and/or DRC) -------------
        rag_index: Optional[Dict[str, Any]] = None
        drc_contexts: Optional[Dict[str, Dict[str, Any]]] = None
        if context_method == "rag":
            assert rag_cache_dir is not None
            rag_index = _load_rag_commit_index(rag_cache_dir, repo_id, commit_sha)
        elif context_method == "drc":
            assert drc_cache_dir is not None
            drc_contexts = _load_drc_commit_contexts(drc_cache_dir, repo_id, commit_sha)
        # Lazy import: only needed for DRC (and only when there's something
        # to compress).
        if context_method == "drc":
            from evaluation.compress_context import compress_oracle_context  # noqa: E402

        # ---- Build (per-QnA) augmented prompts -------------------------
        prompts: List[str] = []
        if context_method == "rag" and rag_index and rag_index["chunks"]:
            queries = [p["prefix"][-rag_query_chars:] for p in pairs]
            q_embs = _embed_rag_queries(
                queries, rag_embed_model, rag_embed_tokenizer, device,
            )
            for p, qe in zip(pairs, q_embs):
                chunks = _rag_retrieve_topk(qe.unsqueeze(0), rag_index, rag_top_k)
                prompts.append(_format_rag_prompt(p["prefix"], chunks))
        elif context_method == "drc" and drc_contexts:
            for p in pairs:
                ctx = drc_contexts.get(_drc_key_for_qna(p), {})
                raw_code = ctx.get("extracted_code", "") if isinstance(ctx, dict) else ""
                if raw_code:
                    compressed = compress_oracle_context(
                        raw_code, p["prefix"], tokenizer,
                        max_tokens=drc_max_tokens,
                    )
                else:
                    compressed = ""
                prompts.append(_format_drc_prompt(p["prefix"], compressed))
        else:
            prompts = [p["prefix"] for p in pairs]

        commit_samples: List[Tuple[float, float, float]] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            inputs = [
                _prepare_prefix_ids(tokenizer, prompt,
                                    max_input_tokens, bos_id)
                for prompt in batch_prompts
            ]
            preds = _generate_batch(
                model, tokenizer, device, inputs,
                max_new_tokens=max_new_tokens,
            )
            for p, pred in zip(batch_pairs, preds):
                m = compute_metrics(pred, p["target"])
                em = 1.0 if m["exact_match"] else 0.0
                ed = float(m["edit_similarity"])
                cb = float(m["code_bleu"])
                commit_samples.append((em, ed, cb))
                all_samples.append((em, ed, cb))
        if commit_samples:
            n_c = len(commit_samples)
            em_m = sum(s[0] for s in commit_samples) / n_c
            ed_m = sum(s[1] for s in commit_samples) / n_c
            cb_m = sum(s[2] for s in commit_samples) / n_c
            per_commit_records.append({
                "repo_id": repo_id,
                "commit_sha": commit_sha,
                "commit_index": g["commit_index"],
                "n_qnas": n_c,
                "exact_match": em_m,
                "edit_similarity": ed_m,
                "code_bleu": cb_m,
            })
        n_done += 1
        n_done_in_run += 1
        # Per-(repo, commit) incremental write -- atomic via tmp+replace.
        _write_suite_json(out_path, suite_name, per_commit_records,
                          all_samples, bootstrap=0, finalized=False,
                          shard_i=shard_i, num_shards=num_shards,
                          n_total_groups=len(group_keys))
        if n_done_in_run % 25 == 0 or n_done == len(group_keys):
            elapsed = (time.time() - t0) / 60
            done_em = sum(s[0] for s in all_samples) / max(len(all_samples), 1)
            rate = n_done_in_run / max(elapsed, 1e-6)
            eta = (len(group_keys) - n_done) / max(rate, 1e-6)
            print(f"  [suite {suite_name} sh{shard_i+1}/{num_shards}] "
                  f"{n_done}/{len(group_keys)} groups "
                  f"({len(all_samples):,} qnas) "
                  f"running_EM={done_em:.4f} elapsed={elapsed:.1f}m "
                  f"ETA={eta:.1f}m", flush=True)

    summary = _summarize(all_samples, bootstrap=bootstrap)
    summary["suite"] = suite_name
    summary["n_qnas"] = len(all_samples)
    summary["n_scored_commits"] = len(per_commit_records)
    summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})

    _write_suite_json(out_path, suite_name, per_commit_records, all_samples,
                      bootstrap=bootstrap, finalized=True, summary=summary,
                      shard_i=shard_i, num_shards=num_shards,
                      n_total_groups=len(group_keys))
    print(f"[suite {suite_name}] shard {shard_i+1}/{num_shards} DONE  "
          f"EM={summary['exact_match']:.4f}  "
          f"EditSim={summary['edit_similarity']:.4f}  "
          f"BLEU={summary['code_bleu']:.4f}  ({len(all_samples):,} qnas, "
          f"{len(per_commit_records):,} commits)", flush=True)
    return summary


def _summarize(samples: List[Tuple[float, float, float]],
               bootstrap: int = 5000) -> Dict[str, Any]:
    if not samples:
        return {"n_qnas": 0, "exact_match": 0.0, "edit_similarity": 0.0,
                "code_bleu": 0.0}
    metric_dicts = [
        {"exact_match": bool(em), "edit_similarity": ed, "code_bleu": cb}
        for (em, ed, cb) in samples
    ]
    agg = aggregate_metrics_with_ci(metric_dicts, n_resamples=int(bootstrap))
    out: Dict[str, Any] = {}
    for k, v in agg.items():
        if isinstance(v, dict) and "mean" in v:
            # ``v`` is the per-metric dict returned by bootstrap_ci:
            # {mean, low, high, n, n_resamples, ci (=conf level, not bounds)}.
            out[k] = float(v["mean"])
            out[f"{k}_ci"] = [float(v.get("low", 0.0)),
                              float(v.get("high", 0.0))]
            out[f"{k}_n"] = int(v.get("n", 0))
        else:
            out[k] = v
    return out


def _write_suite_json(out_path: Path, suite_name: str,
                      per_commit_records: List[Dict[str, Any]],
                      all_samples: List[Tuple[float, float, float]],
                      *,
                      bootstrap: int,
                      finalized: bool,
                      summary: Optional[Dict[str, Any]] = None,
                      shard_i: int = 0,
                      num_shards: int = 1,
                      n_total_groups: int = 0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary is None:
        # bootstrap=0 means "skip CIs" -- we recompute them at merge time.
        summary = _summarize(all_samples, bootstrap=bootstrap) if bootstrap > 0 else {
            "n_qnas": len(all_samples),
            "exact_match": (sum(s[0] for s in all_samples) /
                            max(len(all_samples), 1)) if all_samples else 0.0,
            "edit_similarity": (sum(s[1] for s in all_samples) /
                                max(len(all_samples), 1)) if all_samples else 0.0,
            "code_bleu": (sum(s[2] for s in all_samples) /
                          max(len(all_samples), 1)) if all_samples else 0.0,
        }
        summary["suite"] = suite_name
        summary["n_qnas"] = len(all_samples)
        summary["n_scored_commits"] = len(per_commit_records)
        summary["n_repos"] = len({r["repo_id"] for r in per_commit_records})
    payload: Dict[str, Any] = {
        "finalized": finalized,
        "shard_i": int(shard_i),
        "num_shards": int(num_shards),
        "n_total_groups": int(n_total_groups),
        "summary": summary,
        "per_commit": per_commit_records,
    }
    # Persist raw per-sample (em, ed, cb) tuples too so a downstream merge
    # can compute exact bootstrap CIs on the union across shards. Stored as
    # parallel arrays (smaller JSON than list-of-tuples).
    if finalized:
        payload["raw_samples"] = {
            "exact_match": [int(s[0]) for s in all_samples],
            "edit_similarity": [float(s[1]) for s in all_samples],
            "code_bleu": [float(s[2]) for s in all_samples],
        }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True,
                    choices=["pretrained", "fft", "slora", "rag", "drc",
                             "text2lora", "doc2lora"])
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to FFT checkpoint dir (method=fft) or "
                         "SLoRA adapter dir (method=slora). Unused for pretrained.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="HF id of the base model -- used as-is for pretrained "
                         "and SLoRA; for FFT we load directly from --ckpt.")
    ap.add_argument("--qna-dir", default=DEFAULT_QNA_DIR,
                    help="Directory with v2 qna parquets (qna/<suite>.parquet).")
    ap.add_argument("--suites", nargs="+",
                    default=["ir_val", "ir_test", "cr_val", "cr_test"],
                    choices=["ir_val", "ir_test", "cr_val", "cr_test"])
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--max-input-tokens", type=int, default=4096,
                    help="Left-truncate prefix to keep last N tokens; matches "
                         "v2 Code2LoRA trainers' --max-seq-len (minus target budget).")
    ap.add_argument("--prefix-max-tokens", type=int, default=0,
                    help="If >0, left-truncate each QnA's prefix to its last N "
                         "tokens BEFORE any RAG query / DRC compression / prompt "
                         "assembly. Used to reproduce the static-snapshot "
                         "prefix budget (median 224 tok) on commit-derived "
                         "suites so RAG / DRC don't have to compete with a "
                         "much longer prefix for the 4K window. 0 = disabled.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repo-limit", type=int, default=0,
                    help="Limit number of repos per suite (debug; 0 = all).")
    ap.add_argument("--qnas-per-commit-limit", type=int, default=8,
                    help="Cap QnAs per (repo, commit). Matches the GRU "
                         "trainer's --max-qna-per-commit eval cap so all "
                         "models score the same number of triples per "
                         "commit. 0 = no cap.")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-i", type=int, default=0,
                    help="0-indexed shard ID; with --num-shards > 1 only "
                         "score repos whose sorted-index mod num_shards == shard_i.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of shards. Use SLURM job arrays to "
                         "fan out across GPUs; results land in suffixed "
                         "JSONs that merge_eval_shards.py recombines.")

    # ---- RAG ------------------------------------------------------------
    ap.add_argument("--rag-cache-dir", type=str, default=None,
                    help="Directory with per-(repo, sha) chunk indices "
                         "(.pt) built by "
                         "evaluation/build_rag_cache_per_commit.py. "
                         "Required for --method rag.")
    ap.add_argument("--rag-top-k", type=int, default=3,
                    help="Number of chunks to prepend per QnA.")
    ap.add_argument("--rag-embed-model-name",
                    default="Qwen/Qwen3-Embedding-0.6B",
                    help="Embedder used to embed the per-QnA query for "
                         "retrieval against the per-commit chunk index.")
    ap.add_argument("--rag-query-chars", type=int, default=2000,
                    help="Length of the prefix tail used as the retrieval "
                         "query (matches baselines/rag/test_rag.py).")

    # ---- DRC ------------------------------------------------------------
    ap.add_argument("--drc-cache-dir", type=str, default=None,
                    help="Directory with per-(repo, sha) DRC context JSONs "
                         "built by "
                         "evaluation/build_drc_cache_per_commit.py. "
                         "Required for --method drc.")
    ap.add_argument("--drc-max-tokens", type=int, default=4096,
                    help="Adaptive-budget cap for the prepended DRC context "
                         "(see evaluation/compress_context.py).")

    # ---- Text2LoRA (Code-SFT v2) ----------------------------------------
    ap.add_argument("--text2lora-hypermod-dir", type=str, default=None,
                    help="Run dir produced by "
                         "baselines/text2lora/train_code_sft.py. Must "
                         "contain hypermod.pt, args.yaml, and "
                         "adapter_config.json. Required for --method "
                         "text2lora.")
    ap.add_argument("--text2lora-code-emb-path", type=str, default=None,
                    help="Path to code_embeddings_v2.pt produced by "
                         "baselines/text2lora/extract_code_embeddings_v2.py. "
                         "Required for --method text2lora.")
    ap.add_argument("--text2lora-dir", type=str, default=None,
                    help="text2lora/ project root (used for chat-template "
                         "path resolution). Defaults to the repo's "
                         "text2lora/ directory.")

    # ---- Doc-to-LoRA (Sakana D2L) --------------------------------------
    ap.add_argument("--doc2lora-ckpt", type=str, default=None,
                    help="Path to a Doc2LoRA checkpoint (pytorch_model.bin), "
                         "a checkpoint-XXXX dir, or a run-dir containing "
                         "checkpoint-* sub-dirs. Required for --method "
                         "doc2lora.")
    ap.add_argument("--doc2lora-drc-cache-dir", type=str, default=None,
                    help="Per-(repo, commit) DRC context cache (the same "
                         "dir the drc baseline uses). Required for "
                         "--method doc2lora.")
    ap.add_argument("--doc2lora-max-ctx-tokens", type=int, default=4096,
                    help="Truncate the per-commit DRC document to this "
                         "many tokens before internalization (matches "
                         "max_packed_ctx_len at training time).")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model_for_method(
        args.method, args.base_model, ckpt, device=device,
        text2lora_hypermod_dir=(Path(args.text2lora_hypermod_dir)
                                if args.text2lora_hypermod_dir else None),
        text2lora_code_emb_path=(Path(args.text2lora_code_emb_path)
                                 if args.text2lora_code_emb_path else None),
        text2lora_dir=(Path(args.text2lora_dir) if args.text2lora_dir else None),
        doc2lora_ckpt=(Path(args.doc2lora_ckpt)
                       if args.doc2lora_ckpt else None),
        doc2lora_drc_cache_dir=(Path(args.doc2lora_drc_cache_dir)
                                if args.doc2lora_drc_cache_dir else None),
        doc2lora_max_ctx_tokens=int(args.doc2lora_max_ctx_tokens),
    )

    # Optional: load the retrieval embedder once (used for --method rag).
    rag_embed_model = None
    rag_embed_tokenizer = None
    rag_cache_dir = None
    drc_cache_dir = None
    if args.method == "rag":
        if not args.rag_cache_dir:
            raise SystemExit("--rag-cache-dir is required for --method rag")
        rag_cache_dir = Path(args.rag_cache_dir).expanduser().resolve()
        if not rag_cache_dir.exists():
            raise SystemExit(f"RAG cache dir not found: {rag_cache_dir}")
        from transformers import AutoModel, AutoTokenizer
        print(f"[load] RAG query-embedder: {args.rag_embed_model_name}",
              flush=True)
        rag_embed_tokenizer = AutoTokenizer.from_pretrained(
            args.rag_embed_model_name, use_fast=True,
        )
        rag_embed_model = AutoModel.from_pretrained(
            args.rag_embed_model_name, torch_dtype=torch.bfloat16,
        ).to(device).eval()
    elif args.method == "drc":
        if not args.drc_cache_dir:
            raise SystemExit("--drc-cache-dir is required for --method drc")
        drc_cache_dir = Path(args.drc_cache_dir).expanduser().resolve()
        if not drc_cache_dir.exists():
            raise SystemExit(f"DRC cache dir not found: {drc_cache_dir}")

    shard_suffix = (f"_shard{args.shard_i}of{args.num_shards}"
                    if args.num_shards > 1 else "")
    suite_summaries: Dict[str, Any] = {}
    for suite in args.suites:
        qna_path = Path(args.qna_dir) / f"{suite}.parquet"
        if not qna_path.exists():
            print(f"[suite {suite}] MISSING parquet {qna_path}, skipping",
                  flush=True)
            continue
        out_path = out_dir / f"baseline_{args.method}_{suite}{shard_suffix}.json"
        s = score_suite(
            qna_path=qna_path, suite_name=suite,
            model=model, tokenizer=tok, device=device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            repo_limit=args.repo_limit,
            qnas_per_commit_limit=args.qnas_per_commit_limit,
            out_path=out_path,
            bootstrap=args.bootstrap,
            shard_i=args.shard_i,
            num_shards=args.num_shards,
            prefix_max_tokens=args.prefix_max_tokens,
            # Context injection
            context_method=args.method if args.method in ("rag", "drc") else "none",
            rag_cache_dir=rag_cache_dir, rag_top_k=args.rag_top_k,
            rag_embed_model=rag_embed_model,
            rag_embed_tokenizer=rag_embed_tokenizer,
            rag_query_chars=args.rag_query_chars,
            drc_cache_dir=drc_cache_dir, drc_max_tokens=args.drc_max_tokens,
        )
        suite_summaries[suite] = s

    # Final cross-suite summary file (per shard; merge later).
    summary_path = out_dir / f"baseline_{args.method}_summary{shard_suffix}.json"
    summary_path.write_text(json.dumps({
        "method": args.method,
        "base_model": args.base_model,
        "ckpt": str(ckpt) if ckpt else None,
        "qna_dir": str(args.qna_dir),
        "args": vars(args),
        "suites": suite_summaries,
    }, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
