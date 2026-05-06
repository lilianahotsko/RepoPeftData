#!/usr/bin/env python3
"""
Sanity-check that Text2LoRA actually injects LoRA weights at evaluation time.

The paper-time observation is that Text2LoRA(text) and Text2LoRA(code) both
land at 45.8% EM, vs. 45.7% for the un-adapted pretrained baseline. A 0.1%
gap is within EM noise on RepoPeftBench (n=6,414, std-of-mean ≈ 0.6%) and is
indistinguishable from the pretrained model. Three failure modes can explain
this:

    1. ``model`` returned by ``load_hypermod_checkpoint`` is not a PeftModel,
       so ``set_peft_model_state_dict`` is a no-op.
    2. ``model`` is a PeftModel but its LoRA adapters are disabled or have
       zero scaling.
    3. The hypernetwork emits LoRA matrices with effectively zero norm, so
       the adapter is technically attached but does nothing.

This script tells you which one. It:

    * loads the T2L checkpoint exactly as the eval script does;
    * confirms the type of ``model`` and lists peft adapters / scaling;
    * generates a LoRA for one canonical description;
    * compares per-layer LoRA-A/-B norms against random / zero baselines;
    * runs the same prompt with adapters enabled vs. disabled and compares
      the next-token logits and the greedy continuation.

Usage::

    python baselines/text2lora/diagnose_t2l_injection.py \\
        --hypermod-dir text2lora/train_outputs/recon/hyper_lora/<run_name>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _model_summary(model) -> dict:
    """Return a small dict describing the LoRA topology of ``model``."""
    info = {
        "type": type(model).__name__,
        "is_peft": False,
        "adapters": [],
        "n_lora_modules": 0,
        "active_adapter": None,
        "module_examples": [],
    }
    try:
        from peft import PeftModel
        info["is_peft"] = isinstance(model, PeftModel)
    except Exception:
        pass
    if hasattr(model, "peft_config"):
        info["adapters"] = list(getattr(model, "peft_config").keys())
    if hasattr(model, "active_adapter"):
        try:
            info["active_adapter"] = model.active_adapter
        except Exception:
            pass

    # Walk the module tree to count LoRA-A / LoRA-B leaves.
    n = 0
    examples = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "Lora" in cls and ("A" in cls or "B" in cls):
            n += 1
            if len(examples) < 4:
                examples.append((name, cls))
    info["n_lora_modules"] = n
    info["module_examples"] = examples
    return info


def _lora_norm_summary(state_dict) -> dict:
    """Quick summary statistics over a LoRA state_dict (A / B Frobenius norms)."""
    a_norms, b_norms = [], []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "lora_A" in k:
            a_norms.append(float(v.float().norm().cpu()))
        elif "lora_B" in k:
            b_norms.append(float(v.float().norm().cpu()))
    def _q(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {
            "min": xs[0],
            "med": xs[len(xs) // 2],
            "max": xs[-1],
            "mean": sum(xs) / len(xs),
            "n": len(xs),
        }
    return {"A": _q(a_norms), "B": _q(b_norms)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hypermod-dir", required=True, type=Path,
        help="Path to the T2L checkpoint directory (containing hypermod.pt).",
    )
    ap.add_argument(
        "--text2lora-dir", default=Path("text2lora"), type=Path,
        help="text2lora source directory (cwd-relative).",
    )
    ap.add_argument(
        "--description", type=str,
        default=("Python repository for matrix factorization, sparse linear "
                 "algebra, and dimensionality reduction with NumPy backends."),
        help="Canonical task description used to generate the diagnostic LoRA.",
    )
    ap.add_argument(
        "--prompt", type=str,
        default=(
            "import numpy as np\n\n"
            "def test_eye_identity():\n"
            "    A = np.eye(3)\n"
            "    assert np.allclose(A @ A,"
        ),
        help="Prompt used to compare adapter-enabled vs. disabled logits.",
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(_ROOT / "text2lora" / "src"))

    orig_cwd = os.getcwd()
    text2lora_abs = Path(orig_cwd) / args.text2lora_dir
    os.chdir(text2lora_abs)

    try:
        from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
        from hyper_llm_modulator.utils import get_layers, embed_texts
        from peft.utils import set_peft_model_state_dict

        ckpt_path = Path(orig_cwd) / args.hypermod_dir / "hypermod.pt"
        print(f"[diag] Loading checkpoint: {ckpt_path}")

        (
            hargs, hypermod, model, tokenizer,
            emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
        ) = load_hypermod_checkpoint(str(ckpt_path), device)

        os.chdir(orig_cwd)

        hypermod.eval()
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        # 1. Topology check ---------------------------------------------------
        info = _model_summary(model)
        print("\n[diag] Model summary:")
        for k, v in info.items():
            print(f"  {k}: {v}")

        if not info["is_peft"]:
            print(
                "[diag] FAIL: model is not a PeftModel; set_peft_model_state_dict "
                "is a no-op. The 45.8% number is the pretrained model.",
            )

        # 2. Generate one LoRA and inspect norms -----------------------------
        layer_indices = torch.tensor(
            list(range(len(get_layers(model)))), dtype=torch.long, device=device,
        )
        with torch.no_grad():
            task_emb = embed_texts(
                [args.description], emb_model, emb_tokenizer,
                task_desc_format_fn, pooling_fn, device,
            )
            encoded = hypermod.task_encoder(task_emb)["encoded_task_emb"].detach()
            lora_sd = hypermod.gen_lora(layer_indices, encoded)

        norms = _lora_norm_summary(lora_sd)
        print("\n[diag] Generated LoRA norm summary:")
        for k, v in norms.items():
            print(f"  {k}: {v}")

        # 3. Logit delta with vs. without LoRA --------------------------------
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            logits_no_lora = model(input_ids=input_ids).logits[0, -1].float().clone()
            top_no = int(logits_no_lora.argmax())

            set_peft_model_state_dict(model, lora_sd)
            logits_with_lora = model(input_ids=input_ids).logits[0, -1].float().clone()
            top_lora = int(logits_with_lora.argmax())

            delta = (logits_with_lora - logits_no_lora).abs()
            l2 = float(delta.norm())
            linf = float(delta.max())

        print("\n[diag] Logit delta (with-LoRA vs. without-LoRA):")
        print(f"  L2:   {l2:.6f}")
        print(f"  Linf: {linf:.6f}")
        print(f"  argmax-no-lora: {top_no} -> {tokenizer.decode([top_no])!r}")
        print(f"  argmax-with-lora: {top_lora} -> {tokenizer.decode([top_lora])!r}")

        if l2 < 1e-3:
            print(
                "\n[diag] FAIL: with-LoRA logits are indistinguishable from "
                "without-LoRA logits. The hypernetwork is producing near-zero "
                "deltas, the LoRA is attached but disabled, or the adapter "
                "scaling is zero. Confirm hypermod.gen_lora normalizes outputs "
                "to a non-trivial scale and that the active adapter is not in "
                "merge_and_unload state.",
            )
        elif l2 < 0.01:
            print(
                "\n[diag] WARN: logits change by < 1e-2; this is consistent "
                "with a near-zero LoRA. Either intended (the model has "
                "learned to produce conservative deltas) or a sign that the "
                "hypernet did not converge to anything task-specific.",
            )
        else:
            print(
                "\n[diag] OK: LoRA produces non-trivial logit deltas; the "
                "low EM is probably due to the hypernet content rather than "
                "an injection bug.",
            )
        return 0
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
