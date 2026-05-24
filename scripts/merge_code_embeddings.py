#!/usr/bin/env python3
"""Merge per-repo code-embedding dicts (.pt) for Text2LoRA eval.

Later keys override earlier keys on collision.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Merged .pt path.")
    ap.add_argument("inputs", nargs="+", help="Input .pt files (in order).")
    args = ap.parse_args()

    merged: dict = {}
    for path in args.inputs:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        d = torch.load(str(p), map_location="cpu", weights_only=True)
        if not isinstance(d, dict):
            raise SystemExit(f"expected dict in {p}, got {type(d)}")
        merged.update(d)
        print(f"  {p.name}: +{len(d)} keys -> total {len(merged)}")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(merged, tmp)
    os.replace(tmp, out)
    print(f"Wrote {len(merged)} embeddings -> {out}")


if __name__ == "__main__":
    main()
