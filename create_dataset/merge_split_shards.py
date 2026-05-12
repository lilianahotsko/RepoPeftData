#!/usr/bin/env python3
"""Merge sharded suite JSONs into a single <suite>.json.

Usage::

    python merge_split_shards.py --splits-dir <dir> --suites ir_test cr_val

For each suite, reads ``<suite>.shard0.json`` ... ``<suite>.shardN.json`` and
writes ``<suite>.json`` with the union of ``repositories`` entries. Shard files
are kept on disk for safety; delete manually after the merged file is in use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--suites", nargs="+", required=True)
    args = ap.parse_args()

    sd = Path(args.splits_dir)
    for suite in args.suites:
        shards = sorted(sd.glob(f"{suite}.shard*.json"))
        if not shards:
            print(f"  [{suite}] no shards found; skip")
            continue
        merged: dict = {}
        for sh in shards:
            d = json.loads(sh.read_text(encoding="utf-8"))
            repos = d.get("repositories") or {}
            for k, v in repos.items():
                if k in merged:
                    print(f"  [{suite}] dup key {k!r} in {sh.name}; keeping first")
                    continue
                merged[k] = v
            print(f"  [{suite}] {sh.name}: {len(repos):,} entries")
        out = sd / f"{suite}.json"
        out.write_text(json.dumps({"repositories": merged}), encoding="utf-8")
        print(f"  [{suite}] -> {out} ({len(merged):,} entries)")


if __name__ == "__main__":
    main()
