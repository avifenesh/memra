#!/usr/bin/env python3
"""Add the physical-card identity to every scored JSONL row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.out.open("x", encoding="utf-8") as out:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{args.input}:{line_number}: JSON row is not an object")
            row.update(
                {
                    "physical_gpu_index": args.physical_gpu,
                    "physical_gpu_uuid": args.gpu_uuid,
                    "gpu_lock": args.lock,
                }
            )
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
