#!/usr/bin/env python3
"""Summarize HY3 masked-MTP K=1..8 stage sweep logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


GENERATE = re.compile(
    r"^\[generate\]\s+\d+ tok in ([0-9.]+)s = ([0-9.]+) tok/s .* prime ([0-9.]+)s"
)
SPEC = re.compile(
    r"^\[generate_spec K=(\d+)\]\s+\d+ tok in ([0-9.]+)s = ([0-9.]+) tok/s"
)
ACCEPT = re.compile(
    r"^\s+acceptance: (\d+)/(\d+) = ([0-9.]+)%\s+self-consistency: (PASS|FAIL)"
)
NAME = re.compile(r"stage-(?P<label>.+)-q8(?P<q8>[01])\.log$")


def parse_log(path: Path) -> dict:
    match = NAME.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized stage filename: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    plain = None
    token_line = None
    ks: dict[int, dict] = {}
    pending_k: int | None = None
    for line in lines:
        if found := GENERATE.search(line):
            plain = {
                "elapsed_s": float(found.group(1)),
                "tok_s": float(found.group(2)),
                "prime_s": float(found.group(3)),
            }
        elif line.startswith("  tokens:"):
            token_line = line
        elif found := SPEC.search(line):
            pending_k = int(found.group(1))
            ks[pending_k] = {
                "elapsed_s": float(found.group(2)),
                "tok_s": float(found.group(3)),
            }
        elif pending_k is not None and (found := ACCEPT.search(line)):
            ks[pending_k].update(
                {
                    "accepted": int(found.group(1)),
                    "drafted": int(found.group(2)),
                    "acceptance_pct": float(found.group(3)),
                    "pass": found.group(4) == "PASS",
                }
            )
            pending_k = None
    if plain is None or token_line is None:
        raise ValueError(f"{path}: missing plain generation or token tape")
    if set(ks) != set(range(1, 9)) or not all(row.get("pass") for row in ks.values()):
        raise ValueError(f"{path}: incomplete or failing K=1..8 sweep")
    return {
        "label": match.group("label"),
        "q8": match.group("q8") == "1",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "token_sha256": hashlib.sha256(token_line.encode()).hexdigest(),
        "masked_head_nvfp4": "re-quantized BF16 -> NVFP4" in text,
        "plain": plain,
        "ks": {str(k): ks[k] for k in sorted(ks)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [parse_log(path) for path in args.logs]
    report = {"format": "memra-hy3-masked-stage-sweeps-v1", "rows": rows}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
