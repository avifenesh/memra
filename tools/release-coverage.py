#!/usr/bin/env python3
"""Validate release-battery coverage, independently of success-marker substrings."""

import argparse
from pathlib import Path
import re
import sys


# Same measured ceiling as local-ci.sh. A caller's CI override cannot widen a release.
KERNEL_SKIP_BUDGET = 11


def spec_coverage(lines):
    seen = set()
    passed = set()
    current = None
    summaries = 0
    for line in lines:
        if line.startswith("[generate_spec K="):
            match = re.match(r"^\[generate_spec K=([1-8])\] ", line)
            if not match or int(match[1]) in seen or summaries:
                raise ValueError("missing, duplicate, or out-of-range speculative K")
            current = int(match[1])
            seen.add(current)
        if "self-consistency:" in line:
            if (current is None or current in passed
                    or not line.endswith("self-consistency: PASS (identical to plain target)")):
                raise ValueError(f"K={current}: missing, duplicate, failed, or non-greedy verdict")
            passed.add(current)
        if "=== SELF-CONSISTENCY" in line:
            if line != "=== SELF-CONSISTENCY PASS ===" or passed != set(range(1, 9)):
                raise ValueError("speculative summary failed or malformed")
            summaries += 1
    if seen != set(range(1, 9)) or passed != seen or summaries != 1:
        raise ValueError(f"expected K=1..8 once each; seen={sorted(seen)}, passed={sorted(passed)}, summaries={summaries}")
    return "K=1..8 self-consistency, identical to plain target (8/8)"


def required_cells(paths):
    required = set()
    for path in paths:
        for raw in path.read_text().splitlines():
            name = raw.split("#", 1)[0].strip()
            if not name:
                continue
            if len(name.split()) != 1:
                raise ValueError(f"malformed required-cell manifest: {path}")
            required.add(name)
    if not required:
        raise ValueError("no required kernel cells")
    return required


def kernel_coverage(lines, required):
    summaries = [line for line in lines if line.startswith("ALL GREEN")]
    if len(summaries) != 1 or not lines or lines[-1] != summaries[0]:
        raise ValueError("expected one final ALL GREEN cell/skip summary")
    summary = re.fullmatch(r"ALL GREEN \(([0-9]+) cells, ([0-9]+) skipped\)", summaries[0])
    if not summary:
        raise ValueError("malformed kernel cell/skip summary")
    total, skipped = map(int, summary.groups())
    ran = set()
    skips = {}
    for line in lines:
        if line.startswith("SKIP "):
            match = re.fullmatch(r"SKIP (\S+) \((.+)\)", line)
            if not match or match[1] in skips:
                raise ValueError("malformed or duplicate named kernel skip")
            if "MEMRA_KC_FAST" in match[2] or "MEMRA_KC_ONLY" in match[2]:
                raise ValueError(f"kernel coverage narrowed: {line}")
            skips[match[1]] = line
        elif line.startswith("MISSING REQUIRED CELL ") or line.endswith(" FAIL"):
            raise ValueError(f"kernel cell failed: {line}")
        # Match kernel_check.rs::output_cell_name for successful required cells.
        elif line.endswith((" OK", " HIGH", " IMPROVED", " WORSE")) or " OK (byte-identical)" in line:
            name = re.split(r"[\[(]", line.split()[0].rstrip(":"), maxsplit=1)[0]
            ran.add(name)
    # CellTracker removes an earlier skip if the same cell subsequently executes.
    effective_skips = set(skips) - ran
    missing = required - ran
    if missing:
        raise ValueError(f"required kernel cells not executed: {', '.join(sorted(missing))}")
    if skipped != len(effective_skips):
        raise ValueError(f"kernel skip count mismatch: summary={skipped}, named={len(effective_skips)}")
    if skipped > KERNEL_SKIP_BUDGET:
        raise ValueError(f"kernel skip budget exceeded: {skipped} > {KERNEL_SKIP_BUDGET}")
    # Some sections explicitly record cells in addition to output-derived names. Do not
    # invent exact total equality from text; require at least every observed/required cell.
    executed = total - skipped
    if executed < len(ran) or executed < len(required):
        raise ValueError(f"invalid kernel executed count: {executed}")
    return "\n".join([
        f"ALL GREEN ({total} cells, {executed} executed, {len(required)} required, {skipped} skipped; budget {KERNEL_SKIP_BUDGET})",
        *(skips[name] for name in sorted(effective_skips)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("spec", "kernel"))
    parser.add_argument("--require-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    lines = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    try:
        result = (spec_coverage(lines) if args.kind == "spec"
                  else kernel_coverage(lines, required_cells(args.require_manifest)))
    except (ValueError, OSError) as error:
        print(f"coverage refused: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
