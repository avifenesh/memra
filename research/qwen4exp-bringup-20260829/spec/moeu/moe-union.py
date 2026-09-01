#!/usr/bin/env python3
"""Routed-UNION size per verify chunk, from a shared-format MEMRA_MOE_TRACE file.

The one number the `moeu` lever lives or dies on. At a K=5 verify chunk the qwen4_exp MoE
dispatches t*selected = 60 (token, expert) slots and reads each slot's expert bytes
independently. A union gather would read each DISTINCT expert once per chunk, so the
lever's whole prize is the ratio

    union / pairs   ==   distinct experts in the chunk / t*selected

and 1 - that ratio is the fraction of the section's weight traffic it could remove.

WHY THIS IS A DIFFERENT SCRIPT FROM `moe-hit-rate.py`, which sits beside it: that one
answers "how much of a token's top-k did the PREVIOUS token pick", and it therefore reads
ONLY t == 1 decode lines and explicitly skips every t > 1 line. This one reads ONLY the
t > 1 lines, because a verify chunk IS a t > 1 forward and its union is a within-line
property. Same trace, disjoint halves, two different questions — do not merge them.

Trace format (frozen `memra-ep-map-v1` producer side, one appended line per
(layer, forward)):

    <layer> <t> <e0,e1,...>

with t rows' selections CONCATENATED, TOKEN-MAJOR, t*selected ids on the line. Token-major
is what lets this script recover per-column groups by slicing in chunks of `selected`, and
it is why `--selected` must match the run (default 10 = the qwen4_exp router's top-k). The
script REFUSES a line whose id count is not t*selected rather than guessing the width: a
mis-set --selected would silently report a wrong union.

Reported per t and per layer: mean/min/max union, the union/pairs ratio, and the
incremental-overlap profile (how many of column j's experts were already named by columns
< j) so a union gather's payoff can be attributed to position inside the chunk.

Usage:  moe-union.py <trace.txt> [--selected K] [--t T] [--per-layer]
"""
import sys
from collections import defaultdict


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    path = argv[0]
    selected = 10
    only_t = None
    per_layer = False
    rest = argv[1:]
    for i, a in enumerate(rest):
        if a == "--selected":
            selected = int(rest[i + 1])
        elif a == "--t":
            only_t = int(rest[i + 1])
        elif a == "--per-layer":
            per_layer = True

    # (t) -> list of union sizes ; (t, layer) -> list ; (t, col) -> list of new-expert counts
    by_t = defaultdict(list)
    by_tl = defaultdict(list)
    fresh = defaultdict(list)
    decode_lines = 0
    refused = 0
    malformed = 0

    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            try:
                layer = int(parts[0])
                t = int(parts[1])
                ids = [int(x) for x in parts[2].split(",") if x != ""]
            except ValueError:
                malformed += 1
                continue
            if t == 1:
                # A t == 1 line is a plain decode step: it has no within-forward union to
                # measure (10 distinct picks, 10 slots, ratio 1 by construction). Counted,
                # not folded in — that is `moe-hit-rate.py`'s half of this trace.
                decode_lines += 1
                continue
            if only_t is not None and t != only_t:
                continue
            if len(ids) != t * selected:
                # Never guess the column width. A wrong --selected turns every union into a
                # plausible-looking wrong number, which is the failure class this lane is
                # under orders to avoid.
                refused += 1
                continue
            cols = [ids[c * selected:(c + 1) * selected] for c in range(t)]
            union = len(set(ids))
            by_t[t].append(union)
            by_tl[(t, layer)].append(union)
            seen = set()
            for c, col in enumerate(cols):
                new = len([x for x in col if x not in seen])
                fresh[(t, c)].append(new)
                seen.update(col)

    if not by_t:
        print(
            f"NO t>1 LINES usable in {path} "
            f"(t==1 decode lines={decode_lines}, width-refused={refused}, "
            f"malformed={malformed}).\n"
            "A verify chunk emits t>1 lines only when the trace tap actually fires: under "
            "the single-card device route arm MEMRA_Q4E_ROUTER_AUDIT=1 together with "
            "MEMRA_MOE_TRACE, and run a SPEC arm (--mtp --spec-k K) so chunks exist at all. "
            "If width-refused is large, --selected does not match the run's top-k.",
            file=sys.stderr,
        )
        return 1

    print(f"# trace {path}")
    print(f"# selected={selected} t1_decode_lines={decode_lines} "
          f"width_refused={refused} malformed={malformed}")
    print("t\tpairs\tchunks\tunion_mean\tunion_min\tunion_max\tratio_mean\ttraffic_saved")
    for t in sorted(by_t):
        u = by_t[t]
        pairs = t * selected
        mean = sum(u) / len(u)
        print(f"{t}\t{pairs}\t{len(u)}\t{mean:.2f}\t{min(u)}\t{max(u)}\t"
              f"{mean / pairs:.4f}\t{1 - mean / pairs:.4f}")

    print("\n# fresh experts per column (column 0 is always `selected` by construction)")
    print("t\tcolumn\tfresh_mean\tfresh_min\tfresh_max")
    for (t, c) in sorted(fresh):
        v = fresh[(t, c)]
        print(f"{t}\t{c}\t{sum(v) / len(v):.2f}\t{min(v)}\t{max(v)}")

    if per_layer:
        print("\n# per-layer union")
        print("t\tlayer\tchunks\tunion_mean\tratio_mean")
        for (t, layer) in sorted(by_tl):
            v = by_tl[(t, layer)]
            m = sum(v) / len(v)
            print(f"{t}\t{layer}\t{len(v)}\t{m:.2f}\t{m / (t * selected):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
