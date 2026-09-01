#!/usr/bin/env python3
"""Per-layer expert HIT RATE and co-activation from a shared-format MEMRA_MOE_TRACE file.

Two questions, one pass, no GPU:

  1. **Hit rate** -- how much of a token's top-k the PREVIOUS token already selected, per layer.
     This is the number the expert-speculation lever is priced against: a speculative stage keyed
     off token t-1's selection can only hide latency for the experts it guessed right.
  2. **Co-activation** -- which expert pairs land in the same token's selection, per layer. This is
     the input the owner's co-activation placement doctrine wants, and it is why this reads the
     FROZEN shared format rather than a lane-local one: the same file feeds
     tools/build_expert_placement_map.py, so a hit-rate reading and a placement map are always
     derived from the same trace rather than from two runs that might differ.

Trace format (frozen `memra-ep-map-v1` producer side, one appended line per (layer, forward)):

    <layer> <t> <e0,e1,...>

with `t` rows' selections CONCATENATED on one line -- co-occurrence is "within-line" for t > 1.
That matters here: **only t == 1 lines carry a well-defined "previous token"**, because a prefill
line's ids are many tokens' picks with no per-token delimiter. So the hit rate is computed over
DECODE lines only, and the count of skipped prefill lines is reported rather than hidden. A hit
rate silently averaged over prefill lines would be measuring "did two arbitrary tokens inside one
chunk agree", which is a different question wearing the same name.

Usage:  moe-hit-rate.py <trace.txt> [--top-pairs N] [--selected K]
"""
import sys
from collections import defaultdict


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    path = argv[0]
    top_pairs = 12
    selected = None
    for i, a in enumerate(argv[1:]):
        if a == "--top-pairs":
            top_pairs = int(argv[i + 2])
        elif a == "--selected":
            selected = int(argv[i + 2])

    prev = {}                                   # layer -> previous decode selection (as a set)
    hits = defaultdict(list)                    # layer -> [overlap fraction, ...]
    pair = defaultdict(lambda: defaultdict(int))  # layer -> (a,b) -> count
    seen = defaultdict(int)                     # layer -> decode lines
    widths = defaultdict(set)                   # layer -> observed selection widths
    skipped_prefill = 0
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
            if t != 1:
                # A prefill line is t tokens concatenated with no delimiter: it has no
                # "previous token", and its co-activation is across tokens rather than within
                # one. Counted, not silently folded in.
                skipped_prefill += 1
                continue
            cur = set(ids)
            widths[layer].add(len(ids))
            seen[layer] += 1
            if layer in prev:
                k = selected or len(ids) or 1
                hits[layer].append(len(cur & prev[layer]) / k)
            prev[layer] = cur
            s = sorted(cur)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    pair[layer][(s[i], s[j])] += 1

    if not seen:
        # Loud, not a zero row: an empty read here has always meant the tap did not fire, and
        # this lane spent a while believing an empty traces directory meant "not run yet".
        print(
            f"NO DECODE LINES in {path} "
            f"(prefill lines skipped={skipped_prefill}, malformed={malformed}).\n"
            "Under the single-card device route the tracer rides the ROUTER_AUDIT readback: "
            "arm MEMRA_Q4E_ROUTER_AUDIT=1 together with MEMRA_MOE_TRACE, on a binary that "
            "contains the trace-tap wiring.",
            file=sys.stderr,
        )
        return 1

    print(f"# trace {path}")
    print(f"# decode lines={sum(seen.values())} layers={len(seen)} "
          f"prefill_lines_skipped={skipped_prefill} malformed={malformed}")
    print("layer\tdecode_steps\tsel_width\thit_rate_mean\thit_rate_min\thit_rate_max")
    allh = []
    for layer in sorted(seen):
        h = hits.get(layer, [])
        allh.extend(h)
        w = ",".join(str(x) for x in sorted(widths[layer]))
        if h:
            print(f"{layer}\t{seen[layer]}\t{w}\t{sum(h)/len(h):.4f}\t{min(h):.4f}\t{max(h):.4f}")
        else:
            print(f"{layer}\t{seen[layer]}\t{w}\t-\t-\t-")
    if allh:
        print(f"# FLEET hit_rate mean={sum(allh)/len(allh):.4f} over {len(allh)} transitions")

    print("\n# top co-activated pairs per layer (placement-lane input)")
    print("layer\texpert_a\texpert_b\tcount\tshare_of_steps")
    for layer in sorted(pair):
        top = sorted(pair[layer].items(), key=lambda kv: -kv[1])[:top_pairs]
        for (a, b), c in top:
            print(f"{layer}\t{a}\t{b}\t{c}\t{c/max(seen[layer],1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
