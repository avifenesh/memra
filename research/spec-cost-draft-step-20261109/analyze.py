#!/usr/bin/env python3
"""Tables for the cost-model checks (memra#1093): what K the model picked, and what it bought.

Reads a run directory written by ab_bins.py (runs.jsonl, server-N-ARM.log). Per arm: the K of
every cold-long request (the first request of a boot explores at the rung before the model has
acceptance history, so it is counted apart), the decode rate per class, and each arm's per-boot
median against the base arm's with the U count (how many of the boot pairs the arm wins).

usage: analyze.py RUN_DIR [REFERENCE_ARM]
"""
import collections
import json
import pathlib
import statistics
import sys

run = pathlib.Path(sys.argv[1])
ref_arm = sys.argv[2] if len(sys.argv) > 2 else "base"
rows = [json.loads(line) for line in open(run / "runs.jsonl")]
arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != ref_arm, a))
boots = sorted({r["boot"] for r in rows})
arm_of = {r["boot"]: r["arm"] for r in rows}
first = {b: min(i for i, r in enumerate(rows) if r["boot"] == b) for b in boots}


def long(r):
    return (r["prompt_tokens"] or 0) >= 1024


print("## K picked on cold-long requests\n")
print("| arm | boots | first request of a boot (explores) | later requests: K=2 | K=3 | other |")
print("|---|---:|---|---:|---:|---:|")
for arm in arms:
    explore = collections.Counter()
    later = collections.Counter()
    for i, r in enumerate(rows):
        if r["arm"] != arm or not long(r):
            continue
        k = f"{r['k'][0]}/{r['k'][1]}" if r["k"] else "none"
        (explore if i == first[r["boot"]] else later)[k] += 1
    n2 = sum(v for k, v in later.items() if k.startswith("2/"))
    n3 = sum(v for k, v in later.items() if k.startswith("3/"))
    other = sum(later.values()) - n2 - n3
    exp = ", ".join(f"K={k} x{v}" for k, v in sorted(explore.items()))
    print(f"| {arm} | {sum(1 for b in boots if arm_of[b] == arm)} | {exp} | {n2} | {n3} | {other} |")

classes = collections.OrderedDict()
for r in rows:
    tokens = r["prompt_tokens"] or 0
    size = "short" if tokens < 1024 else "1.2k" if tokens < 2000 else "3.6k" if tokens < 4500 else "5.8k"
    classes.setdefault((r["class"].split("-")[0], size), []).append(r)


def u(a, b):
    return sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)


print("\n## Decode rate\n")
print("| class | arm | n | TTFT p50 (s) | decode median, range (tok/s) | acceptance | per-boot ratio to "
      f"{ref_arm} | U |")
print("|---|---|---:|---:|---|---:|---:|---:|")
for (kind, size), rs in classes.items():
    per_boot = {}
    for arm in arms:
        per_boot[arm] = [statistics.median(r["decode_tok_s"] for r in rs if r["boot"] == b)
                         for b in boots if arm_of[b] == arm and any(r["boot"] == b for r in rs)]
    for arm in arms:
        sel = [r for r in rs if r["arm"] == arm]
        if not sel:
            continue
        d = [r["decode_tok_s"] for r in sel]
        ratio = statistics.median(per_boot[arm]) / statistics.median(per_boot[ref_arm])
        pairs = len(per_boot[arm]) * len(per_boot[ref_arm])
        cmp = "" if arm == ref_arm else f"{ratio:.3f} | {u(per_boot[arm], per_boot[ref_arm]):g}/{pairs}"
        if arm == ref_arm:
            cmp = "1 | "
        print(f"| {'T=1' if kind == 't1' else kind} {size} | {arm} | {len(sel)} | "
              f"{statistics.median(r['ttft_s'] for r in sel):.3f} | {statistics.median(d):.2f}, "
              f"{min(d):.2f}-{max(d):.2f} | {statistics.median(r['acceptance'] for r in sel):.3f} | {cmp} |")

greedy = collections.defaultdict(set)
for r in rows:
    if r["class"].startswith("greedy"):
        greedy[r["prompt"]].add(r["sha"])
print("\ngreedy texts:", {p: sorted(v) for p, v in greedy.items()})
print("errors:", sum(1 for r in rows if r.get("error")))
