"""Step 4 (prefill k3 = A vs kpc = B, four cards): per arm and cell, ttft median/min/max, argmax set, overlap."""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

for d in sys.argv[1:]:
    d = Path(d)
    ttft = defaultdict(list)
    arg = defaultdict(set)
    for cj in sorted(d.glob("run*-*/cells.jsonl")):
        arm = cj.parent.name.split("-")[1]
        for line in cj.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            ttft[(c["cell"], arm)].append(c["ttft_s"] * 1000)
            arg[(c["cell"], arm)].add(c["argmax"])
    print(f"== {d.name}")
    for cell in sorted({k[0] for k in ttft}, key=lambda s: int(s.split("-")[1])):
        a, b = ttft[(cell, "A")], ttft[(cell, "B")]
        overlap = not (max(b) < min(a) or max(a) < min(b))
        print(f"{cell}: A k3 n {len(a)} median {st.median(a):.1f} [{min(a):.1f}, {max(a):.1f}] | "
              f"B kpc n {len(b)} median {st.median(b):.1f} [{min(b):.1f}, {max(b):.1f}] | B/A median {st.median(b)/st.median(a):.3f} "
              f"| ranges overlap {overlap} | argmax A {sorted(arg[(cell,'A')])} B {sorted(arg[(cell,'B')])} "
              f"equal {arg[(cell,'A')] == arg[(cell,'B')] and len(arg[(cell,'A')]) == 1}")
