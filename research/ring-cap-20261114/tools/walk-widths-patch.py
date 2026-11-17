#!/usr/bin/env python3
"""memra#1299 receipts: make the chunked prime walk print every call's width
(`[prime-walk] t=.. base=.. calls=[...]`), so the receipt shows the schedule a continuation
took. Diagnostic only, applied to both arms of the check (main and the PR head): one eprintln
after the schedule is computed, nothing else.
usage: walk-widths-patch.py HYBRID_FORWARD_RS"""
import sys

p = sys.argv[1]
s = open(p).read()
anchor = "        };\n        // CHUNK-ORDER INVARIANCE (lane/chunk-invariance, 2026-08-05; vLLM #38561 shape).\n"
assert s.count(anchor) == 1, s.count(anchor)
line = (
    "        };\n"
    "        eprintln!(\n"
    "            \"[prime-walk] t={t} base={} calls={:?}\",\n"
    "            cache.pos,\n"
    "            ranges.iter().map(|&(a, b)| b - a).collect::<Vec<usize>>()\n"
    "        );\n"
)
s = s.replace(anchor, line + anchor[len("        };\n"):])
open(p, "w").write(s)
print("patched", p)
