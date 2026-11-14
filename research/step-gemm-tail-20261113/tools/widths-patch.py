#!/usr/bin/env python3
"""memra#1275 receipts: make the step35 GEMM prime's ENGAGED line list every call's width
(`calls=[...]`), so the 8,200 tail shows. Diagnostic only, applied to both arms of the check
(main and the PR head); it adds a Vec and a format argument, nothing else.
usage: widths-patch.py HYBRID_FORWARD_RS"""
import re
import sys

p = sys.argv[1]
s = open(p).read()
loop = re.search(r"\n(\s*)for \(chunk_idx, \(start, end\)\) in step_gemm_prime_ranges\(", s)
assert loop, "GEMM loop not found"
ind = loop.group(1)
s = s[:loop.start()] + f"\n{ind}let mut call_widths: Vec<usize> = Vec::new();" + s[loop.start():]
# The first push goes right after the loop header's opening brace (one line on main, four on
# the PR head).
brace = s.index("{\n", loop.end())
s = s[:brace + 2] + f"{ind}    call_widths.push(end - start);\n" + s[brace + 2:]
old = '"[gemm-prime] ENGAGED t={t} base={base} seq_end={seq_end} chunks<={width} (GEMM trunk + grouped MoE)"'
assert s.count(old) == 1
s = s.replace(old, '"[gemm-prime] ENGAGED t={t} base={base} seq_end={seq_end} chunks<={width} calls={call_widths:?} (GEMM trunk + grouped MoE)"')
open(p, "w").write(s)
print("patched", p)
