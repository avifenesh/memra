#!/usr/bin/env python3
"""Summarize `nvcc -Xptxas -v` output: entry, registers, stack, spill stores/loads.

usage: ptxas_regs.py BUILD_LOG [NAME_REGEX]
"""
import re
import sys

log = open(sys.argv[1]).read().splitlines()
pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"^(cand_|fa_prefill_qw_h2p$|fa_prefill_qw_h2$)")
cur = None
rows = {}
for line in log:
    m = re.search(r"Compiling entry function '([^']+)'", line)
    if m:
        cur = m.group(1)
        rows.setdefault(cur, {})
        continue
    if cur is None:
        continue
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
    if m:
        rows[cur].update(stack=int(m.group(1)), st=int(m.group(2)), ld=int(m.group(3)))
    m = re.search(r"Used (\d+) registers", line)
    if m:
        rows[cur]["regs"] = int(m.group(1))
print(f"{'entry':28s} regs stack spill-st spill-ld")
for k, v in rows.items():
    if pat.search(k):
        print(f"{k:28s} {v.get('regs', '?'):>4} {v.get('stack', '?'):>5} {v.get('st', '?'):>8} {v.get('ld', '?'):>8}")
