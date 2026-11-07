#!/usr/bin/env python3
"""Static SASS census per kernel: instruction mix of the hottest loop (the backward branch whose body
holds the most HMMAs) and of the whole function.

usage: cuobjdump -sass BIN > x.sass; sass_loop.py x.sass [NAME_REGEX]
"""
import re
import sys
from collections import Counter

text = open(sys.argv[1]).read().splitlines()
pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"cand_|fa_prefill_qw_h2p$")
funcs = {}
cur = None
for line in text:
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
        cur = m.group(1)
        funcs[cur] = []
        continue
    m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
    if m and cur:
        funcs[cur].append((int(m.group(1), 16), m.group(2).strip()))


def op(ins):
    s = re.sub(r"^@!?U?P\w+\s+", "", ins)
    return s.split()[0].split(".")[0]


for name, ins in funcs.items():
    if not pat.search(name):
        continue
    addr = {a: i for i, (a, _) in enumerate(ins)}
    best = None
    for i, (a, s) in enumerate(ins):
        m = re.search(r"\bBRA\b.*?`?\(?\.?L?_?x?_?(\w*)\)?", s)
        t = re.search(r"0x([0-9a-f]+)", s) if "BRA" in s else None
        if t:
            tgt = int(t.group(1), 16)
            if tgt < a and tgt in addr:
                body = ins[addr[tgt]:i + 1]
                n = sum(1 for _, x in body if op(x) == "HMMA")
                if best is None or n > best[0]:
                    best = (n, addr[tgt], i)
    allc = Counter(op(x) for _, x in ins)
    print(f"== {name}: {len(ins)} instr, HMMA {allc['HMMA']}, LDL {allc['LDL']}, STL {allc['STL']}")
    if best:
        body = ins[best[1]:best[2] + 1]
        c = Counter(op(x) for _, x in body)
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(14))
        print(f"   loop [{best[1]}..{best[2]}]: {len(body)} instr, HMMA {c['HMMA']}, LDL {c['LDL']}, STL {c['STL']}, NOP {c['NOP']}")
        print(f"   mix: {top}")
