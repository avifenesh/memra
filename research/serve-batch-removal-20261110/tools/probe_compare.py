#!/usr/bin/env python3
"""memra #1158: compare the plain and the default (spec) arm of probe_arm.py.

For every prompt: chat text and completion_tokens must be equal, and the native completions
token ids must be equal. A difference prints the first differing index with both streams
around it. Exit 1 on any difference.

usage: probe_compare.py PLAIN_DIR SPEC_DIR
"""
import json
import os
import sys

PLAIN, SPEC = sys.argv[1], sys.argv[2]
bad = 0
for name in ("gate", "sched", "essay"):
    def load(d, kind):
        return json.load(open(os.path.join(d, f"{name}-{kind}.json")))

    def text(r):
        m = r["choices"][0]["message"]
        return (m.get("content") or "") + (m.get("reasoning") or "")

    pc, sc = load(PLAIN, "chat"), load(SPEC, "chat")
    pt, st = text(pc), text(sc)
    pn, sn = pc["usage"]["completion_tokens"], sc["usage"]["completion_tokens"]
    spec = sc["usage"].get("spec") or {}
    ok_text = pt == st and pn == sn
    if not ok_text:
        n = min(len(pt), len(st))
        d = next((j for j in range(n) if pt[j] != st[j]), n)
        print(f"{name}: TEXT DIVERGE at char {d}: plain={pt[d-40:d+40]!r} spec={st[d-40:d+40]!r} "
              f"(chars {len(pt)}/{len(st)}, tokens {pn}/{sn})")
    pi, si = load(PLAIN, "completions")["tokens"], load(SPEC, "completions")["tokens"]
    ok_ids = pi == si
    if not ok_ids:
        n = min(len(pi), len(si))
        d = next((j for j in range(n) if pi[j] != si[j]), n)
        print(f"{name}: ID DIVERGE at token {d}: plain={pi[max(0,d-8):d+8]} spec={si[max(0,d-8):d+8]} "
              f"(lens {len(pi)}/{len(si)})")
    bad += (not ok_text) + (not ok_ids)
    print(f"{name}: text {'equal' if ok_text else 'DIFFERENT'} ({len(pt)} chars, {pn}/{sn} tokens, "
          f"finish {pc['choices'][0].get('finish_reason')}/{sc['choices'][0].get('finish_reason')}); "
          f"ids {'equal' if ok_ids else 'DIFFERENT'} ({len(pi)}/{len(si)}); "
          f"spec rounds={spec.get('rounds')} drafted={spec.get('drafted')} accepted={spec.get('accepted')}")
print("VERDICT:", "EQUAL" if bad == 0 else f"{bad} DIFFERENCES")
sys.exit(1 if bad else 0)
