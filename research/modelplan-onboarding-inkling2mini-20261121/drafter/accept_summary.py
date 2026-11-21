#!/usr/bin/env python3
"""Lane S: acceptance of teacher-forced proposals per position.

For a sequence S (the target's greedy decode of the prompt) and proposals d[P] for the anchor
S[P] at every generated position P, a(P) is the number of leading proposals equal to the target's
next tokens. The step simulation follows a greedy speculative run along S: start at the first
generated position, accept a(P), move to P + a(P) + 1; it is the acceptance a K=7 run on this
target and drafter would have. The target's tokens are S (the GPU route) or, for the oracle pair,
the oracle's own argmax along S (y[q] predicts position q + 1), counted only while it agrees with S.

usage: accept_summary.py PP_DIR [CPU_ARMS.jsonl ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def accept(props, seq, P, target=None):
    a = 0
    for j, d in enumerate(props):
        q = P + 1 + j
        if q >= len(seq):
            break
        want = seq[q]
        if target is not None and target[q - 1] != want:
            break
        if d != want:
            break
        a += 1
    return a


def simulate(per_pos, T0, n):
    P, steps, acc = T0, 0, 0
    while P < n - 1:
        a = per_pos.get(P, 0)
        steps += 1
        acc += a
        P += a + 1
    return steps, acc


pp = Path(sys.argv[1])
metas = {}
for jf in sorted(pp.glob("*.json")):
    if jf.name.endswith(".oracle.json"):
        continue
    m = json.load(open(jf))
    o = pp / f"{m['id']}.oracle.json"
    m["oracle"] = json.load(open(o))["argmax"] if o.exists() else None
    metas[m["id"]] = m
arms = defaultdict(dict)  # arm -> id -> proposals
for pid, m in metas.items():
    arms["gpu-drafter/gpu-taps"][pid] = m["proposals"]
for f in sys.argv[2:]:
    for l in open(f):
        r = json.loads(l)
        arms[f"cpu-{r['arm']}/{r['taps']}-taps"][r["id"]] = r["proposals"]

agree = {pid: (sum(1 for q in range(m["first_anchor"], len(m["seq"])) if m["oracle"] and m["oracle"][q - 1] == m["seq"][q]),
               len(m["seq"]) - m["first_anchor"]) for pid, m in metas.items() if m["oracle"]}
if agree:
    a = sum(x for x, _ in agree.values()); b = sum(y for _, y in agree.values())
    print(f"oracle argmax == GPU greedy on {a}/{b} generated positions")
print("arm\ttarget\tkind\tsteps\taccepted\tacceptance length\ttokens per verify\tmean a(P)")
for arm, by in sorted(arms.items()):
    for tgt in ["gpu", "oracle"]:
        tot = defaultdict(lambda: [0, 0, 0, 0])
        for pid, props in by.items():
            m = metas[pid]
            if tgt == "oracle" and not m["oracle"]:
                continue
            T0, seq = m["first_anchor"], m["seq"]
            per = {T0 + i: accept(p, seq, T0 + i, m["oracle"] if tgt == "oracle" else None) for i, p in enumerate(props)}
            s, acc = simulate(per, T0, len(seq))
            for key in (m["kind"], "all"):
                t = tot[key]; t[0] += s; t[1] += acc; t[2] += sum(per.values()); t[3] += len(per)
        for key, (s, acc, sp, npos) in sorted(tot.items()):
            if s:
                print(f"{arm}\t{tgt}\t{key}\t{s}\t{acc}\t{acc / s:.3f}\t{(acc + s) / s:.3f}\t{sp / npos:.3f}")
