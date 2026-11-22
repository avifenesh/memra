#!/usr/bin/env python3
"""Fill the step 11 README draft's tables from the replay receipts. usage: fill.py DRAFT OUT"""
import json
import sys

src, out = sys.argv[1:3]
s = open(src).read()
A = json.load(open("replay/acceptance-arms.json"))["workloads"]
rows = []
for n, w in A.items():
    f = lambda a: f"{w[a]['served_fixed5']['rounds']}, {w[a]['served_fixed5']['accepted']} / {w[a]['served_fixed5']['drafted']}"
    rows.append(f"| {n} | {w['anchors']} | {w['nvfp4']['mean_kept_per_anchor']:.3f} / {w['bf16']['mean_kept_per_anchor']:.3f} / "
                f"{w['bf16-mlp']['mean_kept_per_anchor']:.3f} | {f('nvfp4')} | {f('bf16')} | {f('bf16-mlp')} |")
s = s.replace("@@ACCEPT@@", "\n".join(rows))
R = {r["name"]: r for r in map(json.loads, open("replay/replay-nv.jsonl"))}
rows = []
for n, r in R.items():
    L = len(r["m_greedy"])
    rows.append(f"| {n} | {sum(r['m_greedy']) / L:.3f} | {sum(r['m_chosen']) / L:.3f} | {sum(r['m_half']) / L:.3f} |")
s = s.replace("@@SAMPLED@@", "(drafts kept a five-draft round, mean over the anchors)\n\n" if False else "\n".join(rows))
q = json.load(open("replay/simq-nv-sampled.json"))
h = json.load(open("replay/sim-nv-sampledcost.json"))
pick = [
    ("five drafts every round (step 10)", h, "fixed5"),
    ("fixed 1", h, "fixed1"), ("fixed 2", h, "fixed2"), ("fixed 3", h, "fixed3"), ("fixed 4", h, "fixed4"),
    ("five drafts, deepseek-v4.1 `DraftGate` stop", h, "fixed5+stop"),
    ("last round's kept + 1", h, "last+1(floor1)"),
    ("step 11's first gate: per-depth survivals + `DraftGate`", h, "surv(a=0.25,init=1.0,stop=True,kmin=1,geo=False)"),
    ("per-depth survivals, no stop", h, "surv(a=0.25,init=1.0,stop=False,kmin=1,geo=False)"),
    ("drafter's raw probabilities", q, "qp(early=True,pause0=True,scale=1.0)"),
    ("**drafter's probabilities x the request's ratio (the gate)**", q, "qp(online a=0.25 w=2.0)"),
    ("every depth chosen in hindsight (bound)", h, "oracle"),
]
rows = []
for label, d, key in pick:
    xs = d[key]
    cells = [f"{1000 * (x['tokens'] - 1) / x['ms']:.2f}" for x in xs]
    tot = 1000 * sum(x["tokens"] - 1 for x in xs) / sum(x["ms"] for x in xs)
    rows.append(f"| {label} | " + " | ".join(cells) + f" | {tot:.2f} |")
s = s.replace("@@POLICIES@@", "\n".join(rows))
open(out, "w").write(s)
print("filled", out)
