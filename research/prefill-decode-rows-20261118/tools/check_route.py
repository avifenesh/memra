#!/usr/bin/env python3
"""check_route.py: the engine's own MoE selections (MEMRA_MOE_WEIGHT_TRACE) at row 41 against the
host top-k row-bisect prints (route lines), for prefill (t=42 forward, last token) and decode step 41."""
import re

trace = [l.split() for l in open("runs/orn-trace-moe.txt")]
t1 = [l for l in trace if l[1] == "1"]
dec = {int(l[0]): l[2] for l in t1[40 * 41:40 * 42]}
pre = {}
for l in trace:
    if l[1] == "42" and int(l[0]) not in pre:  # first t=42 forward is the prefill (prime comes last)
        pairs = l[2].split(",")
        pre[int(l[0])] = ",".join(pairs[-8:])
log = open("runs/orn-route/log").read()
blk = log.split("=== row 41")[1].split("=== row")[0]
ok = {"prefill": 0, "decode": 0}
bad = []
wdiff = 0.0
for m in re.finditer(r"route row=41 layer=(\d+) prefill=(\S+) decode=(\S+)", blk):
    il = int(m.group(1))
    for name, host, eng in (("prefill", m.group(2), pre[il]), ("decode", m.group(3), dec[il])):
        hs = {int(x.split(":")[0]): float(x.split(":")[2]) for x in host.split(",")}
        es = {int(x.split(":")[0]): float(x.split(":")[1]) for x in eng.split(",")}
        if set(hs) == set(es):
            ok[name] += 1
            # engine weights are renormalised over the top-k; renormalise host softmax the same way
            s = sum(hs.values())
            wdiff = max(wdiff, max(abs(hs[k] / s - es[k]) for k in hs))
        else:
            bad.append((name, il, sorted(hs), sorted(es)))
print("layers whose host top-k equals the engine's selection:", ok, "of", len(pre))
print("max |host weight - engine weight| over matching layers:", f"{wdiff:.2e}")
for b in bad:
    print("MISMATCH", b)
