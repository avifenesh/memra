#!/usr/bin/env python3
"""memra#1255 diag log analysis: for every gate pass, the charges of admitted sessions that had
not yet built their route state (no dspark-prime-start line yet), and whether the reading covered
them too. Also the materialization delta each cold DSpark session showed at its prime start.
usage: analyze_diag.py serve.log"""
import re, sys

lines = open(sys.argv[1], errors="replace").read().splitlines()
RE_GATE = re.compile(r"gate-pass id=(?:chatcmpl-)?(\S{1,8})\S* prompt=(\d+) path=(\w+) cost=(\d+)MB draft=(\d+)MB reserve=(\d+)MB required=(\d+)MB required_eager=(\d+)MB eager_arm=(\w+) booked=(\d+)MB limiting_free=(\d+)MB active=(\d+)")
RE_SNAP = re.compile(r"\[diag1255\] (.*?): driver_free=(\d+)MB pool_res=(\d+)MB pool_used=(\d+)MB pool_unused=(\d+)MB eff_free=(\d+)MB")
unstarted = {}  # id -> charge MiB
pending_gate = None
starts = {}
out = []
for i, l in enumerate(lines):
    m = RE_GATE.search(l)
    if m:
        rid, prompt, path, cost, draft, reserve, required, req_e, eager, booked, free, active = m.groups()
        owed = sum(unstarted.values())
        verdict = ""
        if owed:
            verdict = f" OWED {owed}MiB by {sorted(unstarted)}; reading-owed={int(free) - owed} vs required {required}: {'SHORT' if int(free) - owed < int(required) else 'fits'}"
        out.append(f"L{i+1} gate {rid} prompt={prompt} path={path} charge={int(cost)}(+draft incl) required={required} free={free} active={active}{verdict}")
        pending_gate = (rid, int(cost))
        continue
    m = RE_SNAP.search(l)
    if m:
        tag = m.group(1)
        eff = int(m.group(6))
        if tag.startswith("after-admit id="):
            rid = re.search(r"id=(?:chatcmpl-)?(\S{1,8})", tag).group(1)
            dspark = "dspark_on=true" in tag and "cache=false" in tag and "dspark_sess=false" in tag
            if dspark and pending_gate and pending_gate[0] == rid:
                unstarted[rid] = pending_gate[1]
            out.append(f"L{i+1}   admitted {rid} {'COLD-DSPARK (no state)' if dspark else 'state built'} eff={eff}")
        elif tag.startswith("dspark-prime-start-before"):
            rid = re.search(r"id=(?:chatcmpl-)?(\S{1,8})", tag).group(1)
            starts[rid] = eff
        elif tag.startswith("dspark-prime-chunk-before"):
            rid = re.search(r"id=(?:chatcmpl-)?(\S{1,8})", tag).group(1)
            if rid in starts and rid in unstarted:
                out.append(f"L{i+1}   {rid} prime start materialized {starts[rid] - eff}MiB (charged {unstarted[rid]}MiB)")
                unstarted.pop(rid)
        elif tag.startswith("step-oom"):
            out.append(f"L{i+1}   STEP OOM {tag} eff={eff} driver={m.group(2)} unused={m.group(5)}")
    if "step OOM parked" in l or "[diag1255-ring]" in l:
        out.append(f"L{i+1}   {l[:220]}")
print("\n".join(out))
