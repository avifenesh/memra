#!/usr/bin/env python3
"""Steady-state decode cell. Primes the prompt once (so the prefix restores and TTFT ~ 0),
then repeats identical requests; each rep's moe-cache delta is attributed to its own tokens.
usage: steady.py <log> <tag> <arm> <prompt_idx> <max_tokens> <reps>
"""
import json, re, statistics, subprocess, sys

LOG, TAG, ARM, PIDX, MT, REPS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
CNT = re.compile(r"hits=(\d+) misses=(\d+) hit_rate=[\d.]+ staged_bytes=(\d+)")
EPI = re.compile(r"\[moe-fused-epi\] snapshot dispatches=(\d+)")

# glm5_next has 42 routed-MoE layers, so a decoded token that takes the fused arm on every one of
# them is exactly 42 dispatches. engagement = dispatches_per_token / 42. Anything well under 1.0
# means the arm fell closed to the sequential loop (the SLRU could not hold a token's whole
# 3*n_used working set), and a cell measured in that state is NOT measuring the fusion.
MOE_LAYERS = 42

def snap():
    out = subprocess.run(["grep", "-E", r"\[moe-cache\] snapshot", LOG],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return (0, 0, 0)
    m = CNT.search(out[-1])
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)

def snap_epi():
    out = subprocess.run(["grep", "-E", r"\[moe-fused-epi\] snapshot", LOG],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    m = EPI.search(out[-1])
    return int(m.group(1)) if m else None

def run(mt, label):
    a, ae = snap(), snap_epi()
    p = subprocess.run([sys.executable, "/home/ubuntu/probe.py", label, ARM, str(mt), PIDX],
                       capture_output=True, text=True)
    try:
        r = json.loads(p.stdout.strip())
    except Exception:
        return {"error": (p.stdout + p.stderr)[:300]}
    b, be = snap(), snap_epi()
    ct = r.get("completion_tokens") or 0
    if ct:
        r["acc_per_tok"] = round((b[0]-a[0]+b[1]-a[1])/ct, 1)
        r["miss_per_tok"] = round((b[1]-a[1])/ct, 1)
        r["MB_per_tok"] = round((b[2]-a[2])/2**20/ct, 1)
        if ae is not None and be is not None:
            r["epi_per_tok"] = round((be-ae)/ct, 2)
            r["epi_engagement"] = round((be-ae)/ct/MOE_LAYERS, 4)
    return r

run(24, f"{TAG}-prime")   # establish the prefix / affinity checkpoint
rows = []
for i in range(REPS):
    r = run(MT, f"{TAG}-r{i}")
    rows.append(r)
    print(json.dumps(r), flush=True)
ok = [r for r in rows if r.get("server_toks")]
if ok:
    st = sorted(r["server_toks"] for r in ok)
    print(json.dumps({"tag": TAG, "STEADY": True, "arm": ARM, "prompt_idx": PIDX,
                      "max_tokens": MT, "n": len(ok),
                      "server_toks_median": round(statistics.median(st), 3),
                      "server_toks_all": st,
                      "ms_per_tok_median": round(1000/statistics.median(st), 2),
                      "ttft_all": [r["ttft_s"] for r in ok],
                      "MB_per_tok_median": round(statistics.median([r.get("MB_per_tok", 0) for r in ok]), 1),
                      "miss_per_tok_median": round(statistics.median([r.get("miss_per_tok", 0) for r in ok]), 1),
                      "acc_per_tok_median": round(statistics.median([r.get("acc_per_tok", 0) for r in ok]), 1),
                      "epi_per_tok_median": round(statistics.median([r.get("epi_per_tok", 0) for r in ok]), 2),
                      "epi_engagement_median": round(statistics.median([r.get("epi_engagement", 0) for r in ok]), 4),
                      "shas": sorted({r["out_sha"] for r in ok}),
                      "ctoks": sorted({r["completion_tokens"] for r in ok}),
                      "head": ok[-1]["head"]}), flush=True)
