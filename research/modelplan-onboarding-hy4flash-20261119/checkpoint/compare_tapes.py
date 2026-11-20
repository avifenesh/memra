"""Compare two `memra-checkpoint-tape-v2` directories (an oracle and a candidate) as the gate does.

A diagnostic beside the gate (`crates/memra-cli/src/checkpoint_tape.rs` decides). Order: the
request identity, then every indexer call's recalled groups, every router call's experts and
every Engram call's table rows, then the logits. A selection that differs is attributed before any band is read:
the oracle's own group logits say whether the flipped groups were tied within `--tie-band`
(a near-tie a lower-precision backend may flip) or not (a wrong program).

Prints one JSON line: selections (calls, queries, differing queries, near-tie flips, hard
flips), Engram rows (calls, differing), logits (max_abs, max_rel, worst position and token,
argmax agreement per position, KL of the softmax per position: max and mean) and, with
`--band ABS REL`, whether every logit is within `ABS + REL * |oracle|`.

usage: venv/bin/python compare_tapes.py ORACLE_DIR CANDIDATE_DIR [--band ABS REL] [--tie-band X]
       [--positions all|decode]
"""

import argparse
import json
import math
import sys
from array import array

import numpy as np


def load(d):
    t = json.load(open(f"{d}/tape.json"))
    return t


def floats(path, offset=0, count=None):
    a = array("f")
    with open(path, "rb") as f:
        f.seek(offset * 4)
        data = f.read() if count is None else f.read(count * 4)
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def ints(path):
    a = array("i")
    a.frombytes(open(path, "rb").read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


ap = argparse.ArgumentParser()
ap.add_argument("oracle")
ap.add_argument("candidate")
ap.add_argument("--band", nargs=2, type=float)
ap.add_argument("--tie-band", type=float, default=0.024, help="indexer, absolute: INDEXER_TIE_BAND in checkpoint_tape.rs")
ap.add_argument("--router-tie-band", type=float, default=0.009, help="absolute: ROUTER_TIE_BAND in checkpoint_tape.rs")
a = ap.parse_args()
o, c = load(a.oracle), load(a.candidate)
out = {"oracle": o["engine"], "candidate": c["engine"], "request": o["request"]}
for key in ["token_ids", "decode_from", "vocab", "positions"]:
    if o[key] != c[key]:
        print(json.dumps(out | {"status": "failed", "error": f"request mismatch on {key}"}))
        sys.exit(1)

# ---- selections ---------------------------------------------------------------------------------
osel, csel = ints(f"{a.oracle}/{o['indexer']['sel_file']}"), ints(f"{a.candidate}/{c['indexer']['sel_file']}")
olog = floats(f"{a.oracle}/{o['indexer']['logit_file']}")
orec = {(r["call"], r["layer"]): r for r in o["indexer"]["records"]}
crec = {(r["call"], r["layer"]): r for r in c["indexer"]["records"]}
sel = {"calls": len(orec), "queries": 0, "differing_queries": 0, "near_tie_flips": 0, "hard_flips": 0,
       "missing_calls": sorted(set(orec) ^ set(crec))}
for key, r in sorted(orec.items()):
    if key not in crec:
        continue
    q = crec[key]
    if (r["queries"], r["k"], r["groups"]) != (q["queries"], q["k"], q["groups"]):
        sel["missing_calls"].append(list(key))
        continue
    k, ng = r["k"], r["groups"]
    for i in range(r["queries"]):
        sel["queries"] += 1
        a_set = set(osel[r["sel_offset"] + i * k : r["sel_offset"] + (i + 1) * k])
        b_set = set(csel[q["sel_offset"] + i * k : q["sel_offset"] + (i + 1) * k])
        if a_set == b_set:
            continue
        sel["differing_queries"] += 1
        row = olog[r["logit_offset"] + i * ng : r["logit_offset"] + (i + 1) * ng]
        # the last group the oracle's top-k kept (k - 1 groups besides the query's own): a flip
        # is a near tie when every group that entered or left lies within tie_band of it
        finite = sorted((v for v in row if math.isfinite(v)), reverse=True)
        kth = finite[k - 2] if len(finite) >= k - 1 else -math.inf
        moved = (a_set ^ b_set)
        if all(g < ng and math.isfinite(row[g]) and abs(row[g] - kth) <= a.tie_band
               for g in moved):
            sel["near_tie_flips"] += 1
        else:
            sel["hard_flips"] += 1
out["selections"] = sel

# ---- router picks --------------------------------------------------------------------------------
def router_arrays(d, t):
    r = t.get("router")
    if r is None:
        return None
    return (np.fromfile(f"{d}/{r['sel_file']}", dtype="<i4"), np.fromfile(f"{d}/{r['score_file']}", dtype="<f4"),
            {(x["call"], x["layer"]): x for x in r["records"]})


ro, rc = router_arrays(a.oracle, o), router_arrays(a.candidate, c)
rt = {"calls": 0, "queries": 0, "differing_queries": 0, "near_tie_flips": 0, "hard_flips": 0, "self_topk_failures": 0,
      "missing_calls": [] if ro and rc else ["no router records"]}
if ro and rc:
    rt["missing_calls"] = sorted(set(ro[2]) ^ set(rc[2]))
    for key, r in sorted(ro[2].items()):
        q = rc[2].get(key)
        if q is None:
            continue
        rt["calls"] += 1
        if (r["queries"], r["k"], r["experts"]) != (q["queries"], q["k"], q["experts"]):
            rt["missing_calls"].append(list(key))
            continue
        k, e = r["k"], r["experts"]
        A = ro[0][r["sel_offset"]:r["sel_offset"] + r["queries"] * k].reshape(-1, k)
        B = rc[0][q["sel_offset"]:q["sel_offset"] + q["queries"] * k].reshape(-1, k)
        SA = ro[1][r["score_offset"]:r["score_offset"] + r["queries"] * e].reshape(-1, e)
        SB = rc[1][q["score_offset"]:q["score_offset"] + q["queries"] * e].reshape(-1, e)
        for S, P in [(SA, A), (SB, B)]:
            if not np.isfinite(S).all() or ((P < 0) | (P >= e)).any() or (np.diff(P, axis=1) <= 0).any():
                rt["invalid_calls"] = rt.get("invalid_calls", 0) + 1
        own = np.sort(np.argsort(-SB, axis=1, kind="stable")[:, :k], axis=1)
        rt["self_topk_failures"] += int((own != B).any(axis=1).sum())
        rt["queries"] += A.shape[0]
        for i in np.nonzero((A != B).any(axis=1))[0]:
            rt["differing_queries"] += 1
            boundary = float(SA[i, A[i]].min())
            moved = set(A[i].tolist()) ^ set(B[i].tolist())
            if all(abs(float(SA[i, g]) - boundary) <= a.router_tie_band for g in moved):
                rt["near_tie_flips"] += 1
            else:
                rt["hard_flips"] += 1
out["router"] = rt

# ---- Engram rows --------------------------------------------------------------------------------
oe = {(r["call"], r["layer"]): r["rows"] for r in o["engram"]}
ce = {(r["call"], r["layer"]): r["rows"] for r in c["engram"]}
out["engram"] = {"calls": len(oe), "differing": sum(1 for k in oe if ce.get(k) != oe[k]),
                 "missing": len(set(oe) ^ set(ce))}

# ---- logits -------------------------------------------------------------------------------------
V, T = o["vocab"], o["positions"]
x = np.fromfile(f"{a.oracle}/logits.f32", dtype="<f4").reshape(T, V).astype(np.float64)
y = np.fromfile(f"{a.candidate}/logits.f32", dtype="<f4").reshape(T, V).astype(np.float64)
nonfinite = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
d = np.abs(x - y)
flat = int(np.argmax(d))
ax, ay = np.argmax(x, axis=1), np.argmax(y, axis=1)  # the first of equal maxima
lx = x - x.max(axis=1, keepdims=True)
lx -= np.log(np.exp(lx).sum(axis=1, keepdims=True))
ly = y - y.max(axis=1, keepdims=True)
ly -= np.log(np.exp(ly).sum(axis=1, keepdims=True))
kl = (np.exp(lx) * (lx - ly)).sum(axis=1)
per_pos = d.max(axis=1)
out["logits"] = {"positions": T, "max_abs": float(d.max()), "max_rel": float((d / np.maximum(np.abs(x), 1e-6)).max()),
                 "worst_position": flat // V, "worst_token": flat % V, "argmax_equal": int((ax == ay).sum()),
                 "argmax_differs_at": [int(p) for p in np.nonzero(ax != ay)[0][:20]],
                 "kl_max": float(kl.max()), "kl_mean": float(kl.mean()),
                 "max_abs_prefill": float(per_pos[: o["decode_from"]].max()),
                 "max_abs_decode": float(per_pos[o["decode_from"]:].max()) if T > o["decode_from"] else 0.0,
                 "nonfinite": nonfinite}
argmax_equal = out["logits"]["argmax_equal"]
viol = None
if a.band:
    over = d > a.band[0] + a.band[1] * np.abs(x)
    if over.any():
        i = int(np.argmax(over))
        viol = (i // V, i % V, float(d.flat[i]))
ok = (not sel["missing_calls"] and sel["hard_flips"] == 0 and not rt["missing_calls"] and rt["hard_flips"] == 0
      and rt["self_topk_failures"] == 0 and not rt.get("invalid_calls") and out["engram"]["differing"] == 0
      and out["engram"]["missing"] == 0 and argmax_equal == T and nonfinite == 0)
if a.band:
    out["band"] = {"max_abs": a.band[0], "max_rel": a.band[1], "first_violation": viol}
    ok = ok and viol is None
forced = bool(o.get("forced_selections") or c.get("forced_selections"))
out["forced_selections"] = forced
# a forced tape executed the other tape's picks: its numbers are a calibration, never a verdict
out["status"] = "calibration" if forced else ("passed" if ok else "failed")
print(json.dumps(out))
sys.exit(0 if ok and not forced else 1)
