"""Spread of the two selection surfaces between two `memra-checkpoint-tape-v2` tapes of one request.

For every router record (call, layer) and indexer record present in both tapes: the largest
absolute difference of the router selection scores (sigmoid plus bias) and of the finite indexer
group logits (both tapes must agree on which entries are -inf), the median magnitude of each
surface, the per-position logit difference (max, p50, p99), and the queries whose own picks
differ (counted, not attributed; `compare_tapes.py` attributes). Prints one JSON line.

usage: python score_spread.py REFERENCE_DIR CANDIDATE_DIR
"""

import json
import sys

import numpy as np


def load(d):
    t = json.load(open(f"{d}/tape.json"))
    arr = lambda name, dt: np.fromfile(f"{d}/{t[name[0]][name[1]]}", dtype=dt)
    return t, {
        "rsel": arr(("router", "sel_file"), "<i4"),
        "rsc": arr(("router", "score_file"), "<f4"),
        "isel": arr(("indexer", "sel_file"), "<i4"),
        "ilg": arr(("indexer", "logit_file"), "<f4"),
        "logits": np.fromfile(f"{d}/{t['logits']['file']}", dtype="<f4"),
    }


ref_t, ref = load(sys.argv[1])
can_t, can = load(sys.argv[2])
assert ref_t["token_ids"] == can_t["token_ids"] and ref_t["decode_from"] == can_t["decode_from"]
out = {"reference": ref_t["engine"], "candidate": can_t["engine"], "request": ref_t["request"]}

cr = {(r["call"], r["layer"]): r for r in can_t["router"]["records"]}
dmax, mags, rdiff, rq = 0.0, [], 0, 0
worst = None
for r in ref_t["router"]["records"]:
    c = cr.get((r["call"], r["layer"]))
    if c is None:
        continue
    q, e, k = r["queries"], r["experts"], r["k"]
    a = ref["rsc"][r["score_offset"]:r["score_offset"] + q * e]
    b = can["rsc"][c["score_offset"]:c["score_offset"] + q * e]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if d.max() > dmax:
        dmax = float(d.max())
        worst = (r["call"], r["layer"], int(d.argmax() // e))
    mags.append(np.abs(a))
    sa = ref["rsel"][r["sel_offset"]:r["sel_offset"] + q * k].reshape(q, k)
    sb = can["rsel"][c["sel_offset"]:c["sel_offset"] + q * k].reshape(q, k)
    rdiff += int((sa != sb).any(axis=1).sum())
    rq += q
out["router"] = {"queries": rq, "picks_differ": rdiff, "score_max_abs_diff": dmax,
                 "score_median_abs": float(np.median(np.concatenate(mags))) if mags else None,
                 "worst_call_layer_query": worst}

ci = {(r["call"], r["layer"]): r for r in can_t["indexer"]["records"]}
dmax, mags, idiff, iq, inf_mismatch = 0.0, [], 0, 0, 0
for r in ref_t["indexer"]["records"]:
    c = ci.get((r["call"], r["layer"]))
    if c is None:
        continue
    q, g, k = r["queries"], r["groups"], r["k"]
    a = ref["ilg"][r["logit_offset"]:r["logit_offset"] + q * g].astype(np.float64)
    b = can["ilg"][c["logit_offset"]:c["logit_offset"] + q * g].astype(np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    inf_mismatch += int((fa != fb).sum())
    both = fa & fb
    if both.any():
        dmax = max(dmax, float(np.abs(a[both] - b[both]).max()))
        mags.append(np.abs(a[both]))
    sa = ref["isel"][r["sel_offset"]:r["sel_offset"] + q * k].reshape(q, k)
    sb = can["isel"][c["sel_offset"]:c["sel_offset"] + q * k].reshape(q, k)
    idiff += int((sa != sb).any(axis=1).sum())
    iq += q
out["indexer"] = {"queries": iq, "picks_differ": idiff, "logit_max_abs_diff": dmax,
                  "logit_median_abs": float(np.median(np.concatenate(mags))) if mags else None,
                  "finite_pattern_mismatches": inf_mismatch}

v = ref_t["vocab"]
la = ref["logits"].reshape(-1, v).astype(np.float64)
lb = can["logits"].reshape(-1, v).astype(np.float64)
per = np.abs(la - lb).max(axis=1)
out["logits"] = {"positions": int(per.shape[0]), "max_abs": float(per.max()), "worst_position": int(per.argmax()),
                 "p50": float(np.percentile(per, 50)), "p99": float(np.percentile(per, 99)),
                 "argmax_equal": int((la.argmax(axis=1) == lb.argmax(axis=1)).sum()),
                 "max_abs_prefill": float(per[:ref_t["decode_from"]].max()),
                 "max_abs_decode": float(per[ref_t["decode_from"]:].max()) if per.shape[0] > ref_t["decode_from"] else None}
print(json.dumps(out))
