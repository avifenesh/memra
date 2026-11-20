"""Compare two DFly2 drafter tapes (`memra-hy4f-dfly-tape-v1`) of the same request and anchors.

usage: python3 compare_dfly.py ORACLE NATIVE [--atol X] [--json OUT]

Reports, over every anchor and draft step: the largest absolute logit difference, the steps
whose argmax differs, and of those the ones that are near ties of the oracle's logits (the
native argmax scores within `atol` of the oracle's maximum), and the first anchor that moves
most. With `--atol` the tapes pass when every logit is within the band and every argmax equals
the oracle's or is a near tie. A forced native tape drafts along the oracle's tokens
(`forced_drafts`); its argmax is still its own. The verdict prints last; the exit code is 0 for
a pass, 1 for a fail, 2 for tapes that do not describe the same rounds.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("oracle")
ap.add_argument("native")
ap.add_argument("--atol", type=float, default=None)
ap.add_argument("--json", default="")
a = ap.parse_args()


def load(d):
    t = json.load(open(os.path.join(d, "tape.json")))
    raw = open(os.path.join(d, t["logits"]["file"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == t["logits"]["sha256"], f"{d}: logits are not the ones the tape declares"
    return t, np.frombuffer(raw, dtype="<f4").reshape(t["logits"]["shape"])


to, lo = load(a.oracle)
tn, ln = load(a.native)
same = all(to[k] == tn[k] for k in ["request", "token_ids", "k", "vocab", "anchors"])
if not same:
    print(json.dumps({"verdict": "mismatch", "why": "the tapes describe other rounds"}))
    sys.exit(2)
diff = np.abs(lo.astype(np.float64) - ln.astype(np.float64))
max_abs = float(diff.max())
where = np.unravel_index(int(diff.argmax()), diff.shape)
ao = lo.argmax(-1)
an = ln.argmax(-1)
differ = np.argwhere(ao != an)
near = 0
far = []
for n, s in differ:
    gap = float(lo[n, s, ao[n, s]] - lo[n, s, an[n, s]])
    if a.atol is not None and gap <= a.atol:
        near += 1
    else:
        far.append({"anchor": int(to["anchors"][n]), "step": int(s), "oracle": int(ao[n, s]), "native": int(an[n, s]),
                    "gap": gap})
per_anchor = diff.max(axis=(1, 2))
own_o = to["drafts"]
own_n = tn["drafts"]
report = {
    "request": to["request"],
    "anchors": len(to["anchors"]),
    "k": to["k"],
    "oracle": to.get("engine"), "oracle_red": to.get("red_arm"),
    "native": tn.get("engine"), "native_red": tn.get("red_arm"), "native_forced": tn.get("forced_drafts"),
    "bitwise_equal": bool(np.array_equal(lo.view(np.uint32), ln.view(np.uint32))),
    "max_abs": max_abs,
    "max_at": {"anchor": int(to["anchors"][where[0]]), "step": int(where[1]), "token": int(where[2])},
    "per_anchor_max_p50": float(np.median(per_anchor)),
    "per_anchor_max_p99": float(np.quantile(per_anchor, 0.99)),
    "argmax_steps": int(ao.size),
    "argmax_differ": int(len(differ)),
    "argmax_differ_near_ties": near,
    "argmax_differ_far": far[:20],
    "own_drafts_equal_rounds": sum(int(x == y) for x, y in zip(own_o, own_n)),
    "oracle_mean_accepted_teacher_forced": float(np.mean(to.get("accepted_teacher_forced") or [0])),
    "native_mean_accepted_teacher_forced": float(np.mean(tn.get("accepted_teacher_forced") or [0])),
}
if a.atol is not None:
    report["atol"] = a.atol
    report["verdict"] = "pass" if (max_abs <= a.atol and not far and not tn.get("red_arm")) else "fail"
if a.json:
    json.dump(report, open(a.json, "w"), indent=1)
print(json.dumps(report))
sys.exit(0 if report.get("verdict", "pass") == "pass" else 1)
