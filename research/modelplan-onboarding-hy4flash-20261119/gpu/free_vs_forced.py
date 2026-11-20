"""The route's free and forced runs of one request agree bit for bit up to the first flip.

The forced run applies the oracle's picks; the free run its own. Where the route's own picks
equal the oracle's at every layer for every position up to p (in call order: the prefill's
positions, then each decode position), both runs computed the same program on the same
selections, so their logits must be equal bit for bit at every position before the first
position with a differing pick. Prints that position and the check.

usage: python free_vs_forced.py ORACLE_DIR FORCED_DIR FREE_DIR
"""

import json
import sys

import numpy as np


def tape(d):
    return json.load(open(f"{d}/tape.json"))


o, f, r = (tape(d) for d in sys.argv[1:4])
osel = np.fromfile(f"{sys.argv[1]}/{o['router']['sel_file']}", dtype="<i4")
fsel = np.fromfile(f"{sys.argv[2]}/{f['router']['sel_file']}", dtype="<i4")
calls = [(0, o["decode_from"])] + [(t, 1) for t in range(o["decode_from"], o["positions"])]
first = o["positions"]
frec = {(x["call"], x["layer"]): x for x in f["router"]["records"]}
for x in o["router"]["records"]:
    y = frec[(x["call"], x["layer"])]
    q, k = x["queries"], x["k"]
    a = osel[x["sel_offset"]:x["sel_offset"] + q * k].reshape(q, k)
    b = fsel[y["sel_offset"]:y["sel_offset"] + q * k].reshape(q, k)
    d = np.nonzero((a != b).any(axis=1))[0]
    if d.size:
        first = min(first, calls[x["call"]][0] + int(d[0]))
oisel = np.fromfile(f"{sys.argv[1]}/{o['indexer']['sel_file']}", dtype="<i4")
fisel = np.fromfile(f"{sys.argv[2]}/{f['indexer']['sel_file']}", dtype="<i4")
firec = {(x["call"], x["layer"]): x for x in f["indexer"]["records"]}
for x in o["indexer"]["records"]:
    y = firec[(x["call"], x["layer"])]
    q, k = x["queries"], x["k"]
    a = oisel[x["sel_offset"]:x["sel_offset"] + q * k].reshape(q, k)
    b = fisel[y["sel_offset"]:y["sel_offset"] + q * k].reshape(q, k)
    d = np.nonzero((a != b).any(axis=1))[0]
    if d.size:
        first = min(first, calls[x["call"]][0] + int(d[0]))
v = o["vocab"]
lf = np.fromfile(f"{sys.argv[2]}/{f['logits']['file']}", dtype="<f4").reshape(-1, v)
lr = np.fromfile(f"{sys.argv[3]}/{r['logits']['file']}", dtype="<f4").reshape(-1, v)
eq = (lf == lr).all(axis=1)
print(json.dumps({"request": o["request"], "first_flip_position": first,
                  "equal_before_flip": bool(eq[:first].all()), "positions_before_flip": first,
                  "first_unequal_position": int(np.argmin(eq)) if not eq.all() else None}))
