"""Chained drift between two valid realizations of the vendor program: per-layer stream of a tape (hidden [L, s, hc, d])
against a capture (s0/L{l}/out/x), with the same metrics the gate reports (global rel L2, worst token, cosine).
Used with the old-schedule GPU tape (one global max in sparse attention) against the tilelang-schedule capture, both
the vendor program on the same card: the band a full-trunk rewrite's chained drift is read against.

usage: chain_floor.py TAPE.pt CAPTURE_DIR OUT.json"""
import json
import os
import sys

import torch

tape, cap, out = sys.argv[1:4]
t = torch.load(tape, weights_only=False)
hid = t["hidden"]
idx = json.load(open(os.path.join(cap, "index.json")))["tensors"]
rows = []
for li in range(hid.size(0)):
    k = f"s0/L{li}/out/x"
    if k not in idx:
        continue
    f = os.path.join(cap, idx[k]["file"])
    b = torch.from_file(f, size=os.path.getsize(f), dtype=torch.uint8).view(torch.bfloat16).reshape(idx[k]["shape"])[0].double()
    a = hid[li].double()
    s = a.size(0)
    a2, b2 = a.reshape(s, -1), b.reshape(s, -1)
    rel = ((a2 - b2).norm() / b2.norm()).item()
    per = (a2 - b2).norm(dim=1) / b2.norm(dim=1).clamp_min(1e-30)
    cos = (a2 * b2).sum() / (a2.norm() * b2.norm())
    rows.append({"layer": li, "rel_l2": rel, "worst_tok": per.max().item(), "tok_median": per.median().item(), "cos": cos.item()})
    print(f"layer{li} rel {rel:.3e} worst {per.max().item():.3e} med {per.median().item():.3e} cos {cos.item():.8f}", flush=True)
json.dump(rows, open(out, "w"), indent=1)
