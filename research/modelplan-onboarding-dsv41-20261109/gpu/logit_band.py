"""Logit-level comparison of two runs of the same prompt against a reference tape (vendor oracle `.pt`, or a raw f32
logits file from dsv41-run): max |d|, the smallest atol that admits every logit at rtol 0 and at rtol 0.005, argmax and
top-k agreement, KL(ref || other), and with two hidden-state tapes the per-layer stream difference.

usage: logit_band.py REF.pt OTHER(.pt|.f32) [--step 0] [--json OUT]"""
import argparse
import json
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("other"); ap.add_argument("--step", type=int, default=0); ap.add_argument("--json")
a = ap.parse_args()
R = torch.load(a.ref, weights_only=False)
ref = R["full_logits"][a.step].double()
if a.other.endswith(".pt"):
    O = torch.load(a.other, weights_only=False)
    oth = O["full_logits"][a.step].double()
else:
    O = None
    oth = torch.frombuffer(bytearray(open(a.other, "rb").read()), dtype=torch.float32).double()
d = (oth - ref).abs()
out = {
    "vocab": ref.numel(),
    "max_abs": d.max().item(),
    "argmax_ref": int(ref.argmax()), "argmax_other": int(oth.argmax()),
    "atol_at_rtol0": d.max().item(),
    "atol_at_rtol0.005": (d - 0.005 * ref.abs()).clamp_min(0).max().item(),
    "max_rel": (d / ref.abs().clamp_min(1e-6)).max().item(),
    "ref_absmax": ref.abs().max().item(),
}
lp, lq = ref.log_softmax(-1), oth.log_softmax(-1)
out["kl_ref_other"] = (lp.exp() * (lp - lq)).sum().item()
for k in (5, 32):
    out[f"top{k}_overlap"] = len(set(ref.topk(k).indices.tolist()) & set(oth.topk(k).indices.tolist()))
if O is not None and R.get("hidden") is not None and O.get("hidden") is not None:
    per = []
    for li in range(R["hidden"].size(0)):
        x, y = R["hidden"][li].double(), O["hidden"][li].double()
        per.append(((x - y).norm() / x.norm()).item())
    out["hidden_rel_per_layer"] = per
    out["hidden_rel_max"] = max(per)
print(json.dumps({k: v for k, v in out.items() if k != "hidden_rel_per_layer"}, indent=1))
if "hidden_rel_per_layer" in out:
    print("hidden rel per layer:", " ".join(f"{x:.2e}" for x in out["hidden_rel_per_layer"]))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
