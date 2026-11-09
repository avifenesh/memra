"""Per-position KL of a teacher-forced run against the vendor tape, from both sides' top-32 logits.

A teacher-forced tape holds every position's lse and its 32 largest logits (`top_ids`, `top_vals`), and so do the
native `dsv41-run --tf` directory (top_ids.i32, top_vals.f32, lse.f32) and every other vendor tape. At each position
both distributions are known exactly on S, the ids in both top-32 sets; everything else is one "rest" bucket. The KL
of those coarsened distributions, sum over S of p log(p/q) plus p_rest log(p_rest/q_rest), is a lower bound of the
full KL(vendor || other) and is computed the same way for the native run and for the vendor's own floors, so their
ratio is a fair one.

Memorized text is near one-hot at most positions, where every realization agrees; the subsets that carry the signal are
the positions where the vendor is unsure: its entropy (lower bound: top-32 plus the rest as one bucket) above 1 nat,
and the top decile by that entropy. Each subset reports mean / p50 / p99 KL and top-1 agreement.

usage: tf_kl.py TAPE.pt OTHER (native --tf dir, or another vendor tape .pt) [--positions N] [--json OUT]
  --positions N   only the first N positions (to match a shorter floor tape of the same text)"""
import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("tape"); ap.add_argument("other"); ap.add_argument("--json"); ap.add_argument("--positions", type=int)
a = ap.parse_args()
T = torch.load(a.tape, weights_only=False)
vi, vv, vl = T["top_ids"].long(), T["top_vals"].double(), T["lse"].double()


def raw(path, dt):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt)


if a.other.endswith(".pt"):
    O = torch.load(a.other, weights_only=False)
    oi, ov, ol = O["top_ids"].long(), O["top_vals"].double(), O["lse"].double()
else:
    ol = raw(os.path.join(a.other, "lse.f32"), torch.float32).double()
    oi = raw(os.path.join(a.other, "top_ids.i32"), torch.int32).long().view(-1, 32)
    ov = raw(os.path.join(a.other, "top_vals.f32"), torch.float32).double().view(-1, 32)
n = min(len(vl), len(ol), a.positions or len(vl))
kl = torch.empty(n, dtype=torch.float64)
ent = torch.empty(n, dtype=torch.float64)
cover = torch.empty(n, dtype=torch.float64)
for r0 in range(0, n, 8192):
    r1 = min(n, r0 + 8192)
    p = (vv[r0:r1] - vl[r0:r1, None]).exp()
    q = (ov[r0:r1] - ol[r0:r1, None]).exp()
    eq = vi[r0:r1, :, None] == oi[r0:r1, None, :]
    both = eq.any(2)
    q_at = (eq * q[:, None, :]).sum(2)
    ps, qs = p * both, q_at
    pr = (1 - ps.sum(1)).clamp_min(1e-12)
    qr = (1 - qs.sum(1)).clamp_min(1e-12)
    term = torch.where(both, ps * (ps.clamp_min(1e-300).log() - qs.clamp_min(1e-300).log()), torch.zeros_like(ps))
    kl[r0:r1] = term.sum(1) + pr * (pr.log() - qr.log())
    prest = (1 - p.sum(1)).clamp_min(1e-12)
    ent[r0:r1] = -(p * p.clamp_min(1e-300).log()).sum(1) - prest * prest.log()
    cover[r0:r1] = ps.sum(1)
top1 = vi[:n, 0] == oi[:n, 0]
dec = ent >= ent.quantile(0.9) if n > 10 else torch.ones(n, dtype=torch.bool)


def stats(m):
    k = kl[m]
    if len(k) == 0:
        return {"positions": 0}
    return {"positions": int(m.sum()), "mean_kl": k.mean().item(), "p50_kl": k.median().item(),
            "p99_kl": k.quantile(0.99).item() if len(k) > 1 else k.item(), "max_kl": k.max().item(),
            "top1_agree": top1[m].double().mean().item(), "mean_vendor_entropy": ent[m].mean().item(),
            "mean_mass_on_shared_top32": cover[m].mean().item()}


out = {"tape": os.path.basename(a.tape), "other": os.path.basename(a.other.rstrip("/")), "positions": n,
       "subsets": {"all": stats(torch.ones(n, dtype=torch.bool)), "entropy_gt_1nat": stats(ent > 1.0),
                   "entropy_top_decile": stats(dec)}}
print(json.dumps(out, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
