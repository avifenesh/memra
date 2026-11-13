"""The contrast arm against the decode loop: each chat case scored by one teacher-forced prefill of the prompt plus the
vendor's tokens (the prefill program at every decode position, what a loop that prefills again would run), next to
memra's decode loop fed the same tokens; both against the vendor's decode program (REF, the tape with lse).

Per step, for each arm: KL(vendor || arm) from both top-32 sets plus one rest bucket (decode_gate.py's estimator) and
top-1 agreement with the vendor's greedy token; split at the window (positions below 128, where the vendor's decode
reads its ring in another order than its prefill, and from 128 on, where the two programs read the same keys in the same
order), plus the steps where the two arms' top-32 are bitwise equal.

usage: prefill_arm.py --tapes D --ref D --native D --arm D --cases a,b,... [--window 128] [--json OUT]
  --arm D   D/ID: `dsv41-run --ids PROMPT+GEN --tf D/ID` (lse.f32, top_ids.i32, top_vals.f32 per position)"""
import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
for k in ("tapes", "ref", "native", "arm", "cases"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--window", type=int, default=128)
ap.add_argument("--json")
a = ap.parse_args()


def raw(p, dt):
    return torch.frombuffer(bytearray(open(p, "rb").read()), dtype=dt)


def kl_top32(pi, pv, pl, qi, qv, ql):
    p = (pv - pl[:, None]).exp()
    q = (qv - ql[:, None]).exp()
    eq = pi[:, :, None] == qi[:, None, :]
    both = eq.any(2)
    qs = (eq * q[:, None, :]).sum(2)
    ps = p * both
    pr = (1 - ps.sum(1)).clamp_min(1e-12)
    qr = (1 - qs.sum(1)).clamp_min(1e-12)
    term = torch.where(both, ps * (ps.clamp_min(1e-300).log() - qs.clamp_min(1e-300).log()), torch.zeros_like(ps))
    return term.sum(1) + pr * (pr.log() - qr.log())


acc = {"below": {"steps": 0, "kl_dec": 0.0, "kl_pre": 0.0, "t1_dec": 0, "t1_pre": 0, "equal": 0},
       "above": {"steps": 0, "kl_dec": 0.0, "kl_pre": 0.0, "t1_dec": 0, "t1_pre": 0, "equal": 0}}
rows = []
for c in a.cases.split(","):
    T = torch.load(os.path.join(a.tapes, f"{c}.pt"), weights_only=False, map_location="cpu")
    R = torch.load(os.path.join(a.ref, f"{c}.pt"), weights_only=False, map_location="cpu")
    P, gen = len(T["prompt_ids"]), torch.tensor([int(x) for x in T["gen_ids"]]).long()
    G = len(gen)
    ri, rv, rl = R["top_ids"].long(), R["top_vals"].double(), R["lse"].double()
    nd = os.path.join(a.native, c)
    di = raw(f"{nd}/top_ids.i32", torch.int32).long().view(-1, 32)
    dv = raw(f"{nd}/top_vals.f32", torch.float32).double().view(-1, 32)
    dl = raw(f"{nd}/lse.f64", torch.float64)
    ad = os.path.join(a.arm, c)
    pos = torch.arange(G) + P - 1
    pi = raw(f"{ad}/top_ids.i32", torch.int32).long().view(-1, 32)
    # a full run directory holds every position; the banked one only the decode positions
    sel = pos if len(pi) > G else torch.arange(G)
    pi = pi[sel]
    pv = raw(f"{ad}/top_vals.f32", torch.float32).view(-1, 32)[sel]
    pl = raw(f"{ad}/lse.f32", torch.float32).double()[sel]
    kd = kl_top32(ri, rv, rl, di, dv, dl)
    kp = kl_top32(ri, rv, rl, pi, pv.double(), pl)
    td = (di[:, 0] == gen)
    tp = (pi[:, 0] == gen)
    eq = torch.tensor([bool(torch.equal(di[k], pi[k]) and torch.equal(dv[k].float(), pv[k])) for k in range(G)])
    r = {"case": c, "prompt": P, "steps": G}
    for name, m in (("below", pos < a.window), ("above", pos >= a.window)):
        n = int(m.sum())
        r[name] = {"steps": n, "kl_decode": kd[m].mean().item() if n else None,
                   "kl_prefill": kp[m].mean().item() if n else None, "top1_decode": int(td[m].sum()),
                   "top1_prefill": int(tp[m].sum()), "bitwise_equal": int(eq[m].sum())}
        s = acc[name]
        s["steps"] += n
        s["kl_dec"] += kd[m].sum().item()
        s["kl_pre"] += kp[m].sum().item()
        s["t1_dec"] += int(td[m].sum())
        s["t1_pre"] += int(tp[m].sum())
        s["equal"] += int(eq[m].sum())
    rows.append(r)
pooled = {}
for name, s in acc.items():
    n = max(s["steps"], 1)
    pooled[name] = {"steps": s["steps"], "kl_decode": s["kl_dec"] / n, "kl_prefill": s["kl_pre"] / n,
                    "top1_decode": s["t1_dec"], "top1_prefill": s["t1_pre"], "bitwise_equal": s["equal"]}
out = {"window": a.window, "pooled": pooled, "cases": rows}
print(json.dumps(pooled, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
