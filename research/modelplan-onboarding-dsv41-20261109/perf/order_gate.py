#!/usr/bin/env python3
"""The distribution-level gate of a reduction-order change: a new order against the current route, judged on
the vendor's own floor.

A change of a sum's order moves last bits, so the byte gates against the one-card route are re-baselined and
cannot judge it. This gate does: both routes run the decode lane's vendor tapes teacher-forced with the
tape's tokens (`dsv41-run --cases`), and each step's distribution is compared with the vendor's.

Inputs per case ID (the decode lane's layout, `../decode/decode_gate.py`):
  TAPES/ID.pt   the vendor's greedy tape            REF/ID.pt    the vendor teacher-forced with the tape
  FLOOR/ID.pt   the vendor's other realization (global-max attention) on the same tokens
  CUR/ID/       the current route teacher-forced     NEW/ID/      the new order teacher-forced

Per step, KL(REF || x) from both sides' top-32 plus one rest bucket (the decode gate's estimator), for
x = the current route, the new order and the floor. The statistic is the paired difference
  d = KL(REF || new) - KL(REF || cur)
with a bootstrap 95% interval of its mean over steps. Beside it the floor's own figure on the same steps,
f = KL(REF || FLOOR): its mean and the half-width of the bootstrap interval of that mean, which is how far
the vendor's two programs' KL moves with the choice of steps. A group passes when
  d's interval holds zero or lies below it (no measurable harm against the current route on the same
  tokens; an interval wholly above zero is not a pass on its own: it is reported with its size as a
  fraction of the current route's mean KL and the floor's half-width, for a decision),
  the new order's mean KL <= twice the floor's (the band of the prefill and decode gates; a group the
  current route is already over is judged by d alone and marked), and
  its top-1 agreement with the vendor's token is at most 1 point under the floor's (or two binomial standard
  errors when the group is small).
Groups: chat, long, needle (by the tape's prompt length and name) and all.

The router (with --routes-cur and --routes-new, the two routes' `dsv41-run --routes` logs): per routed row
and layer the new order's six experts against the current route's: rows with the same six, rows where only
the least-weight choice was swapped (the sixth against the seventh), any other difference; the largest
weight difference; and what a swap changes downstream: the relative difference of a decode step's MoE
output against the current route, for swapped rows next to unswapped ones.

usage: order_gate.py --tapes D --ref D --floor D --cur D --new D --cases a,b,... [--json OUT] [--boot N]
                     [--routes-cur DIR --routes-new DIR]
"""
import argparse
import json
import math
import os

import torch

ap = argparse.ArgumentParser()
for k in ("tapes", "ref", "floor", "cur", "new", "cases"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--json")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--routes-cur", help="dsv41-run --routes output of the current route (the --out directory)")
ap.add_argument("--routes-new", help="the same of the new order")
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--dim", type=int, default=5120)
a = ap.parse_args()


def raw(path, dt):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt)


def native(d):
    return {"top_ids": raw(os.path.join(d, "top_ids.i32"), torch.int32).long().view(-1, 32),
            "top_vals": raw(os.path.join(d, "top_vals.f32"), torch.float32).double().view(-1, 32),
            "lse": raw(os.path.join(d, "lse.f64"), torch.float64),
            "argmax": raw(os.path.join(d, "argmax.i32"), torch.int32).long()}


def kl_top32(pi, pv, pl, qi, qv, ql):
    """Per-row KL(p || q) over the ids both top-32 sets hold, plus one rest bucket each (the decode gate's)."""
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


def boot(x, n):
    """Mean of x and the bootstrap 95% interval of the mean over its entries (seeded)."""
    g = torch.Generator().manual_seed(20261110)
    m = len(x)
    means = torch.empty(n, dtype=torch.float64)
    for i in range(0, n, 500):
        k = min(500, n - i)
        idx = torch.randint(0, m, (k, m), generator=g)
        means[i:i + k] = x[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return x.mean().item(), lo, hi


def group_of(case, prompt):
    if "needle16800" in case:
        return "needle 257k"
    return "long" if prompt > 30000 else "chat"


steps = {}  # group -> lists
per_case = []
for c in a.cases.split(","):
    T = torch.load(os.path.join(a.tapes, f"{c}.pt"), weights_only=False, map_location="cpu")
    R = torch.load(os.path.join(a.ref, f"{c}.pt"), weights_only=False, map_location="cpu")
    F = torch.load(os.path.join(a.floor, f"{c}.pt"), weights_only=False, map_location="cpu")
    C, N = native(os.path.join(a.cur, c)), native(os.path.join(a.new, c))
    gen = torch.tensor([int(x) for x in T["gen_ids"]]).long()
    G = len(gen)
    assert len(C["argmax"]) == G and len(N["argmax"]) == G and len(F["gen_ids"]) == G, (c, G)
    ri, rv, rl = R["top_ids"].long(), R["top_vals"].double(), R["lse"].double()
    kc = kl_top32(ri, rv, rl, C["top_ids"], C["top_vals"], C["lse"])
    kn = kl_top32(ri, rv, rl, N["top_ids"], N["top_vals"], N["lse"])
    kf = kl_top32(ri, rv, rl, F["top_ids"].long(), F["top_vals"].double(), F["lse"].double())
    row = {"kc": kc, "kn": kn, "kf": kf, "t1c": (C["argmax"] == gen).double(), "t1n": (N["argmax"] == gen).double(),
           "t1f": (F["argmax"].long() == gen).double(),
           "same": torch.tensor([float(torch.equal(C["top_ids"][i], N["top_ids"][i]) and torch.equal(C["top_vals"][i], N["top_vals"][i]))
                                 for i in range(G)], dtype=torch.float64)}
    g = group_of(c, len(T["prompt_ids"]))
    for name in (g, "all"):
        steps.setdefault(name, []).append(row)
    per_case.append({"case": c, "group": g, "prompt": len(T["prompt_ids"]), "steps": G,
                     "kl_cur": kc.mean().item(), "kl_new": kn.mean().item(), "kl_floor": kf.mean().item(),
                     "d_mean": (kn - kc).mean().item(), "d_max": (kn - kc).max().item(),
                     "top1_cur": int(row["t1c"].sum()), "top1_new": int(row["t1n"].sum()), "top1_floor": int(row["t1f"].sum()),
                     "steps_with_equal_top32": int(row["same"].sum())})

out = {"cases": per_case, "groups": []}
print("| group | steps | KL cur | KL new | KL floor | d = new - cur: mean [95% interval] | floor's half-width | top-1 cur / new / floor | "
      "steps with byte-equal top-32 | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name in [g for g in ("chat", "long", "needle 257k", "all") if g in steps]:
    cat = lambda k: torch.cat([r[k] for r in steps[name]])
    kc, kn, kf = cat("kc"), cat("kn"), cat("kf")
    n = len(kc)
    dm, dlo, dhi = boot(kn - kc, a.boot)
    fm, flo, fhi = boot(kf, a.boot)
    half = (fhi - flo) / 2
    t1c, t1n, t1f = cat("t1c").mean().item(), cat("t1n").mean().item(), cat("t1f").mean().item()
    slack = max(0.01, 2 * math.sqrt(max(t1f * (1 - t1f), 1e-12) / n))
    # d: no measurable harm when its interval holds zero or lies below it; wholly above zero is not a pass
    # on its own and is reported for a decision. The band: as the decode gates', except in a group the
    # current route is already over it (judged by d alone and marked).
    cur_over = kc.mean().item() > 2 * fm
    ok_d, ok_band, ok_top = dlo <= 0, cur_over or kn.mean().item() <= 2 * fm, t1n >= t1f - slack
    verdict = "PASS" if ok_d and ok_band and ok_top else ("DECIDE: d above zero" if ok_band and ok_top else "FAIL: " + ", ".join(
        w for w, ok in (("d above zero", ok_d), ("KL band", ok_band), ("top-1", ok_top)) if not ok))
    if cur_over:
        verdict += " (current route over the band: by d alone)"
    frac = f"{100 * dlo / kc.mean().item():+.1f}% to {100 * dhi / kc.mean().item():+.1f}%"
    print(f"| {name} | {n} | {kc.mean().item():.3e} | {kn.mean().item():.3e} | {fm:.3e} | {dm:+.2e} [{dlo:+.2e}, {dhi:+.2e}] = {frac} of KL cur | "
          f"{half:.2e} | {100 * t1c:.1f}% / {100 * t1n:.1f}% / {100 * t1f:.1f}% | {int(cat('same').sum())} | {verdict} |")
    out["groups"].append({"group": name, "steps": n, "kl_cur": kc.mean().item(), "kl_new": kn.mean().item(), "kl_floor": fm,
                          "kl_floor_interval": [flo, fhi], "floor_half_width": half, "d_mean": dm, "d_interval": [dlo, dhi],
                          "top1_cur": t1c, "top1_new": t1n, "top1_floor": t1f, "top1_slack": slack,
                          "pass_d": ok_d, "pass_band": ok_band, "pass_top1": ok_top, "current_route_over_band": cur_over})


def routes(d):
    calls = raw(os.path.join(d, "routes_calls.i32"), torch.int32).long().view(-1, 2)
    ids = raw(os.path.join(d, "routes_ids.i32"), torch.int32).long().view(-1, a.topk)
    w = raw(os.path.join(d, "routes_w.f32"), torch.float32).double().view(-1, a.topk)
    m = raw(os.path.join(d, "routes_moe.bf16"), torch.int16)
    # bf16 is the high half of an fp32
    moe = (m.int() << 16).view(torch.float32).double().view(-1, a.dim)
    return calls, ids, w, moe


if a.routes_cur and a.routes_new:
    ca, ia, wa, ma = routes(a.routes_cur)
    cb, ib, wb, mb = routes(a.routes_new)
    assert torch.equal(ca, cb), "the two runs issued other blocks"
    layer = torch.repeat_interleave(ca[:, 0], ca[:, 1])
    one = torch.repeat_interleave(ca[:, 1] == 1, ca[:, 1])  # rows of one-row blocks: the decode steps
    assert len(layer) == len(ia) and int(one.sum()) == len(ma), (len(layer), len(ia), int(one.sum()), len(ma))
    # a row's experts as sets, and each set's least-weight member
    sa, sb = ia.sort(1).values, ib.sort(1).values
    same_set = (sa == sb).all(1)
    in_b = (ia[:, :, None] == ib[:, None, :]).any(2)  # a's experts that b also chose
    in_a = (ib[:, :, None] == ia[:, None, :]).any(2)
    lost, gained = (~in_b).sum(1), (~in_a).sum(1)
    low_a = wa.argmin(1, keepdim=True)
    low_b = wb.argmin(1, keepdim=True)
    # the sixth-against-seventh swap: one expert out, one in, and each is its row's least-weight choice
    sixth = (lost == 1) & (gained == 1) & (~in_b).gather(1, low_a).squeeze(1) & (~in_a).gather(1, low_b).squeeze(1)
    other = ~same_set & ~sixth
    # weights on the experts both chose
    eq = ia[:, :, None] == ib[:, None, :]
    wd = ((wa[:, :, None] - wb[:, None, :]).abs() * eq).amax((1, 2))
    rel = (ma - mb).norm(dim=1) / ma.norm(dim=1).clamp_min(1e-30)
    f1, o1 = sixth[one], other[one]
    q = lambda v: (f"{v.median().item():.2e} / {v.max().item():.2e}" if len(v) else "-")
    n = len(ia)
    print(f"\nrouter, the new order against the current route: {n} routed rows ({int(one.sum())} of them decode steps) x top-{a.topk}")
    print("| | rows | of all |")
    print("|---|---|---|")
    print(f"| the same six experts | {int(same_set.sum())} | {100 * same_set.double().mean().item():.4f}% |")
    print(f"| another sixth expert (its least-weight choice swapped) | {int(sixth.sum())} | {100 * sixth.double().mean().item():.4f}% |")
    print(f"| any other difference | {int(other.sum())} | {100 * other.double().mean().item():.4f}% |")
    print(f"largest weight difference on a shared expert: {wd.max().item():.3e} (median {wd.median().item():.3e})")
    print("the MoE output of a decode step, relative difference against the current route, median / max: "
          f"another sixth expert {q(rel[f1])} ({int(f1.sum())} steps); other differences {q(rel[o1])} ({int(o1.sum())}); "
          f"the same experts {q(rel[~f1 & ~o1])} ({int((~f1 & ~o1).sum())})")
    per_layer = []
    print("| layer | rows | another sixth | other |")
    print("|---|---|---|---|")
    for l in sorted(set(layer.tolist())):
        sel = layer == l
        per_layer.append({"layer": l, "rows": int(sel.sum()), "sixth": int(sixth[sel].sum()), "other": int(other[sel].sum())})
        if per_layer[-1]["sixth"] or per_layer[-1]["other"]:
            print(f"| {l} | {per_layer[-1]['rows']} | {per_layer[-1]['sixth']} | {per_layer[-1]['other']} |")
    out["router"] = {"rows": n, "decode_steps": int(one.sum()), "same": int(same_set.sum()), "sixth": int(sixth.sum()),
                     "other": int(other.sum()), "max_weight_diff": wd.max().item(), "per_layer": per_layer,
                     "moe_rel_sixth": rel[f1].tolist(), "moe_rel_same_median": rel[~f1 & ~o1].median().item() if (~f1 & ~o1).any() else None}
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
