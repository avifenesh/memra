"""The decode gates: memra's decode loop against the vendor's greedy tapes, judged on the vendor's own decode floor.

Inputs per case ID:
  TAPES/ID.pt     the vendor's greedy tape (prompt, gen_ids, per-step top-32, full logits of the first steps)
  REF/ID.pt       the same oracle teacher-forced with the tape's tokens (make_forced_oracle.py): the tape again, plus
                  every step's lse and argmax. Checked against the tape: same top-32 ids and values, step for step.
  FLOOR/ID.pt     the global-max attention oracle teacher-forced with the same tokens: another valid realization of
                  the vendor's decode program at the same steps (the schedule floor of the prefill gates)
  NATIVE/ID/      `dsv41-run --cases ... ID PROMPT GEN`: memra's decode loop teacher-forced with the same tokens
  GREEDY/ID/      `dsv41-run --cases ... ID PROMPT --gen N`: memra's free-running greedy generation (optional)

Teacher-forced, per step: KL(vendor || other) from both sides' top-32 plus one rest bucket (a lower bound of the full
KL, computed the same way for memra and for the floor; tf_kl.py), the exact KL on the steps with full logits, and top-1
agreement with the vendor's greedy token. A group of cases passes on the floor's band, the rule of the prefill gates:
mean KL at most twice the floor's, top-1 at most 1 point under the floor's (or two binomial standard errors when the
group is small).

Greedy, per case: the exact-match prefix with the vendor's tokens, the step where they part, and whether that step is a
vendor near-tie. The vendor's top-2 margin there (logit of its token minus logit of the runner-up) is set against how
far that same margin moves between the vendor's own two realizations: at every step of the group, the margin of REF's
top-2 ids on REF minus the margin of the same two ids on FLOOR. A divergence is a near-tie when another valid run of the
vendor program picks memra's token there (FLOOR, or the CPU run of CPU/ID.pt when it covers the step), or when the
vendor's margin there is no larger than the p99 of the margin moves between the two vendor realizations in the group.
The prefix steps of the greedy run are also compared byte for byte with the teacher-forced run (same tokens fed, so
the same program: a separate process and another batch of sequences).

usage: decode_gate.py --tapes D --ref D --floor D --native D [--greedy D] [--cpu D] --cases a,b,... --name NAME
                      [--tok TOKENIZER_DIR] [--json OUT]"""
import argparse
import json
import math
import os

import torch

ap = argparse.ArgumentParser()
for k in ("tapes", "ref", "floor", "native", "cases", "name"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--greedy")
ap.add_argument("--cpu", help="the vendor's CPU run, teacher-forced (first steps)")
ap.add_argument("--json")
ap.add_argument("--tok", help="tokenizer dir (decodes the divergence context)")
a = ap.parse_args()
tok = None
if a.tok:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)


def raw(path, dt):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt)


def native(d):
    out = {"top_ids": raw(os.path.join(d, "top_ids.i32"), torch.int32).long().view(-1, 32),
           "top_vals": raw(os.path.join(d, "top_vals.f32"), torch.float32).double().view(-1, 32),
           "lse": raw(os.path.join(d, "lse.f64"), torch.float64),
           "argmax": raw(os.path.join(d, "argmax.i32"), torch.int32).long(),
           "raw": {f: open(os.path.join(d, f), "rb").read() for f in ("top_ids.i32", "top_vals.f32", "lse.f64", "argmax.i32")}}
    fp = os.path.join(d, "full.f32")
    if os.path.exists(fp) and os.path.getsize(fp):
        out["full"] = raw(fp, torch.float32).double().view(-1, 129280)
    gp = os.path.join(d, "gen.i32")
    if os.path.exists(gp):
        out["gen"] = raw(gp, torch.int32).long()
    return out


def kl_top32(pi, pv, pl, qi, qv, ql):
    """Per-row KL(p || q) over the ids both top-32 sets hold, plus one rest bucket each (tf_kl.py)."""
    p = (pv - pl[:, None]).exp()
    q = (qv - ql[:, None]).exp()
    eq = pi[:, :, None] == qi[:, None, :]
    both = eq.any(2)
    qs = (eq * q[:, None, :]).sum(2)
    ps = p * both
    pr = (1 - ps.sum(1)).clamp_min(1e-12)
    qr = (1 - qs.sum(1)).clamp_min(1e-12)
    term = torch.where(both, ps * (ps.clamp_min(1e-300).log() - qs.clamp_min(1e-300).log()), torch.zeros_like(ps))
    prest = (1 - p.sum(1)).clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-300).log()).sum(1) - prest * prest.log()
    return term.sum(1) + pr * (pr.log() - qr.log()), ent


def kl_full(p, q):
    lp, lq = p.log_softmax(-1), q.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def spread(ri, rv, fi, fv):
    """|logit difference| between two realizations on the ids both top-32 rows hold."""
    eq = ri[:, :, None] == fi[:, None, :]
    d = (rv[:, :, None] - fv[:, None, :]).abs()
    return d[eq]


def margin_moves(ri, rv, fi, fv):
    """Per step: REF's top-2 margin minus the same two ids' margin on FLOOR (steps where FLOOR's top-32 holds both)."""
    out = []
    for k in range(len(ri)):
        a, b = (fi[k] == ri[k, 0]).nonzero(), (fi[k] == ri[k, 1]).nonzero()
        if len(a) and len(b):
            out.append((rv[k, 0] - rv[k, 1]).item() - (fv[k, int(a[0])] - fv[k, int(b[0])]).item())
    return out


cases = a.cases.split(",")
rows, per_case, spreads, greedy, moves = [], [], [], [], []
for c in cases:
    T = torch.load(os.path.join(a.tapes, f"{c}.pt"), weights_only=False, map_location="cpu")
    R = torch.load(os.path.join(a.ref, f"{c}.pt"), weights_only=False, map_location="cpu")
    F = torch.load(os.path.join(a.floor, f"{c}.pt"), weights_only=False, map_location="cpu")
    N = native(os.path.join(a.native, c))
    gen = torch.tensor([int(x) for x in T["gen_ids"]]).long()
    G = len(gen)
    assert len(R["gen_ids"]) == G and len(F["gen_ids"]) == G and len(N["argmax"]) == G, (c, G, len(N["argmax"]))
    ref_same = bool(torch.equal(R["top_ids"].long(), T["top_ids"].long()) and torch.equal(R["top_vals"], T["top_vals"]))
    ri, rv, rl = R["top_ids"].long(), R["top_vals"].double(), R["lse"].double()
    fi, fv, fl = F["top_ids"].long(), F["top_vals"].double(), F["lse"].double()
    kn, ent = kl_top32(ri, rv, rl, N["top_ids"], N["top_vals"], N["lse"])
    kf, _ = kl_top32(ri, rv, rl, fi, fv, fl)
    t1n = (N["argmax"] == gen).double()
    t1f = (F["argmax"].long() == gen).double()
    nf = min(len(T["full_logits"]), len(N.get("full", [])), len(F["full_logits"]))
    fkn = kl_full(T["full_logits"][:nf].double(), N["full"][:nf]) if nf else torch.zeros(0)
    fkf = kl_full(T["full_logits"][:nf].double(), F["full_logits"][:nf].double()) if nf else torch.zeros(0)
    sp = spread(ri, rv, fi, fv)
    spreads.append(sp)
    moves += margin_moves(ri, rv, fi, fv)
    rows.append({"kl_native": kn, "kl_floor": kf, "ent": ent, "t1n": t1n, "t1f": t1f, "fkn": fkn, "fkf": fkf})
    per_case.append({"case": c, "prompt": len(T["prompt_ids"]), "steps": G, "ref_reproduces_tape": ref_same,
                     "kl_native_mean": kn.mean().item(), "kl_floor_mean": kf.mean().item(),
                     "kl_native_max": kn.max().item(), "kl_floor_max": kf.max().item(),
                     "top1_native": int(t1n.sum()), "top1_floor": int(t1f.sum()),
                     "full_kl_steps": nf, "full_kl_native_mean": fkn.mean().item() if nf else None,
                     "full_kl_floor_mean": fkf.mean().item() if nf else None,
                     "first_native_miss": int((t1n == 0).nonzero()[0]) if (t1n == 0).any() else None,
                     "first_floor_miss": int((t1f == 0).nonzero()[0]) if (t1f == 0).any() else None})
    if a.greedy and os.path.isdir(os.path.join(a.greedy, c)):
        Q = native(os.path.join(a.greedy, c))
        qg = Q["gen"]
        L = 0
        while L < min(len(qg), G) and int(qg[L]) == int(gen[L]):
            L += 1
        # steps 0..L were fed the same tokens in both native runs: the same program, byte for byte
        same = min(L + 1, len(qg), G)
        ident = all(Q["raw"][f][: same * w] == N["raw"][f][: same * w]
                    for f, w in (("top_ids.i32", 128), ("top_vals.f32", 128), ("lse.f64", 8), ("argmax.i32", 4)))
        g = {"case": c, "vendor_steps": G, "native_steps": len(qg), "exact_prefix": L,
             "all_equal": L == G == len(qg), "forced_vs_greedy_identical_steps": same if ident else 0,
             "forced_vs_greedy_identical": ident}
        if L < min(len(qg), G):
            k = L
            v1, v2 = rv[k, 0].item(), rv[k, 1].item()
            nat = int(qg[k])
            pos = (ri[k] == nat).nonzero()
            rank = int(pos[0]) + 1 if len(pos) else None
            fpick = int(F["argmax"][k])
            cpick = None
            cp = os.path.join(a.cpu, f"{c}.pt") if a.cpu else None
            if cp and os.path.exists(cp):
                CP = torch.load(cp, weights_only=False, map_location="cpu")
                if k < len(CP["argmax"]):
                    cpick = int(CP["argmax"][k])
            qv = Q["top_vals"][k]
            g.update({"diverge_step": k, "vendor_token": int(gen[k]), "native_token": nat,
                      "native_token_vendor_rank": rank, "vendor_margin_logit": v1 - v2,
                      "vendor_p1": math.exp(v1 - rl[k].item()), "vendor_p2": math.exp(v2 - rl[k].item()),
                      "native_margin_logit": (qv[0] - qv[1]).item(), "floor_token": fpick,
                      "floor_picks_native": fpick == nat, "floor_matches_vendor": fpick == int(gen[k]),
                      "cpu_token": cpick, "cpu_picks_native": cpick == nat if cpick is not None else None})
            if tok is not None:
                g["context"] = tok.decode(gen[max(0, k - 12):k].tolist())
                g["vendor_next"] = tok.decode([int(gen[k])])
                g["native_next"] = tok.decode([nat])
        if tok is not None:
            g["native_text"] = tok.decode(qg.tolist())
            g["vendor_text"] = tok.decode(gen.tolist())
        greedy.append(g)

kn = torch.cat([r["kl_native"] for r in rows])
kf = torch.cat([r["kl_floor"] for r in rows])
ent = torch.cat([r["ent"] for r in rows])
t1n = torch.cat([r["t1n"] for r in rows])
t1f = torch.cat([r["t1f"] for r in rows])
fkn = torch.cat([r["fkn"] for r in rows])
fkf = torch.cat([r["fkf"] for r in rows])
sp = torch.cat(spreads)
sp99 = sp.quantile(0.99).item() if len(sp) > 0 else 0.0
mv = torch.tensor(moves, dtype=torch.float64).abs()
mv_stats = {"steps": len(mv), "p50": mv.median().item(), "p99": mv.quantile(0.99).item(), "max": mv.max().item()} if len(mv) else {}


def subset(m):
    n = int(m.sum())
    if n == 0:
        return {"steps": 0}
    pn, pf = t1n[m].mean().item(), t1f[m].mean().item()
    se = math.sqrt(max(pf * (1 - pf), 1e-12) / n)
    slack = max(0.01, 2 * se)
    out = {"steps": n, "kl_native_mean": kn[m].mean().item(), "kl_floor_mean": kf[m].mean().item(),
           "kl_ratio": kn[m].mean().item() / max(kf[m].mean().item(), 1e-300),
           "kl_native_p99": kn[m].quantile(0.99).item() if n > 1 else kn[m].item(),
           "kl_floor_p99": kf[m].quantile(0.99).item() if n > 1 else kf[m].item(),
           "top1_native": pn, "top1_floor": pf, "top1_slack": slack}
    out["pass_kl"] = out["kl_native_mean"] <= 2 * out["kl_floor_mean"]
    out["pass_top1"] = pn >= pf - slack
    out["pass"] = out["pass_kl"] and out["pass_top1"]
    return out


steps_all = torch.ones(len(kn), dtype=torch.bool)
res = {"name": a.name, "cases": cases, "per_case": per_case,
       "pooled": {"all": subset(steps_all), "entropy_gt_1nat": subset(ent > 1.0)},
       "full_logit_steps": {"steps": len(fkn), "kl_native_mean": fkn.mean().item() if len(fkn) else None,
                            "kl_floor_mean": fkf.mean().item() if len(fkf) else None,
                            "kl_native_max": fkn.max().item() if len(fkn) else None,
                            "kl_floor_max": fkf.max().item() if len(fkf) else None},
       "vendor_spread_p99_logit": sp99, "vendor_margin_move": mv_stats, "greedy": greedy}
for g in greedy:
    if "diverge_step" in g:
        g["near_tie"] = bool(g["floor_picks_native"] or g["cpu_picks_native"]
                             or (mv_stats and g["vendor_margin_logit"] <= mv_stats["p99"]))
print(json.dumps({k: v for k, v in res.items() if k != "per_case"}, indent=1))
for p in per_case:
    print(json.dumps(p))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
