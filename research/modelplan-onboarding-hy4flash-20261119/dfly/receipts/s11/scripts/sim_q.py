#!/usr/bin/env python3
"""Draft depth priced from the drafter's own probabilities, replayed over a teacher-forced draft
replay with `q_greedy` (hy4f-spec replay): at anchor i the round's drafts are drawn one step at
a time; after each step the survival of draft j is the product of the drafter's probabilities of
drafts 1..j (at the request's temperature), and the round verifies the depth with the most
expected tokens per ms of its step (the draft steps it ran plus the verify call of the anchor and
the depth's drafts). The round stops drafting once one more draft that surely held could not beat
the best depth so far. A round whose best depth is 0 is a one-token call that paid one draft step.

usage: sim_q.py REPLAY_JSONL COST_JSON [--calib] [--json OUT]
"""
import json
import sys

args = sys.argv[1:]
reqs = [json.loads(x) for x in open(args[0]) if x.strip()]
cost = json.load(open(args[1]))
V, D = cost["verify_ms_by_rows"], cost["draft_ms_by_drafts"]
K = len(D)


def vms(rows):
    return V[rows - 1]


def dms(steps):
    return 0.0 if steps == 0 else D[steps - 1]


def calibration():
    """hold rate of a draft by the drafter's probability, given the drafts before it held"""
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
    n = [0] * (len(bins) - 1)
    h = [0] * (len(bins) - 1)
    for r in reqs:
        for a, (m, q) in enumerate(zip(r["m_greedy"], r["q_greedy"])):
            for j in range(min(len(q), m + 1)):
                b = next(i for i in range(len(bins) - 1) if q[j] < bins[i + 1])
                n[b] += 1
                h[b] += j < m
    return bins, n, h


def round_qp(q, cap, early, scale=1.0, qmap=None):
    """(draft steps, verify depth) for one round"""
    best = (0, 1.0 / vms(1), 0)  # depth, rate, steps
    alive, exp_t = 1.0, 1.0
    steps = 0
    for j in range(min(K, cap)):
        steps = j + 1
        alive *= min(1.0, (qmap(q[j]) if qmap else q[j]) * scale)
        exp_t += alive
        rate = exp_t / (dms(steps) + vms(j + 2))
        # every depth up to j costs the steps run so far
        cands = [(0, 1.0 / (dms(steps) + vms(1)))]
        e, a2 = 1.0, 1.0
        for kk in range(1, steps + 1):
            a2 *= min(1.0, (qmap(q[kk - 1]) if qmap else q[kk - 1]) * scale)
            e += a2
            cands.append((kk, e / (dms(steps) + vms(kk + 1))))
        bk, br = max(cands, key=lambda c: (c[1], -c[0]))
        best = (bk, br, steps)
        if early:
            # one more draft that surely held
            if j + 1 >= min(K, cap):
                break
            opt = (exp_t + alive) / (dms(steps + 1) + vms(j + 3))
            if opt <= br:
                break
    return best[2], best[0]


def walk(r, early=True, pause_on_zero=True, scale=1.0, qmap=None, online=None):
    ids, m, qq, eos, budget = r["ids"], r["m_greedy"], r["q_greedy"], r.get("eos"), r["n"]
    e, t = 1, 0.0
    rounds = drafted = accepted = plain = zero = 0
    pause, stretch = 0, 8
    while True:
        if eos is not None and ids[e - 1] == eos:
            break
        if e >= budget or e >= len(ids):
            break
        room = budget - e
        a = e - 1
        if pause > 0 or room - 1 <= 0:
            pause = max(0, pause - 1)
            t += vms(1)
            plain += 1
            e += 1
            continue
        sc = scale
        if online is not None:
            sc = (online["h"] + online["w"] * online["init"]) / (online["p"] + online["w"])
        steps, k = round_qp(qq[a], room - 1, early, sc, qmap)
        t += dms(steps)
        if k == 0:
            t += vms(1)
            e += 1
            zero += 1
            if pause_on_zero:
                pause, stretch = stretch, min(2 * stretch, 64)
            continue
        stretch = 8
        got = min(m[a], k)
        if online is not None:
            # each verified draft up to the first that failed: its probability against its outcome
            for j in range(min(k, got + 1)):
                qj = qmap(qq[a][j]) if qmap else qq[a][j]
                online["p"] = online["p"] * (1 - online["a"]) + qj
                online["h"] = online["h"] * (1 - online["a"]) + (1.0 if j < got else 0.0)
        n_tok = got + 1
        new = ids[e:e + n_tok]
        if eos is not None and eos in new:
            n_tok = new.index(eos) + 1
        n_tok = min(n_tok, budget - e)
        t += vms(k + 1)
        rounds += 1
        drafted += k
        accepted += got
        e += n_tok
    return {"tokens": e, "ms": t, "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "plain": plain, "zero": zero}


def fixed(r, k):
    ids, m, eos, budget = r["ids"], r["m_greedy"], r.get("eos"), r["n"]
    e, t = 1, 0.0
    while True:
        if eos is not None and ids[e - 1] == eos:
            break
        if e >= budget or e >= len(ids):
            break
        kk = min(k, budget - e - 1)
        if kk <= 0:
            t += vms(1)
            e += 1
            continue
        got = min(m[e - 1], kk)
        n_tok = got + 1
        new = ids[e:e + n_tok]
        if eos is not None and eos in new:
            n_tok = new.index(eos) + 1
        t += dms(kk) + vms(kk + 1)
        e += min(n_tok, budget - e)
    return {"tokens": e, "ms": t}


bins, n, h = calibration()
print("calibration (q bin: n, hold rate):", [(f"{bins[i]:.2f}-{bins[i+1]:.2f}", n[i], round(h[i] / n[i], 3) if n[i] else None) for i in range(len(n))])
res = {}
for k in (3, 5):
    res[f"fixed{k}"] = [{"name": r["name"], **fixed(r, k)} for r in reqs]
for early in (False, True):
    for pz in (False, True):
        for sc in (1.0, 0.9, 1.1):
            res[f"qp(early={early},pause0={pz},scale={sc})"] = [{"name": r["name"], **walk(r, early, pz, sc)} for r in reqs]
def loo_map(skip):
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
    n = [0] * (len(bins) - 1)
    h = [0] * (len(bins) - 1)
    for r in reqs:
        if r["name"] == skip:
            continue
        for a, (m, q) in enumerate(zip(r["m_greedy"], r["q_greedy"])):
            for j in range(min(len(q), m + 1)):
                b = next(i for i in range(len(bins) - 1) if q[j] < bins[i + 1])
                n[b] += 1
                h[b] += j < m
    rate = [h[i] / n[i] if n[i] else (bins[i] + bins[i + 1]) / 2 for i in range(len(n))]
    return lambda q: rate[next(i for i in range(len(bins) - 1) if q < bins[i + 1])]


res["qp(calib loo)"] = [{"name": r["name"], **walk(r, True, True, 1.0, loo_map(r["name"]))} for r in reqs]
for a in (0.05, 0.1, 0.25):
    for w in (2.0, 8.0):
        res[f"qp(online a={a} w={w})"] = [{"name": r["name"], **walk(r, True, True, 1.0, None,
                                           {"h": 0.0, "p": 0.0, "w": w, "init": 1.0, "a": a})} for r in reqs]
names = [r["name"] for r in reqs]
print("| policy | " + " | ".join(names) + " | all (tok/s) |")
print("|---|" + "---:|" * (len(names) + 1))
for p, rows in res.items():
    cells = [f"{1000 * (x['tokens'] - 1) / x['ms']:.2f}" for x in rows]
    tot = 1000 * sum(x["tokens"] - 1 for x in rows) / sum(x["ms"] for x in rows)
    print(f"| {p} | " + " | ".join(cells) + f" | {tot:.2f} |")
for x in res["qp(early=True,pause0=False,scale=1.0)"]:
    print(x)
if "--json" in args:
    json.dump(res, open(args[args.index("--json") + 1], "w"), indent=1)
