#!/usr/bin/env python3
"""Draft-depth policies replayed over teacher-forced draft replays (hy4f-spec replay).

A replay holds, for every anchor i of a request's answer, how many of a K-draft round's drafts
equal the answer's next tokens from the front, for greedy drafts and for drafts drawn with the
request's chooser. The DFly2 block is causal, so a round of k drafts at i keeps min(m[i], k).
This walks the serve route's loop (hy4f_serve: a round after every call, k capped by the room
left in the budget, a plain call when k is 0), with a policy choosing k, and prices each step
with a cost table (hy4f-spec cost: verify_ms_by_rows, draft_ms_by_drafts).

usage: sim.py REPLAY_JSONL COST_JSON [--mode chosen|greedy] [--budget N] [--json OUT]
"""
import json
import math
import sys

args = sys.argv[1:]


def opt(name, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


mode = opt("--mode", "chosen")
budget_override = opt("--budget", None)
out_json = opt("--json", None)
replay_path, cost_path = args[0], args[1]
reqs = [json.loads(x) for x in open(replay_path) if x.strip()]
cost = json.load(open(cost_path))
V = cost["verify_ms_by_rows"]  # index r: r + 1 rows
D = cost["draft_ms_by_drafts"]  # index d: d + 1 drafts
K = len(D)


def step_ms(k):
    return V[0] if k == 0 else V[k] + D[k - 1]


class Fixed:
    def __init__(self, k):
        self.k = k
        self.name = f"fixed{k}"

    def depth(self, cap):
        return min(self.k, cap)

    def observe(self, k, m):
        pass


class AdaptLast:
    """the house's adaptive depth: next round drafts the last round's kept + 1"""

    def __init__(self, floor=1):
        self.k = K
        self.floor = floor
        self.name = f"last+1(floor{floor})"

    def depth(self, cap):
        return min(self.k, cap)

    def observe(self, k, m):
        self.k = max(self.floor, min(K, m + 1))


class DraftGate:
    """memra_engine::dsv41_gpu::draft::DraftGate"""

    def __init__(self):
        self.gain, self.pause, self.stretch = 1.0, 0, 8

    def drafts(self):
        if self.pause > 0:
            self.pause -= 1
            return False
        return True

    def observe(self, tokens, step, plain):
        g = tokens * plain / step
        self.gain += 0.25 * (g - self.gain)
        if self.gain < 1.0:
            self.pause = self.stretch
            self.stretch = min(self.stretch * 2, 64)
            self.gain = 1.0
        elif self.gain > 1.25:
            self.stretch = 8


class Survival:
    """per-depth survival EWMA priced by expected tokens per ms (gate v1 when alpha 0.25, init 1,
    with the stop gate)"""

    def __init__(self, alpha=0.25, init=1.0, stop=True, kmin=1, geometric=False):
        self.a, self.init, self.surv = alpha, init, [init] * K
        self.stop = DraftGate() if stop else None
        self.paused = False
        self.kmin = kmin
        self.geo = geometric
        self.h, self.n = 0.0, 0.0
        self.name = f"surv(a={alpha},init={init},stop={stop},kmin={kmin},geo={geometric})"

    def exp_tokens(self, k):
        alive, t = 1.0, 1.0
        for j in range(k):
            if self.geo:
                s = self.init if self.n == 0 else self.h / self.n
            else:
                s = self.surv[j]
            alive *= s
            t += alive
        return t

    def depth(self, cap):
        if cap <= 0:
            return 0
        if self.stop and not self.stop.drafts():
            self.paused = True
            return 0
        if self.paused:
            self.paused = False
            self.surv = [self.init] * K
            self.h = self.n = 0.0
        best = (None, -1)
        for k in range(self.kmin, min(K, cap) + 1):
            r = self.exp_tokens(k) / step_ms(k)
            if r > best[1]:
                best = (k, r)
        return best[0] if best[0] is not None else 0

    def observe(self, k, m):
        for j in range(k):
            held = 1.0 if j < m else 0.0
            self.surv[j] += self.a * (held - self.surv[j])
            if self.geo:
                self.h = self.h * (1 - self.a) + held
                self.n = self.n * (1 - self.a) + 1.0
            if j >= m:
                break
        if self.stop:
            self.stop.observe(m + 1, step_ms(k), step_ms(0))


class Price:
    """one conditional survival s for every depth (a running average of every observed draft
    outcome at the depths reached), each round's depth priced by expected tokens per ms; k = 0
    (a one-token call) when no depth beats it: then a stretch of plain steps (8, doubling to 64
    while it repeats), then a priced round from s reset to its start"""

    def __init__(self, alpha=0.1, init=0.7, allow0=True, per_depth=False, prior=None, w=0.0):
        self.a, self.init, self.allow0 = alpha, init, allow0
        self.per_depth = per_depth
        self.prior = prior or [init] * K
        self.w = w
        self.h = [0.0] * K
        self.n = [0.0] * K
        self.s = init
        self.pause, self.stretch, self.paused = 0, 8, False
        self.name = (f"price(a={alpha},init={init},allow0={allow0},per_depth={per_depth},w={w}"
                     + (",prior" if prior else "") + ")")

    def surv(self, j):
        if not self.per_depth:
            return self.s
        return (self.h[j] + self.w * self.prior[j]) / (self.n[j] + self.w) if self.n[j] + self.w > 0 else self.prior[j]

    def exp_tokens(self, k):
        alive, t = 1.0, 1.0
        for j in range(k):
            alive *= self.surv(j)
            t += alive
        return t

    def depth(self, cap):
        if cap <= 0:
            return 0
        if self.pause > 0:
            self.pause -= 1
            if self.pause == 0:
                self.s = self.init
                self.h = [0.0] * K
                self.n = [0.0] * K
            return 0
        best = (0, 1.0 / step_ms(0)) if self.allow0 else (None, -1)
        for k in range(1, min(K, cap) + 1):
            r = self.exp_tokens(k) / step_ms(k)
            if r > best[1]:
                best = (k, r)
        if best[0] == 0:
            self.pause = self.stretch
            self.stretch = min(2 * self.stretch, 64)
            return 0
        self.stretch = 8
        return best[0]

    def observe(self, k, m):
        for j in range(k):
            held = 1.0 if j < m else 0.0
            self.s += self.a * (held - self.s)
            self.h[j] = self.h[j] * (1 - self.a) + held
            self.n[j] = self.n[j] * (1 - self.a) + 1.0
            if j >= m:
                break


class StopOnly:
    """full rounds, the stop gate only"""

    def __init__(self):
        self.stop = DraftGate()
        self.name = "fixed5+stop"

    def depth(self, cap):
        if not self.stop.drafts():
            return 0
        return min(K, cap)

    def observe(self, k, m):
        self.stop.observe(m + 1, step_ms(k), step_ms(0))


def walk(req, policy, mode):
    ids = req["ids"]
    m = req["m_" + mode]
    eos = req.get("eos")
    budget = int(budget_override or req["n"])
    # ids[0] is emitted by the prefill's last call
    e = 1
    t = 0.0
    rounds = drafted = accepted = 0
    plain = 0
    while True:
        if eos is not None and ids[e - 1] == eos:
            break
        if e >= budget or e >= len(ids):
            break
        room = budget - e
        k = policy.depth(room - 1)
        k = max(0, min(k, room - 1))
        if k == 0:
            t += step_ms(0)
            plain += 1
            e += 1
            continue
        a = e - 1
        got = min(m[a], k)
        # tokens the round emits: the matched drafts and the route's own next token, stopped at
        # the end token or the budget
        n_tok = got + 1
        new = ids[e:e + n_tok]
        if eos is not None and eos in new:
            n_tok = new.index(eos) + 1
        n_tok = min(n_tok, budget - e)
        t += step_ms(k)
        rounds += 1
        drafted += k
        accepted += got
        policy.observe(k, got)
        e += n_tok
    return {"tokens": e, "ms": t, "rounds": rounds, "drafted": drafted, "accepted": accepted, "plain": plain}


def oracle(req, mode):
    """the best depth at every step (dynamic programming over the anchor): the bound"""
    ids, m = req["ids"], req["m_" + mode]
    eos = req.get("eos")
    budget = int(budget_override or req["n"])
    end = len(ids)
    if eos is not None and eos in ids:
        end = ids.index(eos) + 1
    end = min(end, budget)
    best = [math.inf] * (end + 1)
    best[end] = 0.0
    for e in range(end - 1, 0, -1):
        room = budget - e
        c = step_ms(0) + best[e + 1]
        for k in range(1, min(K, room - 1) + 1):
            n_tok = min(m[e - 1], k) + 1
            c = min(c, step_ms(k) + best[min(end, e + n_tok)])
        best[e] = c
    return {"tokens": end, "ms": best[1]}


policies = [lambda: Fixed(0)] + [lambda k=k: Fixed(k) for k in range(1, K + 1)] + [
    lambda: StopOnly(),
    lambda: AdaptLast(1),
    lambda: Survival(),
    lambda: Survival(stop=False),
    lambda: Survival(alpha=0.1),
    lambda: Survival(alpha=0.1, stop=False),
    lambda: Survival(alpha=0.25, geometric=True),
    lambda: Survival(alpha=0.1, geometric=True),
    lambda: Survival(alpha=0.05, geometric=True),
]
def pooled(skip):
    reach, hold = [0] * K, [0] * K
    for r in reqs:
        if r["name"] == skip:
            continue
        for x in r["m_" + mode]:
            for j in range(K):
                reach[j] += x >= j
                hold[j] += x >= j + 1
    return [hold[j] / reach[j] for j in range(K)]


policies += [
    lambda: Price(0.1, 0.7),
    lambda: Price(0.05, 0.7),
    lambda: Price(0.25, 0.7),
    lambda: Price(0.1, 1.0),
    lambda: Price(0.1, 0.7, allow0=False),
]
res = {}
print(f"mode {mode}; cost table verify {[round(x, 1) for x in V]} draft {[round(x, 1) for x in D]}")
for mk in policies:
    name = mk().name
    rows = []
    for r in reqs:
        rows.append({"name": r["name"], **walk(r, mk(), mode)})
    res[name] = rows
for a in (0.05, 0.1, 0.25):
    for w in (2.0, 8.0):
        rows = []
        for r in reqs:
            pol = Price(a, 0.7, per_depth=True, prior=pooled(r["name"]), w=w)
            rows.append({"name": r["name"], **walk(r, pol, mode)})
        res[pol.name] = rows
orc = [{"name": r["name"], **oracle(r, mode)} for r in reqs]
res["oracle"] = orc
names = [r["name"] for r in reqs]
print("| policy | " + " | ".join(names) + " | all (tok/s) |")
print("|---|" + "---:|" * (len(names) + 1))
for p, rows in res.items():
    cells = [f"{1000 * (x['tokens'] - 1) / x['ms']:.2f}" for x in rows]
    tot = 1000 * sum(x["tokens"] - 1 for x in rows) / sum(x["ms"] for x in rows)
    print(f"| {p} | " + " | ".join(cells) + f" | {tot:.2f} |")
print()
for p in ("fixed5", "surv(a=0.25,init=1.0,stop=True,kmin=1,geo=False)"):
    for x in res[p]:
        print(p, x["name"], {k: x[k] for k in ("tokens", "rounds", "drafted", "accepted", "plain")})
if out_json:
    json.dump(res, open(out_json, "w"), indent=1)
