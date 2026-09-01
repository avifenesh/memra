#!/usr/bin/env python3
"""How big is the routed UNION under routers that are NOT uniform? (moeu lane, mtp13)

This script exists to refute a tempting shortcut rather than to support it. The closed-form
union for INDEPENDENT top-k routing,

    E[union] = N * (1 - (1 - k/N)^t)

gives 57.0 of 60 at qwen4_exp's geometry (N=512 experts, k=10 selected, t=6 verify columns),
i.e. only ~5% of the chunk's expert-byte traffic is duplicated. It is very tempting to stop
there and declare a union-of-experts gather dead on "512 experts top-10 cannot collide".

Do not. Real routers have hotness and temporal correlation, and the union is SENSITIVE to
both. This sweep prices that sensitivity, and the answer is that a moderately hot or
moderately sticky router puts the traffic prize at 27-35%, not 5%. The union lever still
dies -- but on the measured payoff curve (MOEUNION.md section 4), not on this arithmetic.

Reported per router model: mean union of t*k slots, union/pairs, and prize = 1 - ratio.

Usage:  union-sensitivity.py [--experts N] [--selected K] [--t T] [--trials M]
"""
import random
import sys


def mean_union(sampler, experts, selected, t, trials):
    tot = 0
    for _ in range(trials):
        ids = []
        for _ in range(t):
            ids += sampler()
        tot += len(set(ids))
    return tot / trials


def weighted_topk(weights, experts, selected):
    """Weighted sample WITHOUT replacement: top-k routing never repeats within a token."""
    pool = list(range(experts))
    ww = list(weights)
    out = []
    for _ in range(selected):
        total = sum(ww)
        x = random.random() * total
        c = 0.0
        i = 0
        for i, v in enumerate(ww):
            c += v
            if c >= x:
                break
        out.append(pool[i])
        pool.pop(i)
        ww.pop(i)
    return out


def main(argv):
    experts, selected, t, trials = 512, 10, 6, 2000
    for i, a in enumerate(argv):
        if a == "--experts":
            experts = int(argv[i + 1])
        elif a == "--selected":
            selected = int(argv[i + 1])
        elif a == "--t":
            t = int(argv[i + 1])
        elif a == "--trials":
            trials = int(argv[i + 1])
        elif a in ("-h", "--help"):
            print(__doc__)
            return 2

    random.seed(11)
    pairs = t * selected
    closed = experts * (1 - (1 - selected / experts) ** t)
    print(f"# experts={experts} selected={selected} t={t} pairs={pairs} trials={trials}")
    print(f"# closed-form independent union = {closed:.2f} of {pairs} "
          f"(ratio {closed / pairs:.4f}, prize {1 - closed / pairs:.4f})")
    print(f"{'router model':<48s}\tunion\tratio\tprize")

    def row(label, sampler, n):
        u = mean_union(sampler, experts, selected, t, n)
        print(f"{label:<48s}\t{u:.2f}\t{u / pairs:.4f}\t{1 - u / pairs:.4f}")

    row("independent uniform top-k",
        lambda: random.sample(range(experts), selected), trials)

    for a in (0.5, 1.0, 1.5, 2.0):
        w = [1.0 / ((r + 1) ** a) for r in range(experts)]
        row(f"Zipf hotness a={a}",
            lambda w=w: weighted_topk(w, experts, selected), max(trials // 4, 200))

    for h in (0.2, 0.4, 0.6, 0.8):
        state = {}

        def sticky(h=h, state=state):
            keep = int(round(selected * h))
            prev = state.get("p")
            if prev is None:
                cur = random.sample(range(experts), selected)
            else:
                cur = random.sample(prev, keep)
                while len(cur) < selected:
                    c = random.randrange(experts)
                    if c not in cur:
                        cur.append(c)
            state["p"] = cur
            return list(cur)

        row(f"temporal: re-pick {int(h * 100)}% of previous column", sticky, trials)

    print()
    print("# The union lever's reopen bar in MOEUNION.md is union/pairs <= 0.70, i.e. "
          f"union <= {0.70 * pairs:.1f} of {pairs}.")
    print("# Reachable by a Zipf a=1.0-class or 40%-sticky router -- which is exactly why the")
    print("# verdict rests on the measured payoff curve and not on this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
