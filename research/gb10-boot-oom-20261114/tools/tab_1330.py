#!/usr/bin/env python3
"""tab_1330.py RUN_DIR [RUN_DIR...]: memra#1330's read-out of recovery cells (recovery_cell.sh).
Per arm and stratum: creations, first tries out of memory, and for those the time from the first
try to the context (ms), the driver calls it took, the time spent inside failing calls and in
waits; median, p90, p99 (nearest rank) and max; a two-sided Mann-Whitney test of the arms' times
and a bootstrap interval of the difference of medians. Strata: `first` (the round's first
creation, armed on MemFree), `later` (the round's other creations), `holder` (the holder's own
creation, made right before a round). Standard library only."""
import json
import math
import random
import sys


def load(dirs):
    rows = []
    for d in dirs:
        for line in open(f"{d}/creations.jsonl"):
            r = json.loads(line)
            if "skipped" in r:
                continue
            r["stratum"] = "first" if r.get("idx") == 0 else "later"
            r["cell"] = d.rstrip("/").split("/")[-1]
            rows.append(r)
        try:
            for line in open(f"{d}/holders.jsonl"):
                h = json.loads(line)
                rec = h.get("rec")
                if rec:
                    rec["stratum"] = "holder"
                    rec["round"] = h["round"]
                    rec["cell"] = d.rstrip("/").split("/")[-1]
                    rows.append(rec)
        except FileNotFoundError:
            pass
    return rows


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def summ(xs):
    if not xs:
        return "n=0"
    return (f"n={len(xs)} median={pct(xs, .5):g} p90={pct(xs, .9):g} p99={pct(xs, .99):g} "
            f"max={max(xs):g} mean={sum(xs) / len(xs):.1f}")


def mann_whitney(a, b):
    """Two-sided normal approximation with tie correction: (U of a, z, p)."""
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(allv)
    i = 0
    ties = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    n1, n2 = len(a), len(b)
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    z = (u - n1 * n2 / 2) / sd if sd else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return u, z, p


def boot_median_diff(a, b, n=4000, seed=1330):
    rnd = random.Random(seed)
    ds = []
    for _ in range(n):
        sa = [rnd.choice(a) for _ in a]
        sb = [rnd.choice(b) for _ in b]
        ds.append(pct(sa, .5) - pct(sb, .5))
    return pct(ds, .025), pct(ds, .975)


def fails(rows, arm, strata):
    return [r for r in rows if r.get("arm") == arm and r["stratum"] in strata and r.get("failed_first")]


def main(dirs):
    rows = load(dirs)
    arms = sorted({r.get("arm") for r in rows if r.get("arm")})
    print(f"cells: {', '.join(dirs)}")
    died = [r for r in rows if r.get("died") or (r.get("rc") not in (0, None) and r.get("rc") != 2)]
    print(f"rows: {len(rows)}; probe died or another driver error: {len(died)}")
    for strata in (("first",), ("later",), ("holder",), ("first", "later"), ("first", "later", "holder")):
        print(f"\n== stratum {'+'.join(strata)}")
        for arm in arms:
            cr = [r for r in rows if r.get("arm") == arm and r["stratum"] in strata]
            fl = fails(rows, arm, strata)
            unrec = [r for r in fl if r.get("rc") != 0]
            print(f"  {arm}: creations={len(cr)} first try out of memory={len(fl)} "
                  f"({100 * len(fl) / max(1, len(cr)):.1f} %) not recovered={len(unrec)}")
            ok = [r for r in fl if r.get("rc") == 0]
            if not ok:
                continue
            ms = [r["ms_to_ok"] for r in ok]
            tries = [r["tries"] for r in ok]
            in_calls = [sum(t[1] for t in r["try"] if t[2] == 2) for r in ok]
            waits = [r["ms_to_ok"] - sum(t[1] for t in r["try"]) for r in ok]
            fail_try_ms = [t[1] for r in ok for t in r["try"] if t[2] == 2]
            print(f"    ms to the context:        {summ(ms)}")
            print(f"    driver calls (tries):     {summ(tries)}  total={sum(tries)}")
            print(f"    ms inside failing calls:  {summ(in_calls)}")
            print(f"    ms of waits:              {summ([round(w, 1) for w in waits])}")
            print(f"    one failing call, ms:     {summ(fail_try_ms)}")
            gave = [r for r in ok if r.get("shipped_policy_gives_up_after_try")]
            print(f"    over 90 s: {sum(1 for v in ms if v > 90000)}; ladder rows where the shipped policy would have given up: {len(gave)}")
        if len(arms) == 2:
            a = [r["ms_to_ok"] for r in fails(rows, arms[0], strata) if r.get("rc") == 0]
            b = [r["ms_to_ok"] for r in fails(rows, arms[1], strata) if r.get("rc") == 0]
            if len(a) >= 3 and len(b) >= 3:
                u, z, p = mann_whitney(a, b)
                lo, hi = boot_median_diff(a, b)
                print(f"  {arms[0]} vs {arms[1]} ms to the context: Mann-Whitney U={u:.0f} z={z:.2f} p={p:.3g}; "
                      f"median difference {pct(a, .5) - pct(b, .5):.0f} ms, 95 % bootstrap [{lo:.0f}, {hi:.0f}]")
                ta = [r["tries"] for r in fails(rows, arms[0], strata) if r.get("rc") == 0]
                tb = [r["tries"] for r in fails(rows, arms[1], strata) if r.get("rc") == 0]
                print(f"  calls per failed creation: {arms[0]} mean {sum(ta) / len(ta):.1f}, {arms[1]} mean {sum(tb) / len(tb):.1f}")
    # The host at the tries of failed creations: free 2 MiB blocks at the failing tries and at the try that succeeded.
    fl = [r for r in rows if r.get("failed_first") and r.get("rc") == 0]
    if fl:
        at_fail = [sum(t[5]) for r in fl for t in r["try"] if t[2] == 2]
        at_ok = [sum(r["try"][-1][5]) for r in fl]
        mf_fail = [t[3] for r in fl for t in r["try"] if t[2] == 2]
        print(f"\nfree blocks of order >= 9 read before a failing try: {summ(at_fail)}")
        print(f"free blocks of order >= 9 read before the try that succeeded: {summ(at_ok)}")
        print(f"MemFree MB before a failing try: {summ(mf_fail)}")
        print(f"MemAvailable MB before a failing try: {summ([t[4] for r in fl for t in r['try'] if t[2] == 2])}")


if __name__ == "__main__":
    main(sys.argv[1:])
