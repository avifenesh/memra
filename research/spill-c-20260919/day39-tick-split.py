#!/usr/bin/env python3
"""Day 39 per-tick split of the day-39 three-binary tenant-stall cell on the target card (DAY39.md section 1, fixed
before the cell ran).

Day 38's reader (day38-tick-split.py) moved to the day-39 layout; the tick identification, the quantities, the rule,
the outcomes, the `not_defined` handling and the hypothesis line are day 38's, unchanged. Layout: <ev>/<program>/
pass{1,2}/<kind>/, programs p1-b0, p2-b1, p3-b2, p4-b2, p5-b1, p6-b0; contrast b1-b0: o1 = p2-b1 against p1-b0,
o2 = p5-b1 against p6-b0; contrast b2-b1: o1 = p3-b2 against p2-b1, o2 = p4-b2 against p5-b1.

Admissibility is the day-39 reader's and is NOT recomputed: this reader requires day39-stall-reading.py's banked
output to carry the line `DAY39 ADMISSIBLE: 60 of 60 receipts; all=True` and every one of the 60 receipts to exist;
otherwise it refuses and prints why. Nothing is replayed.

Tick identification, per arm run (day 38's):
  f          = the receipt's `fire_at` minus 1: the index in `itl_ms` of the gap from the tenant's 24th token to its
               25th, the gap in progress when the harness fired the intruder.
  stretched  = `itl_ms[i] > 3 * p50`, p50 the run's own median gap.
  tick 1     = the first stretched gap at an index >= f, in time order.
  tick 2     = the next stretched gap after tick 1, in time order.
  A run with fewer than k stretched gaps at an index >= f has no tick k.
Readings per contrast, day 38's rule unchanged, on q=tick1 and q=tick2 separately: (ii) per binary per arm (N=40),
(iii) new minus old per arm per block (N=20 per side), (iv) the DiD per class per block. A tick-k quantity is defined
for a program's arm only when every one of its 20 arm runs has tick k; otherwise `not_defined`.
The hypothesis line, day 38's P1 to P5 unchanged, is evaluated on contrast b1-b0 (option (a) against the base tree,
the contrast day 38 registered it on): P1 demote-on q=tick2 `moved` with d < 0 in both blocks; P2 demote-on q=tick1
`under_resolution`; P3 did-demote q=tick2 `moved` with d < 0; P4 did-demote q=tick1 `under_resolution`; P5 (control)
demote-off `under_resolution` on both ticks; the combination rule is day 38's. Contrast b2-b1 carries no registered
prediction; its lines are read under the same rule and described.
Descriptive lines per run and per program per arm are day 38's.

    day39-tick-split.py <ev_dir> <day39 reading.log>
    day39-tick-split.py --selftest
"""
import json
import math
import os
import statistics
import sys

ORDER = {1: ("prime", "off", "on"), 2: ("on", "off", "prime")}
PROGRAMS = (("p1-b0", "b0"), ("p2-b1", "b1"), ("p3-b2", "b2"), ("p4-b2", "b2"), ("p5-b1", "b1"), ("p6-b0", "b0"))
TREE_PROGS = (("b0", ("p1-b0", "p6-b0")), ("b1", ("p2-b1", "p5-b1")), ("b2", ("p3-b2", "p4-b2")))
CONTRASTS = (("b1-b0", (("o1", "p2-b1", "p1-b0"), ("o2", "p5-b1", "p6-b0"))),
             ("b2-b1", (("o1", "p3-b2", "p2-b1"), ("o2", "p4-b2", "p5-b1"))))
ARMS = (("prime", "prime", "prime"), ("demote-off", "off", "demote"), ("demote-on", "on", "demote"),
        ("promote-off", "off", "promote"), ("promote-on", "on", "promote"))
DAY39_ADMISSIBLE = "DAY39 ADMISSIBLE: 60 of 60 receipts; all=True"
STRETCH = 3.0
FIRE_SLACK_MS = 0.05


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def med_iqr(xs):
    return pct(xs, .5), pct(xs, .75) - pct(xs, .25)


def classify(v, u):
    return "isolated" if abs(v) > u else "under_resolution"


def arm_runs(rec):
    return [r for r in rec["runs"] if r["arm"] != "idle"]


def split(run, fire_at):
    """The positional tick identification of the module docstring, for one arm run."""
    itl, p50 = run["itl_ms"], run["p50"]
    f = fire_at - 1
    after = [(i - f, x) for i, x in enumerate(itl) if i >= f and x > STRETCH * p50]
    before = sum(1 for i, x in enumerate(itl) if i < f and x > STRETCH * p50)
    top = sorted(itl, reverse=True)[:2]
    out = {"f": f, "p50": p50, "after": after, "before": before, "top2": top,
           "tick1": after[0][1] if len(after) >= 1 else None, "off1": after[0][0] if len(after) >= 1 else None,
           "tick2": after[1][1] if len(after) >= 2 else None, "off2": after[1][0] if len(after) >= 2 else None}
    fired = (run.get("intruder") or {}).get("fired_at_ms")
    if fired is not None and run.get("ttft_ms") is not None and len(itl) > f:
        a_fire = run["ttft_ms"] + sum(itl[:f])          # arrival of the 24th token
        a_next = a_fire + itl[f]                        # arrival of the 25th token
        out["fire_ok"] = a_fire - FIRE_SLACK_MS <= fired <= a_next + FIRE_SLACK_MS
    else:
        out["fire_ok"] = False
    return out


def tick_values(rec, k):
    """(values, missing, N) of q=tick{k} over one receipt's arm runs."""
    vals, missing, n = [], 0, 0
    for r in arm_runs(rec):
        if "stall_ms" not in r:
            continue
        n += 1
        v = split(r, rec["fire_at"])[f"tick{k}"]
        if v is None:
            missing += 1
        else:
            vals.append(v)
    return vals, missing, n


def outcome(blocks):
    """blocks: list of (d, unc, state); state is True (readable), 'not_defined'. Day 37's outcomes, unchanged."""
    if len(blocks) != 2:
        return "inadmissible"
    if any(s == "not_defined" for _, _, s in blocks):
        return "not_defined"
    iso = [abs(d) > u for d, u, _ in blocks]
    if all(iso):
        return "moved" if (blocks[0][0] > 0) == (blocks[1][0] > 0) else "order_split"
    return "under_resolution"


def load(ev, reading_log):
    txt = open(reading_log, errors="replace").read()
    if DAY39_ADMISSIBLE not in txt.splitlines():
        raise SystemExit(f"DAY39 TICK REFUSED: {reading_log} does not carry `{DAY39_ADMISSIBLE}`")
    recs = {}
    for prog, _tree in PROGRAMS:
        for p in (1, 2):
            for kind in ORDER[p]:
                for mode in (("prime",) if kind == "prime" else ("demote", "promote")):
                    path = os.path.join(ev, prog, f"pass{p}", kind, mode, "receipt.json")
                    if not os.path.exists(path):
                        raise SystemExit(f"DAY39 TICK REFUSED: {path} missing (the day-39 reader admitted 60 of 60)")
                    recs[(prog, p, kind, mode)] = json.load(open(path))
    if len(recs) != 60:
        raise SystemExit(f"DAY39 TICK REFUSED: {len(recs)} receipts, the day-39 reader admitted 60")
    return recs


def main_read(ev, reading_log):
    recs = load(ev, reading_log)
    print(f"DAY39 TICK ADMISSIBILITY: the day-39 reader's, taken from {reading_log} (`{DAY39_ADMISSIBLE}`); 60 receipts loaded; "
          f"nothing replayed or re-admitted")

    # Descriptive: per run, then per program per arm
    for prog, tree in PROGRAMS:
        for arm, kind, mode in ARMS:
            rows = []
            for p in (1, 2):
                rec = recs[(prog, p, kind, mode)]
                for r in arm_runs(rec):
                    if "stall_ms" not in r:
                        continue
                    s = split(r, rec["fire_at"])
                    rows.append(s)
                    fmt = lambda v: "none" if v is None else f"{v:.1f}"
                    print(f"  DAY39 TICK RUN prog={prog} tree={tree} pass={p} arm={arm} run={r['run_id']} f={s['f']} p50={s['p50']:.2f} "
                          f"stretched_after_fire={[(o, round(x, 1)) for o, x in s['after']]} stretched_before_fire={s['before']} "
                          f"tick1={fmt(s['tick1'])} (offset {s['off1']}) tick2={fmt(s['tick2'])} (offset {s['off2']}) "
                          f"top2={[round(x, 1) for x in s['top2']]} fire_ok={s['fire_ok']}")
            cnt = [len(s["after"]) for s in rows]
            o1 = [s["off1"] for s in rows if s["off1"] is not None]
            o2 = [s["off2"] for s in rows if s["off2"] is not None]
            adj = sum(1 for s in rows if s["off1"] is not None and s["off2"] is not None and s["off2"] == s["off1"] + 1)
            pair = sum(1 for s in rows if s["tick2"] is not None and sorted([s["tick1"], s["tick2"]]) == sorted(s["top2"]))
            mm = lambda xs: f"{statistics.median(xs):.0f} ({min(xs)} to {max(xs)})" if xs else "none"
            print(f"DAY39 TICK SHAPE prog={prog} tree={tree} arm={arm} N={len(rows)} stretched_after_fire median={mm(cnt)} "
                  f"stretched_before_fire_sum={sum(s['before'] for s in rows)} tick1_offset={mm(o1)} tick2_offset={mm(o2)} "
                  f"runs_with_tick1={len(o1)}/{len(rows)} runs_with_tick2={len(o2)}/{len(rows)} tick2_right_after_tick1={adj}/{len(rows)} "
                  f"pair_equals_day37_top2={pair}/{len(rows)} fire_ok={sum(1 for s in rows if s['fire_ok'])}/{len(rows)}")

    def pool(progs, kind, mode, k):
        xs, missing, n = [], 0, 0
        for prog in progs:
            for p in (1, 2):
                v, m, nn = tick_values(recs[(prog, p, kind, mode)], k)
                xs.extend(v)
                missing += m
                n += nn
        return xs, missing, n

    all_verdicts = {}
    for k in (1, 2):
        qname = f"tick{k}"
        # (ii) per binary per arm
        per = {}
        for tree, progs in TREE_PROGS:
            for arm, kind, mode in ARMS:
                xs, miss, n = pool(progs, kind, mode, k)
                if miss or not xs:
                    per[(tree, arm)] = None
                    print(f"DAY39 TICK BINARY q={qname} tree={tree} arm={arm} N={n} -> not_defined ({miss} of {n} runs lack tick {k})")
                    continue
                m, u = med_iqr(xs)
                per[(tree, arm)] = (m, u)
                print(f"DAY39 TICK BINARY q={qname} tree={tree} arm={arm} N={len(xs)} median={m:.1f} iqr={u:.1f} -> defined")
            for cls in ("demote", "promote"):
                on, off = per.get((tree, f"{cls}-on")), per.get((tree, f"{cls}-off"))
                if on and off:
                    d, unc = on[0] - off[0], math.sqrt(on[1] ** 2 + off[1] ** 2)
                    print(f"DAY39 TICK BINARY q={qname} tree={tree} class={cls} on_minus_off={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
                else:
                    print(f"DAY39 TICK BINARY q={qname} tree={tree} class={cls} on_minus_off -> not_defined")
        for cname, blocks in CONTRASTS:
            # (iii) new minus old per arm per block
            verdict, block_vals, outs = [], {}, {}
            for arm, kind, mode in ARMS:
                bl = []
                for bname, pn, po in blocks:
                    xn, mn_, nn = pool((pn,), kind, mode, k)
                    xo, mo_, no = pool((po,), kind, mode, k)
                    if mn_ or mo_ or not xn or not xo:
                        print(f"DAY39 TICK CONTRAST contrast={cname} q={qname} arm={arm} block={bname} ({pn} against {po}): not_defined "
                              f"({mn_} of {nn} new runs, {mo_} of {no} old runs lack tick {k})")
                        bl.append((0.0, 0.0, "not_defined"))
                        continue
                    mn, un = med_iqr(xn)
                    mo, uo = med_iqr(xo)
                    block_vals[(arm, bname)] = (mn, un, mo, uo)
                    d, unc = mn - mo, math.sqrt(un ** 2 + uo ** 2)
                    bl.append((d, unc, True))
                    print(f"DAY39 TICK CONTRAST contrast={cname} q={qname} arm={arm} block={bname} ({pn} against {po}): new {mn:.1f} (iqr {un:.1f}, N={len(xn)}) "
                          f"old {mo:.1f} (iqr {uo:.1f}, N={len(xo)}) d={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
                o = outcome(bl)
                outs[arm] = (o, bl)
                verdict.append(f"{arm} " + " ".join(f"{b[0]}={'nd' if x[2] == 'not_defined' else format(x[0], '+.1f') + '/' + format(x[1], '.1f')}"
                                                    for b, x in zip(blocks, bl)) + f" {o}")
            # (iv) the DiD per class per block
            for cls in ("demote", "promote"):
                bl = []
                for bname, _pn, _po in blocks:
                    on, off = block_vals.get((f"{cls}-on", bname)), block_vals.get((f"{cls}-off", bname))
                    if not on or not off:
                        print(f"DAY39 TICK DID contrast={cname} q={qname} class={cls} block={bname}: not_defined (an arm of the block is not_defined)")
                        bl.append((0.0, 0.0, "not_defined"))
                        continue
                    d = (on[0] - off[0]) - (on[2] - off[2])
                    unc = math.sqrt(on[1] ** 2 + off[1] ** 2 + on[3] ** 2 + off[3] ** 2)
                    bl.append((d, unc, True))
                    print(f"DAY39 TICK DID contrast={cname} q={qname} class={cls} block={bname}: (on-off)_new {on[0] - off[0]:+.1f} (on-off)_old {on[2] - off[2]:+.1f} "
                          f"d={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
                o = outcome(bl)
                outs[f"did-{cls}"] = (o, bl)
                verdict.append(f"did-{cls} " + " ".join(f"{b[0]}={'nd' if x[2] == 'not_defined' else format(x[0], '+.1f') + '/' + format(x[1], '.1f')}"
                                                        for b, x in zip(blocks, bl)) + f" {o}")
            print(f"DAY39 TICK VERDICT contrast={cname} q={qname}: " + "; ".join(verdict))
            all_verdicts[(cname, k)] = outs

    # The hypothesis line, day 38's, on contrast b1-b0 only
    verdicts = {k: all_verdicts[("b1-b0", k)] for k in (1, 2)}

    def moved_neg(o):
        return o[0] == "moved" and all(d < 0 for d, _, _ in o[1])

    def state(o, pred):
        return "not_defined" if o[0] == "not_defined" else ("holds" if pred(o) else "fails")

    p = {
        "P1": state(verdicts[2]["demote-on"], moved_neg),
        "P2": state(verdicts[1]["demote-on"], lambda o: o[0] == "under_resolution"),
        "P3": state(verdicts[2]["did-demote"], moved_neg),
        "P4": state(verdicts[1]["did-demote"], lambda o: o[0] == "under_resolution"),
    }
    c1, c2 = verdicts[1]["demote-off"][0], verdicts[2]["demote-off"][0]
    p5 = "not_defined" if "not_defined" in (c1, c2) else ("holds" if c1 == c2 == "under_resolution" else "fails")
    nd = [k for k, v in p.items() if v == "not_defined"]
    if nd:
        h = f"H not readable ({', '.join(nd)} not_defined)"
    else:
        need = ("P1", "P2", "P3", "P4") if p5 != "fails" else ("P3", "P4")
        failed = [k for k in need if p[k] != "holds"]
        h = "consistent with H" if not failed else f"H refuted ({', '.join(k + ' fails' for k in failed)})"
        if p5 == "fails":
            h += " (P5 fails: the demote-on lines read through the DiD)"
    print(f"DAY39 TICK HYPOTHESIS contrast=b1-b0 P1 demote-on tick2 moved negative: {p['P1']}; P2 demote-on tick1 under_resolution: {p['P2']}; "
          f"P3 did-demote tick2 moved negative: {p['P3']}; P4 did-demote tick1 under_resolution: {p['P4']}; "
          f"P5 demote-off tick1 and tick2 under_resolution: {p5} (tick1 {c1}, tick2 {c2}) -> {h}")
    return 0

def selftest():
    """Synthetic runs only (no receipt is read): the identification of the docstring on constructed gap series."""
    base = [7.3] * 399
    cases = []

    def run(pairs, fired_offset=0.2):
        itl = list(base)
        for i, x in pairs:
            itl[i] = x
        return {"arm": "demote", "itl_ms": itl, "p50": statistics.median(itl), "stall_ms": 0.0, "ttft_ms": 30.0,
                "intruder": {"fired_at_ms": 30.0 + sum(itl[:23]) + fired_offset}}

    cases.append(("two adjacent, larger first", run([(23, 70.0), (24, 41.0)]), (70.0, 0, 41.0, 1, 0)))
    cases.append(("two adjacent, larger SECOND: named by position", run([(23, 62.8), (24, 68.7)]), (62.8, 0, 68.7, 1, 0)))
    cases.append(("one normal tick between", run([(24, 70.0), (26, 49.0)]), (70.0, 1, 49.0, 3, 0)))
    cases.append(("a stretched gap before the fire is not a tick", run([(10, 90.0), (23, 70.0), (24, 41.0)]), (70.0, 0, 41.0, 1, 1)))
    cases.append(("one stretched gap only: no tick 2", run([(23, 56.0), (24, 9.9)]), (56.0, 0, None, None, 0)))
    cases.append(("a gap at exactly 3 x p50 is not stretched", run([(23, 70.0), (24, 21.9)]), (70.0, 0, None, None, 0)))
    ok = True
    for name, r, (t1, o1, t2, o2, before) in cases:
        s = split(r, 24)
        got = (s["tick1"], s["off1"], s["tick2"], s["off2"], s["before"])
        good = got == (t1, o1, t2, o2, before) and s["fire_ok"]
        ok = ok and good
        print(f"  selftest {'ok' if good else 'FAIL'}: {name}: got tick1={got[0]} off1={got[1]} tick2={got[2]} off2={got[3]} "
              f"before={got[4]} fire_ok={s['fire_ok']}")
    late = run([(23, 70.0), (24, 41.0)], fired_offset=70.0 + 1.0)
    good = split(late, 24)["fire_ok"] is False
    ok = ok and good
    print(f"  selftest {'ok' if good else 'FAIL'}: a fire time after the 25th token's arrival reads fire_ok=False")
    o = outcome([(-5.0, 1.0, True), (-6.0, 2.0, True)])
    o2 = outcome([(-5.0, 1.0, True), (6.0, 2.0, True)])
    o3 = outcome([(-5.0, 1.0, True), (-1.0, 2.0, True)])
    o4 = outcome([(-5.0, 1.0, True), (0.0, 0.0, "not_defined")])
    good = (o, o2, o3, o4) == ("moved", "order_split", "under_resolution", "not_defined")
    ok = ok and good
    print(f"  selftest {'ok' if good else 'FAIL'}: outcomes {o}, {o2}, {o3}, {o4}")
    print(f"DAY39 TICK SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main_read(sys.argv[1], sys.argv[2]))
