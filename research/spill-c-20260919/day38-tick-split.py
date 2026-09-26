#!/usr/bin/env python3
"""Day 38 per-tick split of day 37's two-binary tenant-stall cell on the RTX 5090 class (DAY38.md section 1, fixed
before this reader ran on the day-37 receipts).

A POST-HOC READER OVER A COMPLETED CELL. The day-37 receipts were read on day 37 (the worst-tick stall and the two
largest gaps summed, `day37-stall-reading.py`); this reader splits the summed pair by position. It is evidence for
attribution, not a new cell. Day 37's reader with one change, the quantity: the tenant's two stretched ticks named by
POSITION relative to the intruder's fire, not by rank, each read under day 37's rule unchanged.

Layout, programs, blocks and arms are day 37's: <ev>/<program>/pass{1,2}/<kind>/, programs p1-base, p2-opta, p3-opta,
p4-base; o1 = p2-opta against p1-base, o2 = p3-opta against p4-base.

Admissibility is day 37's and is NOT recomputed: the reader requires day 37's banked `reading.log` to carry the line
`DAY37 ADMISSIBLE: 40 of 40 receipts; all=True` and every one of the 40 receipts to exist; nothing is replayed.

Tick identification, per arm run (fixed now):
  f          = the receipt's `fire_at` minus 1: the index in `itl_ms` of the gap from the tenant's 24th token to its
               25th, the gap in progress when the harness fired the intruder (`itl_ms[i]` spans token i+1 to i+2).
  stretched  = `itl_ms[i] > 3 * p50`, p50 the run's own median gap (day 35's descriptive term, A day 16's).
  tick 1     = the first stretched gap at an index >= f, in time order.
  tick 2     = the next stretched gap after tick 1, in time order.
  A run with fewer than k stretched gaps at an index >= f has no tick k.
The quantities q=tick1 and q=tick2 are the per-run values of those gaps in ms (raw gaps, as day 37's `top1_plus_top2`
is raw gaps).

Readings, day 37's rule unchanged, on q=tick1 and on q=tick2 separately:
  (ii)  per binary per arm, pooled over its four receipts (N=40): median, IQR, ON minus OFF per class with unc.
  (iii) option (a) minus base per arm per block, each side the program's two passes pooled (N=20):
        d = median(opta) - median(base), unc = sqrt(iqr_opta^2 + iqr_base^2), `isolated` when |d| > unc; per arm
        `moved` when both blocks are isolated with the same sign, `order_split` when both are isolated with opposite
        signs, `under_resolution` otherwise.
  (iv)  the DiD per class per block, (on - off)_opta - (on - off)_base, unc = sqrt of the four IQRs squared summed,
        the same outcomes.
  A tick-k quantity is defined for a program's arm only when every one of its 20 arm runs has tick k. Where it is
  not, the line prints `not_defined (R of N runs lack tick k)` and no outcome is read from it; a block or DiD that
  uses it, and the arm's outcome, print `not_defined`.
The hypothesis line (DAY38.md section 1), evaluated mechanically from the outcomes above:
  P1 demote-on q=tick2 `moved` with d < 0 in both blocks; P2 demote-on q=tick1 `under_resolution`;
  P3 did-demote q=tick2 `moved` with d < 0 in both blocks; P4 did-demote q=tick1 `under_resolution`;
  P5 (control) demote-off `under_resolution` on both ticks.
  Any of P1 to P4 not_defined: `H not readable`. P5 holds: `consistent with H` iff P1 to P4 hold, else `H refuted`
  naming the failed predictions. P5 fails: the demote-on lines are read through the DiD (day 37's rule), so
  `consistent with H` iff P3 and P4 hold, else `H refuted`.
  The promote class is read under the same rule with no prediction registered.
Descriptive (printed, not ruled): per run, f, p50 and the stretched gaps at and after f as (offset from f, ms), the
count of stretched gaps before f, tick 1 and tick 2 with their offsets; per program per arm, the stretched-count, the
offsets, the runs where tick 2 is the gap right after tick 1, the runs where {tick 1, tick 2} equals day 37's
{top1, top2}, and the fire check (the harness's `fired_at_ms` between the reconstructed arrivals of the 24th and 25th
tokens, `ttft_ms` plus the cumulative gaps, 0.05 ms slack for the gaps' 3-decimal rounding).

    day38-tick-split.py <ev_dir> <day37 reading.log>
    day38-tick-split.py --selftest
"""
import json
import math
import os
import statistics
import sys

ORDER = {1: ("prime", "off", "on"), 2: ("on", "off", "prime")}
PROGRAMS = (("p1-base", "base"), ("p2-opta", "opta"), ("p3-opta", "opta"), ("p4-base", "base"))
BLOCKS = (("o1", "p2-opta", "p1-base"), ("o2", "p3-opta", "p4-base"))
ARMS = (("prime", "prime", "prime"), ("demote-off", "off", "demote"), ("demote-on", "on", "demote"),
        ("promote-off", "off", "promote"), ("promote-on", "on", "promote"))
DAY37_ADMISSIBLE = "DAY37 ADMISSIBLE: 40 of 40 receipts; all=True"
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
    if DAY37_ADMISSIBLE not in txt.splitlines():
        raise SystemExit(f"DAY38 REFUSED: {reading_log} does not carry `{DAY37_ADMISSIBLE}`")
    recs = {}
    for prog, _tree in PROGRAMS:
        for p in (1, 2):
            for kind in ORDER[p]:
                for mode in (("prime",) if kind == "prime" else ("demote", "promote")):
                    path = os.path.join(ev, prog, f"pass{p}", kind, mode, "receipt.json")
                    if not os.path.exists(path):
                        raise SystemExit(f"DAY38 REFUSED: {path} missing (day 37 admitted 40 of 40)")
                    recs[(prog, p, kind, mode)] = json.load(open(path))
    if len(recs) != 40:
        raise SystemExit(f"DAY38 REFUSED: {len(recs)} receipts, day 37 admitted 40")
    return recs


def main_read(ev, reading_log):
    recs = load(ev, reading_log)
    print(f"DAY38 ADMISSIBILITY: day 37's, taken from {reading_log} (`{DAY37_ADMISSIBLE}`); 40 receipts loaded; "
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
                    print(f"  DAY38 RUN prog={prog} tree={tree} pass={p} arm={arm} run={r['run_id']} f={s['f']} p50={s['p50']:.2f} "
                          f"stretched_after_fire={[(o, round(x, 1)) for o, x in s['after']]} stretched_before_fire={s['before']} "
                          f"tick1={fmt(s['tick1'])} (offset {s['off1']}) tick2={fmt(s['tick2'])} (offset {s['off2']}) "
                          f"top2={[round(x, 1) for x in s['top2']]} fire_ok={s['fire_ok']}")
            cnt = [len(s["after"]) for s in rows]
            o1 = [s["off1"] for s in rows if s["off1"] is not None]
            o2 = [s["off2"] for s in rows if s["off2"] is not None]
            adj = sum(1 for s in rows if s["off1"] is not None and s["off2"] is not None and s["off2"] == s["off1"] + 1)
            pair = sum(1 for s in rows if s["tick2"] is not None and sorted([s["tick1"], s["tick2"]]) == sorted(s["top2"]))
            mm = lambda xs: f"{statistics.median(xs):.0f} ({min(xs)} to {max(xs)})" if xs else "none"
            print(f"DAY38 SHAPE prog={prog} tree={tree} arm={arm} N={len(rows)} stretched_after_fire median={mm(cnt)} "
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

    verdicts = {}
    for k in (1, 2):
        qname = f"tick{k}"
        # (ii) per binary per arm
        per = {}
        for tree, progs in (("base", ("p1-base", "p4-base")), ("opta", ("p2-opta", "p3-opta"))):
            for arm, kind, mode in ARMS:
                xs, miss, n = pool(progs, kind, mode, k)
                if miss or not xs:
                    per[(tree, arm)] = None
                    print(f"DAY38 BINARY q={qname} tree={tree} arm={arm} N={n} -> not_defined ({miss} of {n} runs lack tick {k})")
                    continue
                m, u = med_iqr(xs)
                per[(tree, arm)] = (m, u)
                print(f"DAY38 BINARY q={qname} tree={tree} arm={arm} N={len(xs)} median={m:.1f} iqr={u:.1f} -> defined")
            for cls in ("demote", "promote"):
                on, off = per.get((tree, f"{cls}-on")), per.get((tree, f"{cls}-off"))
                if on and off:
                    d, unc = on[0] - off[0], math.sqrt(on[1] ** 2 + off[1] ** 2)
                    print(f"DAY38 BINARY q={qname} tree={tree} class={cls} on_minus_off={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
                else:
                    print(f"DAY38 BINARY q={qname} tree={tree} class={cls} on_minus_off -> not_defined")
        # (iii) option (a) minus base per arm per block
        verdict, block_vals, outs = [], {}, {}
        for arm, kind, mode in ARMS:
            bl = []
            for bname, po, pb in BLOCKS:
                xo, mo_, no = pool((po,), kind, mode, k)
                xb, mb_, nb = pool((pb,), kind, mode, k)
                if mo_ or mb_ or not xo or not xb:
                    print(f"DAY38 OPTA-BASE q={qname} arm={arm} block={bname} ({po} against {pb}): not_defined "
                          f"({mo_} of {no} opta runs, {mb_} of {nb} base runs lack tick {k})")
                    bl.append((0.0, 0.0, "not_defined"))
                    continue
                mo, uo = med_iqr(xo)
                mb, ub = med_iqr(xb)
                block_vals[(arm, bname)] = (mo, uo, mb, ub)
                d, unc = mo - mb, math.sqrt(uo ** 2 + ub ** 2)
                bl.append((d, unc, True))
                print(f"DAY38 OPTA-BASE q={qname} arm={arm} block={bname} ({po} against {pb}): opta {mo:.1f} (iqr {uo:.1f}, N={len(xo)}) "
                      f"base {mb:.1f} (iqr {ub:.1f}, N={len(xb)}) d={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
            o = outcome(bl)
            outs[arm] = (o, bl)
            verdict.append(f"{arm} " + " ".join(f"{b[0]}={'nd' if x[2] == 'not_defined' else format(x[0], '+.1f') + '/' + format(x[1], '.1f')}"
                                                for b, x in zip(BLOCKS, bl)) + f" {o}")
        # (iv) the DiD per class per block
        for cls in ("demote", "promote"):
            bl = []
            for bname, _po, _pb in BLOCKS:
                on, off = block_vals.get((f"{cls}-on", bname)), block_vals.get((f"{cls}-off", bname))
                if not on or not off:
                    print(f"DAY38 DID q={qname} class={cls} block={bname}: not_defined (an arm of the block is not_defined)")
                    bl.append((0.0, 0.0, "not_defined"))
                    continue
                d = (on[0] - off[0]) - (on[2] - off[2])
                unc = math.sqrt(on[1] ** 2 + off[1] ** 2 + on[3] ** 2 + off[3] ** 2)
                bl.append((d, unc, True))
                print(f"DAY38 DID q={qname} class={cls} block={bname}: (on-off)_opta {on[0] - off[0]:+.1f} (on-off)_base {on[2] - off[2]:+.1f} "
                      f"d={d:+.1f} unc={unc:.1f} -> {classify(d, unc)}")
            o = outcome(bl)
            outs[f"did-{cls}"] = (o, bl)
            verdict.append(f"did-{cls} " + " ".join(f"{b[0]}={'nd' if x[2] == 'not_defined' else format(x[0], '+.1f') + '/' + format(x[1], '.1f')}"
                                                    for b, x in zip(BLOCKS, bl)) + f" {o}")
        print(f"DAY38 VERDICT q={qname}: " + "; ".join(verdict))
        verdicts[k] = outs

    # The hypothesis line (DAY38.md section 1), mechanical
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
    print(f"DAY38 HYPOTHESIS P1 demote-on tick2 moved negative: {p['P1']}; P2 demote-on tick1 under_resolution: {p['P2']}; "
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
    print(f"DAY38 SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main_read(sys.argv[1], sys.argv[2]))
