#!/usr/bin/env python3
"""Day 39 reading of the three-binary tenant-stall cell on the target card (DAY39.md section 1, fixed before the run).

Day 37's reader (day37-stall-reading.py) with three binaries and two contrasts; the rule, the readings, the quantities
and the four outcomes are day 37's, unchanged. Layout: <ev>/<program>/pass{1,2}/<kind>/, programs p1-b0, p2-b1,
p3-b2, p4-b2, p5-b1, p6-b0 in that order, each program day 35's layout (pass1/{prime,off,on}, pass2/{on,off,prime};
prime boot: prime/receipt.json; off and on boots: demote/ and promote/). B0 is `091a931c0` (day 37's base tree, no
option (a)), B1 `9717e8d57` (option (a) without the D2H spans), B2 `160929a92` (option (a) plus A day 30's D2H spans).
The harness is day35_stall_cell.py byte-for-byte; its rule line is replayed from every receipt.

Per receipt, admissibility is day 37's; the ON demote line set is per tree:
  b0: day 37's `base` set: every ON demote run with a demote line has `demote submitted off the tick`, a
      `D2H receipt ... require=ok` and `demote published off the tick`.
  b1: day 37's `opta` set: `demote submitted off the tick`, a `D2H receipt ... require=ok`, `demote copy complete off
      the tick` and `demote digests landed off the tick`, and zero `demote published off the tick`.
  b2: the `opta` set, and every `demote copy complete off the tick` line of the run carries an `items=N (K KV, S f32
      spans)` term with S >= 1 (A day 30's worker.rs prints it on every copy-complete line; the day-30 receipts'
      A3 reading counts 132 of 132).
The rest is day 37's: `STALL REPLAY: PASS`; errors=0; one tenant text across the 20 runs; every intruder inside the
tenant's window; no `[prefix-host]` line in the prime boot; no `off the tick` line under OFF; the ON promote shape
(1,1,1,1,0); none of the BAD lines in the boot's server.log; demote receipts >= 9 server_demote_ms, promote receipts
== 10 server_promote_ms.

Readings (all figures are the harness's per-run stalls; median and IQR = p75 - p25; unc in quadrature):
  (i)   per program per pass: day 35's lines.
  (ii)  per binary per arm, pooled over its four receipts (two programs x two passes, N=40), ON minus OFF per class.
  (iii) per contrast, new minus old per arm, per order block, each side the program's two passes pooled (N=20):
        contrast b1-b0: o1 = p2-b1 against p1-b0 (B1 after B0), o2 = p5-b1 against p6-b0 (B1 before B0);
        contrast b2-b1: o1 = p3-b2 against p2-b1 (B2 after B1), o2 = p4-b2 against p5-b1 (B2 before B1).
        d = median(new) - median(old), unc = sqrt(iqr_new^2 + iqr_old^2), `isolated` when |d| > unc. Per arm:
        `moved` when both blocks are isolated with the same sign; `order_split` when both are isolated with opposite
        signs; `under_resolution` otherwise; `inadmissible` when any receipt of either block is.
  (iv)  per contrast, the DiD per class per block: (on - off)_new - (on - off)_old, unc = sqrt of the four IQRs
        squared summed, the same outcomes.
  (v)   the secondary quantity top1_plus_top2 (the tenant's two largest ITL gaps per run, summed), (ii) to (iv) on it.
  (vi)  the option (a) ledger per tree (b1, b2): every `demote digests landed off the tick` line of the ON demote arms'
        runs, split into the boot's first three demotes and the 4th onward; b0's count of these lines (expected 0).
  (vii) day 35's attribution per tree (b0: `demote published off the tick`, b1 and b2: `demote copy complete off the
        tick`), described, not ruled on.
Nothing here is tuned; no threshold beyond the cell's own IQR. `admissible=False` decides nothing and says so.

    day39-stall-reading.py <ev_dir>
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day35_stall_cell.py")
BAD = re.compile(r"demote failed|promote failed|demote refused|promote refused|restore refused|capture refused|latched off|TIER DISABLED")
ORDER = {1: ("prime", "off", "on"), 2: ("on", "off", "prime")}
PROGRAMS = (("p1-b0", "b0"), ("p2-b1", "b1"), ("p3-b2", "b2"), ("p4-b2", "b2"), ("p5-b1", "b1"), ("p6-b0", "b0"))
TREE_PROGS = (("b0", ("p1-b0", "p6-b0")), ("b1", ("p2-b1", "p5-b1")), ("b2", ("p3-b2", "p4-b2")))
CONTRASTS = (("b1-b0", (("o1", "p2-b1", "p1-b0"), ("o2", "p5-b1", "p6-b0"))),
             ("b2-b1", (("o1", "p3-b2", "p2-b1"), ("o2", "p4-b2", "p5-b1"))))
ARMS = (("prime", "prime", "prime"), ("demote-off", "off", "demote"), ("demote-on", "on", "demote"),
        ("promote-off", "off", "promote"), ("promote-on", "on", "promote"))
RE_COMPLETION_BY_SET = {
    "base": re.compile(r"demote published off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion"),
    "opta": re.compile(r"demote copy complete off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion"),
}
RE_COMPLETION = {"b0": RE_COMPLETION_BY_SET["base"], "b1": RE_COMPLETION_BY_SET["opta"], "b2": RE_COMPLETION_BY_SET["opta"]}
RE_SPANS = re.compile(r"items=\d+ \(\d+ KV, (\d+) f32 spans\)")
RE_PROMOTE_COMPLETION = re.compile(r"promote published off the tick: ticket complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion")
RE_LEDGER = re.compile(
    r"demote digests landed off the tick: ticket seq=(\d+), (\d+) payloads \(([0-9.]+)MB\) hashed in ([0-9.]+)ms on the hash "
    r"helper, landed after (\d+) poll\(s\) \(([^)]*)\); the owner thread held ([0-9.]+)ms across the demote: pre-submit "
    r"([0-9.]+), copy settle ([0-9.]+) over (\d+) poll\(s\), hashing polls ([0-9.]+), take-back bind and publish ([0-9.]+); "
    r"owner in-completion ([0-9.]+)ms; wall ([0-9.]+)ms t0 to publication; (\d+) hit\(s\) parked on the Hashing entry "
    r"\((\d+) re-park\(s\)\)")
LEDGER_FIELDS = ("seq", "payloads", "mb", "hashed_in", "landed_polls", "mode", "owner_held", "pre_submit", "copy_settle",
                 "settle_polls", "hashing_polls", "take_back_publish", "owner_in_completion", "wall", "parked_hits", "reparks")


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def med_iqr(xs):
    return pct(xs, .5), pct(xs, .75) - pct(xs, .25)


def classify(v, u):
    return "isolated" if abs(v) > u else "under_resolution"


def replay(path):
    rep = subprocess.run([sys.executable, HARNESS, "--replay", path], capture_output=True, text=True)
    return "STALL REPLAY: PASS" in rep.stdout


def arm_runs(rec):
    return [r for r in rec["runs"] if r["arm"] != "idle"]


def stalls(rec):
    return [r["stall_ms"] for r in arm_runs(rec) if "stall_ms" in r]


def gap_sums(rec):
    out = []
    for r in arm_runs(rec):
        if "stall_ms" in r:
            g = sorted(r["itl_ms"], reverse=True)
            out.append(g[0] + g[1])
    return out


def landed(rec):
    out = []
    for r in arm_runs(rec):
        i = r.get("intruder") or {}
        if "error" in i or r.get("tenant_wall_ms") is None or i.get("wall_ms") is None:
            out.append(False)
        else:
            out.append(i["fired_at_ms"] + i["wall_ms"] <= r["tenant_wall_ms"])
    return out


def count(lines, needle):
    return sum(1 for ln in lines if needle in ln)


def ledger(line):
    m = RE_LEDGER.search(line)
    if not m:
        return None
    d = dict(zip(LEDGER_FIELDS, m.groups()))
    return {k: (v if k == "mode" else (int(v) if k in ("seq", "payloads", "landed_polls", "settle_polls", "parked_hits", "reparks") else float(v)))
            for k, v in d.items()}


def admissible(tree, kind, mode, rec, log):
    s = rec["summary"]
    why = []
    if s["errors"]:
        why.append(f"errors={len(s['errors'])}")
    if len(s["tenant_text_shas"]) != 1:
        why.append("tenant text differs")
    land = landed(rec)
    if not all(land):
        why.append(f"intruder outside the tenant's window in {sum(1 for x in land if not x)} of {len(land)} runs")
    bad = BAD.findall(log)
    if bad:
        why.append(f"bad lines {sorted(set(bad))}")
    runs = arm_runs(rec)
    if mode == "prime":
        n = sum(count(r["server_log_lines"], "[prefix-host]") for r in runs)
        if n:
            why.append(f"{n} [prefix-host] lines in the prime boot")
    if mode == "demote" and len(s["server_demote_ms"]) < 9:
        why.append(f"server_demote_ms={len(s['server_demote_ms'])} (<9)")
    if mode == "promote" and len(s["server_promote_ms"]) != 10:
        why.append(f"server_promote_ms={len(s['server_promote_ms'])} (!=10)")
    if kind == "off" and "off the tick" in log:
        why.append("off-the-tick lines under OFF")
    if kind == "on" and mode == "demote":
        for r in runs:
            L = r["server_log_lines"]
            if not r["server_demote_ms"]:
                continue
            common = count(L, "demote submitted off the tick") >= 1 and any("D2H receipt" in ln and "require=ok" in ln for ln in L)
            if tree == "b0":
                ok = common and count(L, "demote published off the tick") >= 1
                what = "the door's three lines"
            else:
                ok = (common and count(L, "demote copy complete off the tick") >= 1
                      and count(L, "demote digests landed off the tick") >= 1 and count(L, "demote published off the tick") == 0)
                what = "the option (a) line set"
                if tree == "b2":
                    cc = [ln for ln in L if "demote copy complete off the tick" in ln]
                    spans = [RE_SPANS.search(ln) for ln in cc]
                    ok = ok and all(m and int(m.group(1)) >= 1 for m in spans)
                    what = "the option (a) line set with an f32 spans term on every copy-complete line"
            if not ok:
                why.append(f"run {r['run_id']} demote without {what}")
    if kind == "on" and mode == "promote":
        for r in runs:
            L = r["server_log_lines"]
            # Five independent lines. "request parked" is the TAIL of the submitted line, never a line of its own
            # (revuto on integ44 #651): it is asserted as part of that line, not counted again.
            shape = (sum(1 for ln in L if "promote submitted off the tick" in ln and ln.rstrip().endswith("request parked")),
                     sum(1 for ln in L if "H2D receipt" in ln and "require=ok" in ln),
                     count(L, "promote published off the tick"),
                     count(L, "restore not routed (contracts door)"), count(L, "restore submitted off the tick"))
            if shape != (1, 1, 1, 1, 0):
                why.append(f"run {r['run_id']} promote shape {shape} != (1,1,1,1,0)")
    return not why, "; ".join(why) or "errors=0, tenant text identical, every intruder inside the window, the server lines as pre-registered"


def outcome(blocks):
    """blocks: list of (d, unc, admissible). The four pre-registered outcomes."""
    if len(blocks) != 2 or not all(a for _, _, a in blocks):
        return "inadmissible"
    iso = [abs(d) > u for d, u, _ in blocks]
    if all(iso):
        return "moved" if (blocks[0][0] > 0) == (blocks[1][0] > 0) else "order_split"
    return "under_resolution"


def main():
    ev = sys.argv[1]
    recs, adm, logs = {}, {}, {}
    for prog, tree in PROGRAMS:
        for p in (1, 2):
            for kind in ORDER[p]:
                bd = os.path.join(ev, prog, f"pass{p}", kind)
                logp = os.path.join(bd, "server.log")
                log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
                logs[(prog, p, kind)] = log
                for mode in (("prime",) if kind == "prime" else ("demote", "promote")):
                    key = (prog, p, kind, mode)
                    path = os.path.join(bd, mode, "receipt.json")
                    if not os.path.exists(path):
                        print(f"  {prog}/pass{p}/{kind}/{mode}: NO RECEIPT (inadmissible)")
                        adm[key] = False
                        continue
                    rec = json.load(open(path))
                    rep = replay(path)
                    ok, why = admissible(tree, kind, mode, rec, log)
                    recs[key] = rec
                    adm[key] = ok and rep
                    land = landed(rec)
                    tt = sorted({r["tenant_tokens"] for r in rec["runs"]})
                    print(rec["rule_line"])
                    print(f"  {prog}/pass{p}/{kind}/{mode}: replay={'PASS' if rep else 'FAIL'} admissible={ok and rep} ({why}); "
                          f"tenant_tokens={tt} tenant_max_tokens={rec.get('tenant_max_tokens')} landed={sum(land)}/{len(land)} "
                          f"intruder_wall_ms={rec['summary']['intruder_wall_ms']} "
                          f"tenant_wall_ms_median={statistics.median([r['tenant_wall_ms'] for r in rec['runs'] if r.get('tenant_wall_ms')]):.0f}")

    # (i) day 35's lines per program per pass
    for prog, tree in PROGRAMS:
        for p in (1, 2):
            if (prog, p, "prime", "prime") in recs:
                pm, pu = med_iqr(stalls(recs[(prog, p, "prime", "prime")]))
                a = adm[(prog, p, "prime", "prime")]
                print(f"DAY39 STALL prog={prog} tree={tree} pass={p} class=prime stall_median={pm:.1f} iqr={pu:.1f} n_per_order=5 pooled=10 "
                      f"-> {'admissible' if a else 'inadmissible'}")
            for cls in ("demote", "promote"):
                vals = {}
                for kind in ("off", "on"):
                    if (prog, p, kind, cls) not in recs:
                        continue
                    vals[kind] = med_iqr(stalls(recs[(prog, p, kind, cls)]))
                    s = recs[(prog, p, kind, cls)]["summary"]
                    print(f"DAY39 STALL prog={prog} tree={tree} pass={p} class={cls} arm={kind} stall_median={vals[kind][0]:.1f} iqr={vals[kind][1]:.1f} "
                          f"idle_p50={s['idle']['p50']:.1f} arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} "
                          f"-> {'admissible' if adm[(prog, p, kind, cls)] else 'inadmissible'}")
                if "off" in vals and "on" in vals:
                    d = vals["on"][0] - vals["off"][0]
                    unc = math.sqrt(vals["on"][1] ** 2 + vals["off"][1] ** 2)
                    a = adm[(prog, p, "off", cls)] and adm[(prog, p, "on", cls)]
                    print(f"DAY39 STALL prog={prog} tree={tree} pass={p} class={cls} on_minus_off={d:+.1f} unc={unc:.1f} -> "
                          f"{classify(d, unc) if a else 'inadmissible'}")

    def pool(progs, kind, mode, fn):
        xs, ok, n = [], True, 0
        for prog in progs:
            for p in (1, 2):
                key = (prog, p, kind, mode)
                ok = ok and adm.get(key, False)
                if key in recs:
                    xs.extend(fn(recs[key]))
                    n += 1
        return xs, ok, n

    for qname, fn in (("stall", stalls), ("top1_plus_top2", gap_sums)):
        # (ii) per binary per arm
        per = {}
        for tree, progs in TREE_PROGS:
            for arm, kind, mode in ARMS:
                xs, ok, n = pool(progs, kind, mode, fn)
                if xs:
                    m, u = med_iqr(xs)
                    per[(tree, arm)] = (m, u, ok)
                    print(f"DAY39 BINARY q={qname} tree={tree} arm={arm} N={len(xs)} median={m:.1f} iqr={u:.1f} receipts={n}/4 "
                          f"-> {'admissible' if ok else 'inadmissible'}")
                else:
                    print(f"DAY39 BINARY q={qname} tree={tree} arm={arm} N=0 -> inadmissible")
            for cls in ("demote", "promote"):
                on, off = per.get((tree, f"{cls}-on")), per.get((tree, f"{cls}-off"))
                if on and off:
                    d, unc = on[0] - off[0], math.sqrt(on[1] ** 2 + off[1] ** 2)
                    print(f"DAY39 BINARY q={qname} tree={tree} class={cls} on_minus_off={d:+.1f} unc={unc:.1f} -> "
                          f"{classify(d, unc) if on[2] and off[2] else 'inadmissible'}")
        for cname, blocks in CONTRASTS:
            # (iii) new minus old per arm per block
            verdict = []
            block_vals = {}
            for arm, kind, mode in ARMS:
                bl = []
                for bname, pn, po in blocks:
                    xn, an, _ = pool((pn,), kind, mode, fn)
                    xo, ao, _ = pool((po,), kind, mode, fn)
                    if not xn or not xo:
                        print(f"DAY39 CONTRAST contrast={cname} q={qname} arm={arm} block={bname} ({pn} against {po}): no receipts -> inadmissible")
                        bl.append((0.0, 0.0, False))
                        continue
                    mn, un = med_iqr(xn)
                    mo, uo = med_iqr(xo)
                    block_vals[(arm, bname)] = (mn, un, mo, uo, an and ao)
                    d, unc = mn - mo, math.sqrt(un ** 2 + uo ** 2)
                    bl.append((d, unc, an and ao))
                    print(f"DAY39 CONTRAST contrast={cname} q={qname} arm={arm} block={bname} ({pn} against {po}): new {mn:.1f} (iqr {un:.1f}, N={len(xn)}) "
                          f"old {mo:.1f} (iqr {uo:.1f}, N={len(xo)}) d={d:+.1f} unc={unc:.1f} -> "
                          f"{classify(d, unc) if an and ao else 'inadmissible'}")
                o = outcome(bl)
                verdict.append(f"{arm} " + " ".join(f"{b[0]}={x[0]:+.1f}/{x[1]:.1f}" for b, x in zip(blocks, bl)) + f" {o}")
            # (iv) difference in differences per class per block
            for cls in ("demote", "promote"):
                bl = []
                for bname, pn, po in blocks:
                    on, off = block_vals.get((f"{cls}-on", bname)), block_vals.get((f"{cls}-off", bname))
                    if not on or not off:
                        print(f"DAY39 DID contrast={cname} q={qname} class={cls} block={bname}: missing arm -> inadmissible")
                        bl.append((0.0, 0.0, False))
                        continue
                    d = (on[0] - off[0]) - (on[2] - off[2])
                    unc = math.sqrt(on[1] ** 2 + off[1] ** 2 + on[3] ** 2 + off[3] ** 2)
                    a = on[4] and off[4]
                    bl.append((d, unc, a))
                    print(f"DAY39 DID contrast={cname} q={qname} class={cls} block={bname}: (on-off)_new {on[0] - off[0]:+.1f} (on-off)_old {on[2] - off[2]:+.1f} "
                          f"d={d:+.1f} unc={unc:.1f} -> {classify(d, unc) if a else 'inadmissible'}")
                o = outcome(bl)
                verdict.append(f"did-{cls} " + " ".join(f"{b[0]}={x[0]:+.1f}/{x[1]:.1f}" for b, x in zip(blocks, bl)) + f" {o}")
            print(f"DAY39 VERDICT contrast={cname} q={qname}: " + "; ".join(verdict))

    # (vi) the option (a) ledger on the ON demote arms' runs, and the boot-wide parked lines
    for prog, tree in PROGRAMS:
        for p in (1, 2):
            log = logs.get((prog, p, "on"), "")
            order = [e["seq"] for e in (ledger(ln) for ln in log.splitlines()) if e]
            parked = count(log.splitlines(), "hit parked on a Hashing entry")
            published = count(log.splitlines(), "demote published off the tick")
            detached = count(log.splitlines(), "hash helper detached")
            print(f"DAY39 LEDGER-BOOT prog={prog} tree={tree} pass={p} on-boot: digests_landed_lines={len(order)} "
                  f"demote_published_lines={published} hit_parked_lines={parked} hash_helper_detached_lines={detached}")
    for ltree in ("b1", "b2"):
        rows = {"first3": [], "4th_on": []}
        for prog, tree in PROGRAMS:
            if tree != ltree:
                continue
            for p in (1, 2):
                key = (prog, p, "on", "demote")
                if key not in recs:
                    continue
                order = [e["seq"] for e in (ledger(ln) for ln in logs[(prog, p, "on")].splitlines()) if e]
                for r in arm_runs(recs[key]):
                    for ln in r["server_log_lines"]:
                        e = ledger(ln)
                        if e:
                            idx = order.index(e["seq"]) + 1 if e["seq"] in order else None
                            rows["first3" if idx is not None and idx <= 3 else "4th_on"].append(e)
        for part, es in rows.items():
            if not es:
                print(f"DAY39 LEDGER tree={ltree} part={part} N=0")
                continue
            med = {k: statistics.median([e[k] for e in es]) for k in ("pre_submit", "copy_settle", "hashing_polls", "take_back_publish",
                                                                      "owner_in_completion", "owner_held", "hashed_in", "wall", "landed_polls", "settle_polls")}
            print(f"DAY39 LEDGER tree={ltree} part={part} N={len(es)} pre_submit={med['pre_submit']:.2f} copy_settle={med['copy_settle']:.2f} "
                  f"hashing_polls={med['hashing_polls']:.2f} take_back_publish={med['take_back_publish']:.2f} "
                  f"owner_in_completion={med['owner_in_completion']:.2f} owner_held={med['owner_held']:.2f} hashed_in={med['hashed_in']:.1f} "
                  f"wall={med['wall']:.1f} landed_polls_median={med['landed_polls']:.0f} settle_polls_median={med['settle_polls']:.0f} "
                  f"parked_hits_sum={sum(e['parked_hits'] for e in es)} reparks_sum={sum(e['reparks'] for e in es)} "
                  f"payloads={sorted({e['payloads'] for e in es})} modes={sorted({e['mode'] for e in es})} (medians over the ON demote arms' run lines)")

    # (vii) day 35's attribution per tree
    for prog, tree in PROGRAMS:
        for p in (1, 2):
            key = (prog, p, "on", "demote")
            if key not in recs:
                continue
            ins, comps, imc = [], [], []
            for r in arm_runs(recs[key]):
                c = [float(m.group(1)) for ln in r["server_log_lines"] for m in [RE_COMPLETION[tree].search(ln)] if m]
                ins += r["server_demote_ms"]
                comps += c
                imc += [a - b for a, b in zip(r["server_demote_ms"], c)]
            offk = (prog, p, "off", "demote")
            off_ins = recs[offk]["summary"]["server_demote_ms"] if offk in recs else []
            if ins and comps and imc and off_ins:
                print(f"DAY39 ATTRIBUTION prog={prog} tree={tree} pass={p} ON demote: demote_in median={statistics.median(ins):.1f} (N={len(ins)}) "
                      f"completion median={statistics.median(comps):.1f} (N={len(comps)}) in_minus_completion median={statistics.median(imc):.1f} "
                      f"(N={len(imc)}); OFF demote_in median={statistics.median(off_ins):.1f} (N={len(off_ins)})")
            else:
                print(f"DAY39 ATTRIBUTION prog={prog} tree={tree} pass={p} ON demote: no completion line to subtract")
    print(f"DAY39 ADMISSIBLE: {sum(1 for v in adm.values() if v)} of {len(adm)} receipts; all={all(adm.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
