#!/usr/bin/env python3
"""Day 35 reading of the tenant-stall cell on the RTX 5090 class (DAY35.md pre-registration, fixed before the run).

Layout: <ev>/pass1/{prime,off,on}/ and <ev>/pass2/{on,off,prime}/, each boot with server.log and, per harness mode,
<mode>/receipt.json (prime boot: prime/; off and on boots: demote/ and promote/). The harness is
day35_stall_cell.py (lane A's stall_cell.py plus --tenant-max-tokens); its rule line is replayed from every receipt.

Per receipt, admissibility (all must hold): `STALL REPLAY: PASS`; errors=0; the tenant's text identical across the
20 runs; every arm run's intruder returned inside the tenant's window (fired_at_ms + wall_ms <= tenant_wall_ms);
the server lines per arm: prime boot has no `[prefix-host]` line at all; OFF arms have no `off the tick` line; the
ON demote arm's every run with a demote line has `demote submitted off the tick`, a `D2H receipt ... require=ok` and
`demote published off the tick`; the ON promote arm's every run has exactly one `promote submitted off the tick`,
one `H2D receipt ... require=ok`, one `promote published off the tick`, one `request parked`, one
`restore not routed (contracts door)` and zero `restore submitted off the tick`; the boot's server.log carries none
of `demote failed|promote failed|demote refused|promote refused|restore refused|capture refused|latched off|
TIER DISABLED`. Demote receipts carry at least 9 server_demote_ms, promote receipts exactly 10 server_promote_ms.

Per pass, per class (demote, promote): stall_median(arm) is the harness's rule-line figure (median of the 10 per-run
stalls, N=5 per order, both orders), IQR the p75 - p25 of those 10; on_minus_off with unc in quadrature;
`isolated` when |d| > unc, else `under_resolution`. The prime arm per pass: stall_median and IQR alone. Pooled
over the two passes (N=20 per arm) as context. The day-27 attribution on this card, from the ON demote runs:
`demote_in` (`[prefix-host] demote: ... in Y ms`), `completion` (`demote published off the tick: ... X ms from
submission to completion`), `in - completion` per run and its median; the tenant's two largest gaps per run and
their sum; the OFF demote's `in` median beside it. Nothing here is tuned; no threshold beyond the cell's own IQR.
`admissible=False` decides nothing and says so.

    day35-stall-reading.py <ev_dir>
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
RE_COMPLETION = re.compile(r"demote published off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion")
RE_PROMOTE_COMPLETION = re.compile(r"promote published off the tick: ticket complete after \d+ poll\(s\), ([0-9.]+)ms from submission to completion")


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


def admissible(kind, mode, rec, log):
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
            if r["server_demote_ms"] and not (count(L, "demote submitted off the tick") >= 1
                                              and any("D2H receipt" in ln and "require=ok" in ln for ln in L)
                                              and count(L, "demote published off the tick") >= 1):
                why.append(f"run {r['run_id']} demote without the door's three lines")
    if kind == "on" and mode == "promote":
        for r in runs:
            L = r["server_log_lines"]
            # Five independent lines. On this tree "request parked" is the TAIL of the submitted
            # line, never a line of its own (revuto on integ44 #651): it is asserted as part of
            # that line, not counted again.
            shape = (sum(1 for ln in L if "promote submitted off the tick" in ln and ln.rstrip().endswith("request parked")),
                     sum(1 for ln in L if "H2D receipt" in ln and "require=ok" in ln),
                     count(L, "promote published off the tick"),
                     count(L, "restore not routed (contracts door)"), count(L, "restore submitted off the tick"))
            if shape != (1, 1, 1, 1, 0):
                why.append(f"run {r['run_id']} promote shape {shape} != (1,1,1,1,0)")
    return not why, "; ".join(why) or "errors=0, tenant text identical, every intruder inside the window, the server lines as pre-registered"


def door_numbers(rec):
    """Per ON demote run: demote_in, completion, in - completion; the tenant's two largest gaps."""
    rows = []
    for r in arm_runs(rec):
        ins = r["server_demote_ms"]
        comps = [float(m.group(1)) for ln in r["server_log_lines"] for m in [RE_COMPLETION.search(ln)] if m]
        gaps = sorted(r["itl_ms"], reverse=True)[:2]
        rows.append({"run": r["run_id"], "demote_in": ins, "completion": comps,
                     "in_minus_completion": [round(a - b, 1) for a, b in zip(ins, comps)],
                     "top_gaps": [round(g, 1) for g in gaps], "gap_sum": round(sum(gaps), 1), "stall": round(r.get("stall_ms", float("nan")), 1),
                     "p50": round(r.get("p50", float("nan")), 2)})
    return rows


def main():
    ev = sys.argv[1]
    recs, adm = {}, {}
    for p in (1, 2):
        for kind in ORDER[p]:
            bd = os.path.join(ev, f"pass{p}", kind)
            logp = os.path.join(bd, "server.log")
            log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
            for mode in (("prime",) if kind == "prime" else ("demote", "promote")):
                path = os.path.join(bd, mode, "receipt.json")
                if not os.path.exists(path):
                    print(f"  pass{p}/{kind}/{mode}: NO RECEIPT (inadmissible)")
                    adm[(p, kind, mode)] = False
                    continue
                rec = json.load(open(path))
                rep = replay(path)
                ok, why = admissible(kind, mode, rec, log)
                recs[(p, kind, mode)] = rec
                adm[(p, kind, mode)] = ok and rep
                land = landed(rec)
                tt = sorted({r["tenant_tokens"] for r in rec["runs"]})
                print(rec["rule_line"])
                print(f"  pass{p}/{kind}/{mode}: replay={'PASS' if rep else 'FAIL'} admissible={ok and rep} ({why}); "
                      f"tenant_tokens={tt} tenant_max_tokens={rec.get('tenant_max_tokens')} landed={sum(land)}/{len(land)} "
                      f"intruder_wall_ms={rec['summary']['intruder_wall_ms']} "
                      f"tenant_wall_ms_median={statistics.median([r['tenant_wall_ms'] for r in rec['runs'] if r.get('tenant_wall_ms')]):.0f}")
    verdict = []
    pooled = {}
    for p in (1, 2):
        if (p, "prime", "prime") in recs:
            pm, pu = med_iqr(stalls(recs[(p, "prime", "prime")]))
            pooled.setdefault("prime", []).extend(stalls(recs[(p, "prime", "prime")]))
            a = adm[(p, "prime", "prime")]
            s = recs[(p, "prime", "prime")]["summary"]
            print(f"DAY35 STALL pass={p} class=prime stall_median={pm:.1f} iqr={pu:.1f} n_per_order=5 pooled=10 "
                  f"idle_p50={s['idle']['p50']:.1f} idle_p99={s['idle']['p99']:.1f} arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} "
                  f"-> {'admissible' if a else 'inadmissible'}")
            verdict.append(f"prime pass{p} {pm:.1f} (iqr {pu:.1f}) {'admissible' if a else 'inadmissible'}")
        for cls in ("demote", "promote"):
            vals = {}
            for kind in ("off", "on"):
                if (p, kind, cls) not in recs:
                    continue
                m, u = med_iqr(stalls(recs[(p, kind, cls)]))
                vals[kind] = (m, u)
                pooled.setdefault(f"{cls}-{kind}", []).extend(stalls(recs[(p, kind, cls)]))
                s = recs[(p, kind, cls)]["summary"]
                print(f"DAY35 STALL pass={p} class={cls} arm={kind} stall_median={m:.1f} iqr={u:.1f} idle_p50={s['idle']['p50']:.1f} "
                      f"idle_p99={s['idle']['p99']:.1f} arm_p99={s['arm']['p99']:.1f} arm_max={s['arm']['max']:.1f} "
                      f"server_{cls}_ms={[round(x, 1) for x in s[f'server_{cls}_ms']]} -> {'admissible' if adm[(p, kind, cls)] else 'inadmissible'}")
            if "off" in vals and "on" in vals:
                d = vals["on"][0] - vals["off"][0]
                unc = math.sqrt(vals["on"][1] ** 2 + vals["off"][1] ** 2)
                a = adm[(p, "off", cls)] and adm[(p, "on", cls)]
                c = classify(d, unc) if a else "inadmissible"
                print(f"DAY35 STALL pass={p} class={cls} on_minus_off={d:+.1f} unc={unc:.1f} -> {c}")
                verdict.append(f"{cls} pass{p} off {vals['off'][0]:.1f} on {vals['on'][0]:.1f} on-off {d:+.1f} (unc {unc:.1f}) {c}")
    for k, xs in pooled.items():
        m, u = med_iqr(xs)
        print(f"DAY35 STALL pooled arm={k} N={len(xs)} stall_median={m:.1f} iqr={u:.1f} (context: both passes)")
    # the day-27 attribution on this card
    for p in (1, 2):
        if (p, "on", "demote") in recs:
            rows = door_numbers(recs[(p, "on", "demote")])
            imc = [x for r in rows for x in r["in_minus_completion"]]
            ins = [x for r in rows for x in r["demote_in"]]
            comps = [x for r in rows for x in r["completion"]]
            off_ins = recs[(p, "off", "demote")]["summary"]["server_demote_ms"] if (p, "off", "demote") in recs else []
            print(f"DAY35 ATTRIBUTION pass={p} ON demote: demote_in median={statistics.median(ins):.1f} (N={len(ins)}) "
                  f"completion median={statistics.median(comps):.1f} (N={len(comps)}) "
                  f"in_minus_completion median={statistics.median(imc):.1f} min={min(imc):.1f} max={max(imc):.1f} (N={len(imc)}); "
                  f"OFF demote_in median={statistics.median(off_ins):.1f} (N={len(off_ins)})" if off_ins and imc else
                  f"DAY35 ATTRIBUTION pass={p} ON demote: no completion line to subtract (no `demote published off the tick` in the run lines)")
            for r in rows:
                print(f"  pass{p} on/demote run {r['run']}: demote_in={r['demote_in']} completion={r['completion']} "
                      f"in-completion={r['in_minus_completion']} top_gaps={r['top_gaps']} gap_sum={r['gap_sum']} stall={r['stall']} p50={r['p50']}")
        if (p, "on", "promote") in recs:
            comps = [float(m.group(1)) for r in arm_runs(recs[(p, "on", "promote")]) for ln in r["server_log_lines"]
                     for m in [RE_PROMOTE_COMPLETION.search(ln)] if m]
            ins = recs[(p, "on", "promote")]["summary"]["server_promote_ms"]
            dem = recs[(p, "on", "promote")]["summary"]["server_demote_ms"]
            if comps and ins:
                print(f"DAY35 ATTRIBUTION pass={p} ON promote: promote_in median={statistics.median(ins):.1f} (N={len(ins)}) "
                      f"completion median={statistics.median(comps):.1f} (N={len(comps)}) inline demote_in median={statistics.median(dem):.1f} (N={len(dem)})")
    print("DAY35 STALL VERDICT: " + "; ".join(verdict) + f"; admissible={all(adm.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
