#!/usr/bin/env python3
"""WP-A day 25 reading of the double-park cell (DAY25.md Task 1 pre-registration, fixed before the run).

Layout: <ev>/o1/bNN-on|off/promote/receipt.json, <ev>/o2/... Arm ON = today's tree with the contracts door
(MEMRA_KV_HOST_CONTRACTS=1: the promote parks, the hit's restore parks again), arm OFF = the same tree with the
door off (the pre-door program: on-tick promote, on-tick device-hit copy). Per arm per order: the five boots'
stall_median (the harness's rule-line figure) -> cell_median, min, max, IQR; the request's end-to-end latency
(intruder.wall_ms, 50 runs) -> median, p95, IQR; on_minus_off per order with unc in quadrature, `isolated` when
|d| > unc, else `under_resolution`; pooled over both orders as context. Admissibility per receipt: STALL REPLAY
PASS, errors=0, tenant text identical, promote receipt == 10 server_promote_ms, no
`demote failed|promote failed|promote refused|TIER DISABLED` in the boot's server.log.

The decomposition (a), from the ON receipts' server_log_lines per promote run: `request parked` count, the
promote's submission-to-completion and `in` ms, the demote's submission-to-completion and `in` ms, the restore's
submission-to-completion and to-re-admission; the slack (re-admission minus completion); the residual
re-admission - (idle p50 + (demote in - demote completion)). From the tenant's ITL series per arm run: the two
largest gaps and their sum. Nothing here is tuned; no threshold beyond the cell's own IQR.

    day25-double-park-reading.py <ev_dir>
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stall_cell.py")
BAD = re.compile(r"demote failed|promote failed|promote refused|TIER DISABLED")
CONTEXT = "C day 29 arm X (door ON, same shape) 149.6 / 149.7; A day 18 run 2 and C day 23 (door ON) 81.9; day-16 OFF 85.0, C day 23 OFF 85.2"


def replay(path):
    rep = subprocess.run([sys.executable, HARNESS, "--replay", path], capture_output=True, text=True)
    return "STALL REPLAY: PASS" in rep.stdout


def iqr(xs):
    if len(xs) < 2:
        return float("nan")
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def ms_after(line, key):
    """The float that follows `key` in `line`, up to `ms`."""
    i = line.find(key)
    if i < 0:
        return None
    try:
        return float(line[i + len(key):].split("ms")[0].strip())
    except ValueError:
        return None


def ms_before(line, key):
    """The float that precedes `key` (`..., 89.9ms from submission to completion`)."""
    i = line.find(key)
    if i < 0:
        return None
    head = line[:i].rstrip()
    j = head.rfind(" ")
    tok = head[j + 1:] if j >= 0 else head
    tok = tok.split(",")[-1]
    try:
        return float(tok.replace("ms", ""))
    except ValueError:
        return None


def decompose(run):
    lines = run.get("server_log_lines", [])
    d = {"parked": sum("request parked" in ln for ln in lines)}
    for ln in lines:
        if "promote published off the tick" in ln:
            d["promote_completion"] = ms_before(ln, "ms from submission to completion")
        elif "[prefix-host] promote: " in ln:
            d["promote_in"] = ms_after(ln, " in ")
        elif "demote published off the tick" in ln:
            d["demote_completion"] = ms_before(ln, "ms from submission to completion")
        elif "[prefix-host] demote: " in ln:
            d["demote_in"] = ms_after(ln, " in ")
        elif "restore landed off the tick" in ln:
            d["restore_completion"] = ms_before(ln, "ms from submission to completion")
            d["restore_readmission"] = ms_before(ln, "ms to re-admission")
    itl = sorted(run.get("itl_ms", []), reverse=True)
    d["top2"] = itl[:2]
    return d


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def main():
    ev = sys.argv[1]
    per = {}  # (arm, order) -> list of dict per boot
    all_admissible = True
    for order in ("o1", "o2"):
        d = os.path.join(ev, order)
        for b in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            arm = b.rsplit("-", 1)[-1]
            bd = os.path.join(ev, order, b)
            logp = os.path.join(bd, "server.log")
            log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
            bad = BAD.findall(log)
            p = os.path.join(bd, "promote", "receipt.json")
            if not os.path.exists(p):
                print(f"  {order}/{b}: NO RECEIPT (inadmissible)")
                all_admissible = False
                continue
            rec = json.load(open(p))
            s = rec["summary"]
            ok = replay(p)
            ident = len(s["tenant_text_shas"]) == 1
            n_pro = len(s["server_promote_ms"])
            adm = ok and not s["errors"] and ident and n_pro == 10 and not bad
            all_admissible &= adm
            arm_runs = [r for r in rec["runs"] if r["arm"] == "promote"]
            walls = [r["intruder"]["wall_ms"] for r in arm_runs if r.get("intruder") and "wall_ms" in r["intruder"]]
            decs = [decompose(r) for r in arm_runs]
            row = {"boot": b, "stall": s["stall_ms"]["median"], "arm_p99": s["arm"]["p99"], "arm_max": s["arm"]["max"],
                   "idle_p50": s["idle"]["p50"], "idle_p99": s["idle"]["p99"], "walls": walls, "decs": decs, "adm": adm,
                   "promote_in": s["server_promote_ms"], "demote_in": s["server_demote_ms"]}
            print(rec["rule_line"])
            print(f"  {order}/{b} arm={arm} replay={'PASS' if ok else 'FAIL'} errors={len(s['errors'])} "
                  f"tenant_text_identical={ident} server_promote_n={n_pro} bad_lines={len(bad)} admissible={adm} "
                  f"stall_median={row['stall']:.1f} arm_p99={row['arm_p99']:.1f} arm_max={row['arm_max']:.1f} "
                  f"idle_p50={row['idle_p50']:.2f} idle_p99={row['idle_p99']:.1f} "
                  f"wall_median={med(walls):.1f} parked_per_run={sorted(set(x['parked'] for x in decs))}")
            per.setdefault((arm, order), []).append(row)
    print(f"ADMISSIBLE all_receipts={all_admissible}")
    # (b) the pair
    pooled = {"on": {"stall": [], "wall": []}, "off": {"stall": [], "wall": []}}
    for arm in ("on", "off"):
        for order in ("o1", "o2"):
            rows = [r for r in per.get((arm, order), []) if r["adm"]]
            stalls = [r["stall"] for r in rows]
            walls = [w for r in rows for w in r["walls"]]
            pooled[arm]["stall"] += stalls
            pooled[arm]["wall"] += walls
            if rows:
                print(f"DAY25 DOUBLE-PARK arm={arm} order={order} N_boots={len(stalls)} boots={[r['boot'] for r in rows]} "
                      f"stall_medians={[round(v, 1) for v in stalls]} stall_cell_median={statistics.median(stalls):.1f} "
                      f"min={min(stalls):.1f} max={max(stalls):.1f} IQR={iqr(stalls):.1f} "
                      f"arm_p99s={[round(r['arm_p99'], 1) for r in rows]} idle_p50s={[round(r['idle_p50'], 2) for r in rows]} "
                      f"| request e2e N_runs={len(walls)} wall_median={med(walls):.1f} p95={pct(walls, 0.95):.1f} IQR={iqr(walls):.1f}")
            else:
                print(f"DAY25 DOUBLE-PARK arm={arm} order={order} N=0 (no admissible boot)")
        if pooled[arm]["stall"]:
            print(f"DAY25 DOUBLE-PARK arm={arm} pooled N_boots={len(pooled[arm]['stall'])} "
                  f"stall_cell_median={statistics.median(pooled[arm]['stall']):.1f} IQR={iqr(pooled[arm]['stall']):.1f} "
                  f"| request e2e N_runs={len(pooled[arm]['wall'])} wall_median={med(pooled[arm]['wall']):.1f} IQR={iqr(pooled[arm]['wall']):.1f}")
    for order in ("o1", "o2"):
        on = [r for r in per.get(("on", order), []) if r["adm"]]
        off = [r for r in per.get(("off", order), []) if r["adm"]]
        if on and off:
            for what, key in (("stall", "stall"), ("e2e", "wall")):
                xs = [r["stall"] for r in on] if key == "stall" else [w for r in on for w in r["walls"]]
                ys = [r["stall"] for r in off] if key == "stall" else [w for r in off for w in r["walls"]]
                d = statistics.median(xs) - statistics.median(ys)
                unc = math.sqrt(iqr(xs) ** 2 + iqr(ys) ** 2)
                cl = "isolated" if abs(d) > unc else "under_resolution"
                print(f"DAY25 DOUBLE-PARK {what} order={order} on_minus_off={d:+.1f} unc={unc:.1f} -> {cl} "
                      f"(on {statistics.median(xs):.1f}, off {statistics.median(ys):.1f})")
    # (a) the decomposition, ON runs
    decs = [x for order in ("o1", "o2") for r in per.get(("on", order), []) if r["adm"] for x in r["decs"]]
    idle_p50 = med([r["idle_p50"] for order in ("o1", "o2") for r in per.get(("on", order), []) if r["adm"]])
    if decs:
        n = len(decs)
        parked = sorted(set(x["parked"] for x in decs))
        rr = [x.get("restore_readmission") for x in decs]
        rc = [x.get("restore_completion") for x in decs]
        slack = [a - b for a, b in zip(rr, rc) if a is not None and b is not None]
        dem_hash = [x["demote_in"] - x["demote_completion"] for x in decs
                    if x.get("demote_in") is not None and x.get("demote_completion") is not None]
        residual = [x["restore_readmission"] - (idle_p50 + (x["demote_in"] - x["demote_completion"])) for x in decs
                    if x.get("restore_readmission") is not None and x.get("demote_in") is not None
                    and x.get("demote_completion") is not None]
        top1 = [x["top2"][0] for x in decs if len(x["top2"]) > 0]
        top2 = [x["top2"][1] for x in decs if len(x["top2"]) > 1]
        print(f"DAY25 DECOMPOSITION arm=on N_runs={n} parked_per_run={parked} "
              f"restore_readmission median={med(rr):.1f} range={rng(rr)[0]:.1f}..{rng(rr)[1]:.1f} "
              f"restore_completion median={med(rc):.1f} slack(readmission-completion) median={med(slack):.2f} max={max(slack) if slack else float('nan'):.2f} "
              f"promote_completion median={med([x.get('promote_completion') for x in decs]):.1f} "
              f"promote_in median={med([x.get('promote_in') for x in decs]):.1f} "
              f"demote_completion median={med([x.get('demote_completion') for x in decs]):.1f} "
              f"demote_in median={med([x.get('demote_in') for x in decs]):.1f} demote_in-completion median={med(dem_hash):.1f} "
              f"idle_p50(tick)={idle_p50:.2f} residual median={med(residual):+.1f} range={rng(residual)[0]:+.1f}..{rng(residual)[1]:+.1f} "
              f"| tenant top gaps: largest median={med(top1):.1f} second median={med(top2):.1f} sum median={med([a + b for a, b in zip(top1, top2)]):.1f}")
    decs_off = [x for order in ("o1", "o2") for r in per.get(("off", order), []) if r["adm"] for x in r["decs"]]
    if decs_off:
        top1 = [x["top2"][0] for x in decs_off if len(x["top2"]) > 0]
        top2 = [x["top2"][1] for x in decs_off if len(x["top2"]) > 1]
        print(f"DAY25 DECOMPOSITION arm=off N_runs={len(decs_off)} parked_per_run={sorted(set(x['parked'] for x in decs_off))} "
              f"promote_in median={med([x.get('promote_in') for x in decs_off]):.1f} "
              f"demote_in median={med([x.get('demote_in') for x in decs_off]):.1f} "
              f"| tenant top gaps: largest median={med(top1):.1f} second median={med(top2):.1f}")
    print(f"CONTEXT (other sittings, same box, never the reading): {CONTEXT}")
    return 0 if all_admissible else 1


if __name__ == "__main__":
    sys.exit(main())
