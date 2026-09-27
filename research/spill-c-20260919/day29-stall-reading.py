#!/usr/bin/env python3
"""Day 29 reading of decision cell (i) of Move 1 (DAY29.md pre-registration, fixed before the run).

Layout: <ev>/o1/bNN-x|y/{demote,promote}/receipt.json, <ev>/o2/..., <ev>/dry-y/demote/receipt.json.
Arm X = today's tree (the copy-stream program), arm Y = the day-16 tree 1646d421b (the owner-stream program),
door ON in both, the day-16 script's `on` boot. Per class, per arm, per order: the five boots' stall_median
(the harness's rule-line figure) -> cell_median (median of the five), min, max, IQR (p75 - p25 of the five);
pooled over both orders (median of the ten). y_minus_x per class per order with unc in quadrature;
`isolated` when |y_minus_x| > unc, else `under_resolution`. Admissibility per receipt: STALL REPLAY PASS,
errors=0, tenant_text_identical, demote receipt >= 9 server_demote_ms, promote receipt == 10 server_promote_ms,
no `demote failed|promote failed|promote refused|TIER DISABLED` in the boot's server.log. Cell (i)'s clause on
arm X: pooled cell median <= the largest idle p99 over the sitting's boots of that class. Cross-sitting
figures are printed as context only. No threshold beyond the cell's own IQR; nothing here is tuned.

    day29-stall-reading.py <ev_dir>
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spill-a-20260919", "stall_cell.py")
BAD = re.compile(r"demote failed|promote failed|promote refused|TIER DISABLED")
CONTEXT = {  # same box, other sittings; context beside the same-window pair, never the reading
    "demote": "A day 16 ON 193.5 (owner stream); A day 17 149.6, A day 18 149.7, C day 23 149.5 (copy stream)",
    "promote": "A day 16 ON 162.8 (owner stream); A day 18 run 2 81.9, C day 23 81.9 (copy stream)",
}
CLASSES = ("demote", "promote")


def replay(path):
    rep = subprocess.run([sys.executable, HARNESS, "--replay", path], capture_output=True, text=True)
    return "STALL REPLAY: PASS" in rep.stdout


def iqr(xs):
    if len(xs) < 2:
        return float("nan")
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def boots(ev, order):
    d = os.path.join(ev, order)
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))


def main():
    ev = sys.argv[1]
    # the dry boot: recorded, part of no quantity
    dry = os.path.join(ev, "dry-y", "demote", "receipt.json")
    if os.path.exists(dry):
        rec = json.load(open(dry))
        print(f"DRY BOOT arm=y replay={'PASS' if replay(dry) else 'FAIL'} errors={len(rec['summary']['errors'])} "
              f"rule_line: {rec['rule_line']}")
    else:
        print("DRY BOOT arm=y: no receipt (the boot did not run or failed; see ev/DRY-FAILED)")
    per = {}  # (cls, arm, order) -> list of (boot, stall_median, arm_p99, arm_max, idle_p99, admissible)
    all_admissible = True
    for order in ("o1", "o2"):
        for b in boots(ev, order):
            arm = b.rsplit("-", 1)[-1]
            bd = os.path.join(ev, order, b)
            log = open(os.path.join(bd, "server.log"), errors="replace").read() if os.path.exists(os.path.join(bd, "server.log")) else ""
            bad = BAD.findall(log)
            for cls in CLASSES:
                p = os.path.join(bd, cls, "receipt.json")
                if not os.path.exists(p):
                    print(f"  {order}/{b}/{cls}: NO RECEIPT (inadmissible)")
                    all_admissible = False
                    continue
                rec = json.load(open(p))
                s = rec["summary"]
                ok = replay(p)
                ident = len(s["tenant_text_shas"]) == 1
                n_dem, n_pro = len(s["server_demote_ms"]), len(s["server_promote_ms"])
                count_ok = (n_dem >= 9) if cls == "demote" else (n_pro == 10)
                adm = ok and not s["errors"] and ident and count_ok and not bad
                all_admissible &= adm
                sm = s["stall_ms"]["median"]
                print(rec["rule_line"])
                print(f"  {order}/{b}/{cls} arm={arm} replay={'PASS' if ok else 'FAIL'} errors={len(s['errors'])} "
                      f"tenant_text_identical={ident} server_demote_n={n_dem} server_promote_n={n_pro} "
                      f"bad_lines={len(bad)} admissible={adm} stall_median={sm:.1f} arm_p99={s['arm']['p99']:.1f} "
                      f"arm_max={s['arm']['max']:.1f} idle_p99={s['idle']['p99']:.1f}")
                per.setdefault((cls, arm, order), []).append((b, sm, s["arm"]["p99"], s["arm"]["max"], s["idle"]["p99"], adm))
    print(f"ADMISSIBLE all_receipts={all_admissible}")
    verdicts = {}
    for cls in CLASSES:
        idle_p99_sitting = max((r[4] for k, v in per.items() if k[0] == cls for r in v), default=float("nan"))
        pooled = {}
        for arm in ("x", "y"):
            pooled[arm] = []
            for order in ("o1", "o2"):
                rows = [r for r in per.get((cls, arm, order), []) if r[5]]
                vals = [r[1] for r in rows]
                pooled[arm] += vals
                if vals:
                    print(f"DAY29 CELL(i) class={cls} arm={arm} order={order} N={len(vals)} boots={[r[0] for r in rows]} "
                          f"stall_medians={[round(v, 1) for v in vals]} cell_median={statistics.median(vals):.1f} "
                          f"min={min(vals):.1f} max={max(vals):.1f} IQR={iqr(vals):.1f} "
                          f"arm_p99s={[round(r[2], 1) for r in rows]} arm_maxs={[round(r[3], 1) for r in rows]}")
                else:
                    print(f"DAY29 CELL(i) class={cls} arm={arm} order={order} N=0 (no admissible boot)")
            if pooled[arm]:
                print(f"DAY29 CELL(i) class={cls} arm={arm} pooled N={len(pooled[arm])} cell_median={statistics.median(pooled[arm]):.1f} "
                      f"min={min(pooled[arm]):.1f} max={max(pooled[arm]):.1f} IQR={iqr(pooled[arm]):.1f}")
        for order in ("o1", "o2"):
            xs = [r[1] for r in per.get((cls, "x", order), []) if r[5]]
            ys = [r[1] for r in per.get((cls, "y", order), []) if r[5]]
            if xs and ys:
                d = statistics.median(ys) - statistics.median(xs)
                unc = math.sqrt((iqr(xs) if len(xs) > 1 else 0) ** 2 + (iqr(ys) if len(ys) > 1 else 0) ** 2)
                cl = "isolated" if abs(d) > unc else "under_resolution"
                print(f"DAY29 CELL(i) class={cls} order={order} y_minus_x={d:+.1f} unc={unc:.1f} -> {cl} "
                      f"(Y owner stream {statistics.median(ys):.1f} N={len(ys)}; X copy stream {statistics.median(xs):.1f} N={len(xs)})")
        if pooled["x"]:
            xm = statistics.median(pooled["x"])
            met = xm <= idle_p99_sitting
            verdicts[cls] = met
            print(f"DAY29 CELL(i) CLAUSE class={cls} stall_median(second stream)={xm:.1f} idle_p99_sitting={idle_p99_sitting:.1f} "
                  f"-> {'clause_met' if met else 'clause_not_met'}")
        else:
            verdicts[cls] = False
            print(f"DAY29 CELL(i) CLAUSE class={cls}: no admissible arm X boot -> clause_not_read")
        print(f"  context (other sittings on this box, not the reading): {cls}: {CONTEXT[cls]}")
    overall = all(verdicts.get(c, False) for c in CLASSES) and all_admissible
    print(f"DAY29 CELL(i) CLAUSE: {'MET' if overall else 'NOT MET'} (demote={verdicts.get('demote')} promote={verdicts.get('promote')} "
          f"admissible={all_admissible}); executed-not-qualified")


if __name__ == "__main__":
    main()
