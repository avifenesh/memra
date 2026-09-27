#!/usr/bin/env python3
"""WP-A day 26 reading: the acceptance gate of proposal 1 (DAY25.md, lead ruling 36), fixed before the run.

    day26-reading.py <receipts_root>     (the mirrored /root/spill-receipts/a-day26)

Clauses, verbatim from DAY25.md "Proposal 1" and ruling 36 (the numbers in them are day 25's, never re-read here):
 (1) in every ON promote run of the double-park cell `request parked` reads 1, `restore submitted off the tick`
     reads 0 and the typed refusal `restore not routed (contracts door)` reads 1 (100 of 100);
 (2) the request's ON e2e median drops by the re-admission median within the pair's unc: day 25 read
     `on_minus_off` +105.8 / +105.9 and a re-admission median of 90.1, so the expected `on_minus_off` is about
     +15.8 (221.4 to about 131); PASS when |on_minus_off - 15.8| <= unc (this pair's quadrature unc), per order;
 (3) the tenant's ON stall stays within IQR of 149.4 (day 25's ON IQR was 0.1): a move either way is a FINDING,
     reported with the decomposition, never tuned;
 (4) the day-21 restore arm (a hit on an entry NOT promoted this admission) still parks once and lands 100 of 100
     (`request parked` 1, `restore submitted` 1, `restore landed` 1, `restore not routed` 0, per run);
 (5) the hit gate OFF and ON `ALL GREEN`, and (ruling 36) the ON arm's route counts equal day 24's exactly:
     spec-on boot `capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13
     refused_contracts_door=0 restore_refused=0 latched=0`, spec-off boot `capture_submitted=2 capture_published=2
     restore_submitted=3 restore_landed=3`, `30 route submission(s)`, 11 spec-boundary captures with the draft plane.
Readings (not clauses): the identity ON arm's route lines (day 24's tree read promote published 1, restore
submitted 2, parked 3 on this shape; the refusal is expected to read 1, 1, 2 plus one typed line) and the fault
gate's four promote cells (day 24: restore submitted 1 after the second promote; expected 0 plus the typed line).
The double-park pair itself is read by day25-double-park-reading.py over <root>/double-park/ev (its DAY25 labels
name the reader's day, not the cell's); this script re-derives the per-order medians it needs from the receipts.
"""
import importlib.util
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d25", os.path.join(HERE, "day25-double-park-reading.py"))
d25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d25)

NOT_ROUTED = "restore not routed (contracts door)"
EXPECT_ON_MINUS_OFF = 105.85 - 90.1   # +15.75: day 25's e2e delta minus the re-admission median
DAY25_ON_STALL, DAY25_ON_STALL_IQR = 149.4, 0.1
DAY24_HIT_ON = dict(capture_submitted=12, capture_published=12, restore_submitted=13, restore_landed=13,
                    refused_contracts_door=0, restore_refused=0, latched=0)
DAY24_HIT_OFF = dict(capture_submitted=2, capture_published=2, restore_submitted=3, restore_landed=3)
DAY24_ROUTES, DAY24_SPEC_BOUNDARY_DRAFT = 30, 11


def count(lines, key):
    return sum(key in ln for ln in lines)


def load_boots(ev):
    per = {}
    admissible = True
    for order in ("o1", "o2"):
        d = os.path.join(ev, order)
        for b in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            arm = b.rsplit("-", 1)[-1]
            bd = os.path.join(d, b)
            p = os.path.join(bd, "promote", "receipt.json")
            logp = os.path.join(bd, "server.log")
            log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
            if not os.path.exists(p):
                print(f"  {order}/{b}: NO RECEIPT (inadmissible)")
                admissible = False
                continue
            rec = json.load(open(p))
            s = rec["summary"]
            ok = d25.replay(p)
            adm = ok and not s["errors"] and len(s["tenant_text_shas"]) == 1 and len(s["server_promote_ms"]) == 10 \
                and not d25.BAD.findall(log)
            admissible &= adm
            runs = [r for r in rec["runs"] if r["arm"] == "promote"]
            per.setdefault((arm, order), []).append({
                "boot": b, "adm": adm, "stall": s["stall_ms"]["median"], "idle_p50": s["idle"]["p50"],
                "walls": [r["intruder"]["wall_ms"] for r in runs if r.get("intruder") and "wall_ms" in r["intruder"]],
                "runs": runs, "decs": [d25.decompose(r) for r in runs]})
    return per, admissible


def main():
    root = sys.argv[1]
    verdicts = []
    # ---- the double-park cell: clauses (1) (2) (3)
    ev = os.path.join(root, "double-park", "ev")
    per, admissible = load_boots(ev)
    print(f"DAY26 DOUBLE-PARK ADMISSIBLE all_receipts={admissible}")
    on_runs = [r for o in ("o1", "o2") for row in per.get(("on", o), []) if row["adm"] for r in row["runs"]]
    parked = [count(r.get("server_log_lines", []), "request parked") for r in on_runs]
    submitted = [count(r.get("server_log_lines", []), "restore submitted off the tick") for r in on_runs]
    not_routed = [count(r.get("server_log_lines", []), NOT_ROUTED) for r in on_runs]
    c1 = len(on_runs) > 0 and all(p == 1 for p in parked) and all(s == 0 for s in submitted) and all(n == 1 for n in not_routed)
    print(f"DAY26 CLAUSE 1 arm=on N_runs={len(on_runs)} parked_per_run={sorted(set(parked))} "
          f"restore_submitted_per_run={sorted(set(submitted))} not_routed_per_run={sorted(set(not_routed))} "
          f"runs_with_parked_1_submitted_0_not_routed_1={sum(1 for p, s, n in zip(parked, submitted, not_routed) if (p, s, n) == (1, 0, 1))} "
          f"-> {'PASS' if c1 else 'FAIL'}")
    verdicts.append(c1)
    off_runs = [r for o in ("o1", "o2") for row in per.get(("off", o), []) if row["adm"] for r in row["runs"]]
    print(f"DAY26 CLAUSE 1 context arm=off N_runs={len(off_runs)} "
          f"parked_per_run={sorted(set(count(r.get('server_log_lines', []), 'request parked') for r in off_runs))} "
          f"not_routed_per_run={sorted(set(count(r.get('server_log_lines', []), NOT_ROUTED) for r in off_runs))}")
    one = next((r for r in on_runs if count(r.get("server_log_lines", []), NOT_ROUTED)), None)
    if one:
        for ln in one["server_log_lines"]:
            if NOT_ROUTED in ln:
                print(f"DAY26 TYPED LINE (one ON run, verbatim): {ln}")
                break
    c2_all, c3_all = True, True
    for order in ("o1", "o2"):
        on = [r for r in per.get(("on", order), []) if r["adm"]]
        off = [r for r in per.get(("off", order), []) if r["adm"]]
        if not on or not off:
            print(f"DAY26 CLAUSE 2/3 order={order}: N=0 on one arm (no admissible boot)")
            c2_all = c3_all = False
            continue
        on_w = [w for r in on for w in r["walls"]]
        off_w = [w for r in off for w in r["walls"]]
        d = statistics.median(on_w) - statistics.median(off_w)
        unc = math.sqrt(d25.iqr(on_w) ** 2 + d25.iqr(off_w) ** 2)
        c2 = abs(d - EXPECT_ON_MINUS_OFF) <= unc
        c2_all &= c2
        print(f"DAY26 CLAUSE 2 e2e order={order} on_median={statistics.median(on_w):.1f} off_median={statistics.median(off_w):.1f} "
              f"on_minus_off={d:+.1f} unc={unc:.1f} expected={EXPECT_ON_MINUS_OFF:+.1f} (day 25: +105.85 minus 90.1) "
              f"|d-expected|={abs(d - EXPECT_ON_MINUS_OFF):.1f} -> {'PASS' if c2 else 'FAIL'}")
        on_s = [r["stall"] for r in on]
        off_s = [r["stall"] for r in off]
        m = statistics.median(on_s)
        within = abs(m - DAY25_ON_STALL) <= DAY25_ON_STALL_IQR
        c3_all &= within
        ds = m - statistics.median(off_s)
        us = math.sqrt(d25.iqr(on_s) ** 2 + d25.iqr(off_s) ** 2)
        print(f"DAY26 CLAUSE 3 stall order={order} on_stall_medians={[round(v, 1) for v in on_s]} on_cell_median={m:.1f} IQR={d25.iqr(on_s):.1f} "
              f"off_cell_median={statistics.median(off_s):.1f} on_minus_off={ds:+.1f} unc={us:.1f} -> {'isolated' if abs(ds) > us else 'under_resolution'} | "
              f"against day 25's 149.4 (IQR 0.1): {m - DAY25_ON_STALL:+.1f} -> "
              f"{'within IQR' if within else 'FINDING (moved, reported with the decomposition, not tuned)'}")
    verdicts.append(c2_all)
    # the decomposition of the ON runs, for the finding either way
    decs = [x for o in ("o1", "o2") for row in per.get(("on", o), []) if row["adm"] for x in row["decs"]]
    idle = d25.med([row["idle_p50"] for o in ("o1", "o2") for row in per.get(("on", o), []) if row["adm"]])
    if decs:
        dem = [x["demote_in"] - x["demote_completion"] for x in decs if x.get("demote_in") is not None and x.get("demote_completion") is not None]
        top1 = [x["top2"][0] for x in decs if len(x["top2"]) > 0]
        top2 = [x["top2"][1] for x in decs if len(x["top2"]) > 1]
        print(f"DAY26 DECOMPOSITION arm=on N_runs={len(decs)} parked_per_run={sorted(set(x['parked'] for x in decs))} "
              f"restore_readmission={[x.get('restore_readmission') for x in decs if x.get('restore_readmission') is not None][:3]} "
              f"promote_completion median={d25.med([x.get('promote_completion') for x in decs]):.1f} "
              f"promote_in median={d25.med([x.get('promote_in') for x in decs]):.1f} "
              f"demote_completion median={d25.med([x.get('demote_completion') for x in decs]):.1f} "
              f"demote_in median={d25.med([x.get('demote_in') for x in decs]):.1f} demote_in-completion median={d25.med(dem):.1f} "
              f"idle_p50(tick)={idle:.2f} | tenant top gaps: largest median={d25.med(top1):.1f} second median={d25.med(top2):.1f} "
              f"sum median={d25.med([a + b for a, b in zip(top1, top2)]):.1f}")
    decs_off = [x for o in ("o1", "o2") for row in per.get(("off", o), []) if row["adm"] for x in row["decs"]]
    if decs_off:
        top1 = [x["top2"][0] for x in decs_off if len(x["top2"]) > 0]
        top2 = [x["top2"][1] for x in decs_off if len(x["top2"]) > 1]
        print(f"DAY26 DECOMPOSITION arm=off N_runs={len(decs_off)} parked_per_run={sorted(set(x['parked'] for x in decs_off))} "
              f"promote_in median={d25.med([x.get('promote_in') for x in decs_off]):.1f} demote_in median={d25.med([x.get('demote_in') for x in decs_off]):.1f} "
              f"| tenant top gaps: largest median={d25.med(top1):.1f} second median={d25.med(top2):.1f}")
    # ---- clause (4): the restore arm
    rp = os.path.join(root, "restore-arm", "ev", "restore", "receipt.json")
    if os.path.exists(rp):
        rec = json.load(open(rp))
        ok = d25.replay(rp)
        runs = [r for r in rec["runs"] if r["arm"] == "restore"]
        L = [r.get("server_log_lines", []) for r in runs]
        p1 = [count(l, "request parked") for l in L]
        s1 = [count(l, "restore submitted off the tick") for l in L]
        l1 = [count(l, "restore landed off the tick") for l in L]
        n0 = [count(l, NOT_ROUTED) for l in L]
        cached = sorted(set(r["intruder"].get("cached_tokens") for r in runs if r.get("intruder") and "error" not in r["intruder"]))
        good = sum(1 for a, b, c, d in zip(p1, s1, l1, n0) if (a, b, c, d) == (1, 1, 1, 0))
        c4 = ok and len(runs) == 100 and good == 100 and not rec["summary"]["errors"]
        print(f"DAY26 CLAUSE 4 restore-arm arm=on replay={'PASS' if ok else 'FAIL'} N_runs={len(runs)} errors={len(rec['summary']['errors'])} "
              f"parked_per_run={sorted(set(p1))} submitted_per_run={sorted(set(s1))} landed_per_run={sorted(set(l1))} not_routed_per_run={sorted(set(n0))} "
              f"runs_parked_1_submitted_1_landed_1={good} cached_tokens={cached} stall_median={rec['summary']['stall_ms']['median']:.1f} -> {'PASS' if c4 else 'FAIL'}")
        print(rec["rule_line"])
        verdicts.append(c4)
    else:
        print("DAY26 CLAUSE 4 restore-arm: NO RECEIPT -> FAIL")
        verdicts.append(False)
    # ---- clause (5): the hit gate, both arms, and its ON counts against day 24
    g = os.path.join(root, "gates")
    c5 = True
    for arm in ("off", "on"):
        lp = os.path.join(g, f"hitgate-{arm}.log")
        txt = open(lp, errors="replace").read() if os.path.exists(lp) else ""
        verdict = next((ln.strip() for ln in txt.splitlines() if "SPEC-ON-CACHE-HIT GATE:" in ln), "NO VERDICT LINE")
        oks = len(re.findall(r"^\s*ok: ", txt, re.M))
        fails = len(re.findall(r"^\s*FAIL: ", txt, re.M))
        green = "ALL GREEN" in verdict
        c5 &= green
        print(f"DAY26 CLAUSE 5 hitgate-{arm}: {verdict} (ok={oks} FAIL={fails})")
        if arm == "on":
            census = [ln.strip() for ln in txt.splitlines() if "capture_submitted=" in ln]
            routes = re.search(r"(\d+) route submission\(s\)", txt)
            for ln in census:
                print(f"DAY26 CLAUSE 5 hitgate-on census: {ln}")
            kv = [dict(re.findall(r"(\w+)=(\d+)", ln)) for ln in census]
            spec_on = next((d for d in kv if d.get("restore_submitted") == "13"), None)
            spec_off = next((d for d in kv if d.get("restore_submitted") == "3"), None)
            same_on = spec_on is not None and all(int(spec_on.get(k, -1)) == v for k, v in DAY24_HIT_ON.items())
            same_off = spec_off is not None and all(int(spec_off.get(k, -1)) == v for k, v in DAY24_HIT_OFF.items())
            n_routes = int(routes.group(1)) if routes else -1
            slog = os.path.join(g, "hitgate-on", "qwen-on-server.log")
            stxt = open(slog, errors="replace").read() if os.path.exists(slog) else ""
            sb_draft = sum(1 for ln in stxt.splitlines() if "capture submitted off the tick (spec-boundary)" in ln and "draft plane" in ln)
            nr = stxt.count(NOT_ROUTED) + (open(os.path.join(g, "hitgate-on", "qwen-off-server.log"), errors="replace").read().count(NOT_ROUTED) if os.path.exists(os.path.join(g, "hitgate-on", "qwen-off-server.log")) else 0)
            counts_same = same_on and same_off and n_routes == DAY24_ROUTES and sb_draft == DAY24_SPEC_BOUNDARY_DRAFT and nr == 0
            c5 &= counts_same
            print(f"DAY26 CLAUSE 5 hitgate-on counts against day 24: spec_on_census_equal={same_on} spec_off_census_equal={same_off} "
                  f"route_submissions={n_routes} (day 24: {DAY24_ROUTES}) spec_boundary_captures_with_draft_plane={sb_draft} (day 24: {DAY24_SPEC_BOUNDARY_DRAFT}) "
                  f"not_routed_lines={nr} (must be 0: the gate has no promote-then-hit shape) -> {'PASS' if counts_same else 'FAIL'}")
    verdicts.append(c5)
    # ---- readings: identity ON arm's route lines; the fault gate's promote cells; the other gates' verdicts
    for cell in ("identity-default-on", "identity-plain-on"):
        lp = os.path.join(g, cell, "host-on-server.log")
        if os.path.exists(lp):
            t = open(lp, errors="replace").read()
            print(f"DAY26 READING {cell} route lines: promote_published={t.count('promote published off the tick')} "
                  f"restore_submitted={t.count('restore submitted off the tick')} restore_landed={t.count('restore landed off the tick')} "
                  f"parked={t.count('request parked')} not_routed={t.count(NOT_ROUTED)} hits={t.count('[prefix-cache] hit: ')} "
                  f"(day 24's tree: 1, 2, 2, 3, 0, 2)")
    for cell in ("promote-presubmit", "promote-postpublish", "promote-readyview", "promote-reject"):
        lp = os.path.join(g, "contract-fault", f"{cell}-server.log")
        if os.path.exists(lp):
            t = open(lp, errors="replace").read()
            print(f"DAY26 READING fault {cell}: promote_published={t.count('promote published off the tick')} "
                  f"restore_submitted={t.count('restore submitted off the tick')} parked={t.count('request parked')} not_routed={t.count(NOT_ROUTED)}")
    for name, key in (("identity-default-off", "KV-HOST-SPILL IDENTITY GATE:"), ("identity-default-on", "KV-HOST-SPILL IDENTITY GATE:"),
                      ("identity-plain-off", "KV-HOST-SPILL IDENTITY GATE:"), ("identity-plain-on", "KV-HOST-SPILL IDENTITY GATE:"),
                      ("failure-off", "KV-HOST-SPILL FAILURE GATE:"), ("failure-on", "KV-HOST-SPILL FAILURE GATE:"),
                      ("contract-fault", "KV-HOST-CONTRACT-FAULT GATE:"), ("twin-off", "->"), ("twin-on", "->")):
        lp = os.path.join(g, f"{name}.log")
        if os.path.exists(lp):
            t = open(lp, errors="replace").read()
            v = [ln.strip() for ln in t.splitlines() if key in ln]
            oks = len(re.findall(r"^\s*ok: ", t, re.M))
            print(f"DAY26 GATE {name}: {v[-1] if v else 'NO VERDICT LINE'} (ok={oks})")
    print(f"DAY26 ACCEPTANCE clauses (1) (2) (4) (5) = {['PASS' if v else 'FAIL' for v in verdicts]}; clause (3) is a finding either way (see CLAUSE 3 lines)")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
