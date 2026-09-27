#!/usr/bin/env python3
"""Day 43 reader for cell `resid` (research/spill-c-20260919/DAY43.md section 1, registered before any I6 code).

Integrity clauses, per-arm readings, the no-regression clause (I6D against BASE) and the budget reading (I6G against
BASE). Every figure is a median over the named runs of lines the binaries printed. No clause moves after a run.

usage: day43-resid.py <cell-dir> [--rig NAME]
"""
import importlib.util
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N_TOKENS = 32
ARMS = ["off", "base", "i6d", "i6g"]
DOOR = ["base", "i6d", "i6g"]
STAGE_MS = ["demand_ns", "verify_ns", "step_ns", "pread_ns", "stage_ns", "alloc_ns", "drain_ns", "enqueue_ns",
            "validate_ns", "finish_ns", "collect_ns", "publish_ns", "miss_total_ns"]
STAGE_N = ["gpu_misses", "host_hits", "host_misses", "reads"]


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        run["trace_lines"] = text.count("[expert-host-slru] key=")
        run["trace_misses"] = text.count(" hit=false ")
        run["plan_line"] = "[experts-via-tier] host_bank_plan" in text
        runs[log.stem] = run
    fails = []

    def check(ok, name):
        if not ok:
            fails.append(name)

    check(len(runs) == 40, f"runs={len(runs)} (40 expected)")
    for label, r in runs.items():
        check(r["exit"] == 0, f"{label} exit={r['exit']}")
        check(r.get("match", "").endswith("MATCH"), f"{label} no MATCH")
        check(r.get("slots") == 9986, f"{label} slots={r.get('slots')}")
        if r["arm"] == "off":
            check(r.get("door_lines", 0) == 0, f"{label} OFF printed a door line")
            continue
        check("installed_t" in r and "physical_reads" in r, f"{label} no installed/physical_reads")
        check(all(p in r["stages"] for p in ("gate", "generate", "warm", "window")), f"{label} stage lines")
        check(r["plan_line"] == (r["arm"] != "base"), f"{label} plan line presence")
        close = r["stages"].get("close", {})
        demands = close.get("host_hits", 0) + close.get("host_misses", 0)
        check(r["physical_reads"] == r["trace_misses"], f"{label} physical_reads != hit=false lines")
        check(r["trace_lines"] == demands, f"{label} trace lines {r['trace_lines']} != host demands {demands}")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY43 RESID CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    off = {o: statistics.median(values("off", "window_s", o)) for o in (None, "o1", "o2")}
    for arm in DOOR:
        cost = {o: (statistics.median(values(arm, "window_s", o)) - off[o]) * 1000 / N_TOKENS for o in (None, "o1", "o2")}
        gen = (statistics.median(values(arm, "gen_s")) - statistics.median(values("off", "gen_s"))) * 1000 / N_TOKENS
        ons = [r for r in runs.values() if r["arm"] == arm]
        per = {}
        for key in STAGE_MS + STAGE_N:
            per[key] = statistics.median(
                (r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS for r in ons)
        print(f"DAY43 ARM {arm} window_door_ms_per_token pooled={cost[None]:.2f} o1={cost['o1']:.2f} o2={cost['o2']:.2f}"
              f" gen_door_ms_per_token={gen:.2f} window_s median={statistics.median(values(arm, 'window_s')):.3f}"
              f" iqr={iqr(values(arm, 'window_s')):.3f} | per token: "
              + " ".join(f"{k[:-3]}={per[k] / 1e6:.3f}" for k in STAGE_MS)
              + " " + " ".join(f"{k}={per[k]:.1f}" for k in STAGE_N))

    def compare(a, b):
        """(pooled diff, o1 diff, o2 diff, noise) of median window seconds, arm a minus arm b."""
        diff = {o: statistics.median(values(a, "window_s", o)) - statistics.median(values(b, "window_s", o))
                for o in (None, "o1", "o2")}
        noise = max(iqr(values(a, "window_s")), iqr(values(b, "window_s")))
        return diff, noise

    diff, noise = compare("i6d", "base")
    ok = all(diff[o] <= noise for o in (None, "o1", "o2"))
    print(f"DAY43 CLAUSE no_regression i6d_minus_base pooled={diff[None]:+.3f} o1={diff['o1']:+.3f}"
          f" o2={diff['o2']:+.3f} noise={noise:.3f} rule <=noise pooled and both orders -> {'PASS' if ok else 'FAIL'}")
    diff, noise = compare("i6g", "base")
    hits = statistics.median(
        (r["stages"]["window"].get("host_hits", 0) - r["stages"]["warm"].get("host_hits", 0)) / N_TOKENS
        for r in runs.values() if r["arm"] == "i6g")
    if hits > 0 and all(diff[o] < -noise for o in ("o1", "o2")):
        verdict = "resid_budget_helps"
    elif all(diff[o] > noise for o in ("o1", "o2")):
        verdict = "resid_budget_hurts"
    else:
        verdict = "resid_budget_flat"
    print(f"DAY43 READING budget i6g_minus_base pooled={diff[None]:+.3f} o1={diff['o1']:+.3f} o2={diff['o2']:+.3f}"
          f" noise={noise:.3f} host_hits_per_token={hits:.1f} -> {verdict}")
    print(f"DAY43 RESID rig={rig} integrity={integrity} no_regression={'PASS' if ok else 'FAIL'} budget={verdict}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
