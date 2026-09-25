#!/usr/bin/env python3
"""Day 45 reader for cell `fill` (research/spill-c-20260919/DAY45.md section 1, registered before any fill code).
Integrity, clause (i) FILL beats I9G in the window, clause (ii) the fill reaches the window, and the gen reading.

usage: day45-fill.py <cell-dir> [--rig NAME]
"""
import importlib.util
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N_TOKENS = 32
FILL_CLOSE = re.compile(r"\[experts-via-tier\] fill (fill_reads=.*)")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


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
        close = FILL_CLOSE.search(text)
        run["fill_close"] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", close.group(1))) if close else None
        runs[log.stem] = run
    fails = []

    def check(ok, name):
        if not ok:
            fails.append(name)

    check(len(runs) == 30, f"runs={len(runs)} (30 expected)")
    for label, r in runs.items():
        check(r["exit"] == 0, f"{label} exit={r['exit']}")
        check(r.get("match", "").endswith("MATCH"), f"{label} no MATCH")
        check(r.get("slots") == 9986, f"{label} slots={r.get('slots')}")
        if r["arm"] == "off":
            check(r.get("door_lines", 0) == 0, f"{label} OFF printed a door line")
            continue
        check("installed_t" in r and "physical_reads" in r, f"{label} no installed/physical_reads")
        check(all(p in r["stages"] for p in ("gate", "generate", "warm", "window")), f"{label} stage lines")
        close = r["stages"].get("close", {})
        demands = close.get("host_hits", 0) + close.get("host_misses", 0)
        check(r["physical_reads"] == r["trace_misses"], f"{label} physical_reads != hit=false lines")
        check(r["trace_lines"] == demands, f"{label} trace lines != host demands")
        if r["arm"] == "fill":
            check(r["fill_close"] is not None, f"{label} no fill close line")
            check((r["fill_close"] or {}).get("fill_refused", 1) == 0, f"{label} fill_refused != 0")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY45 FILL CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def window_delta(r, key):
        return (r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS

    for arm in ("i9g", "fill"):
        rs = [r for r in runs.values() if r["arm"] == arm]
        off_w = med(values("off", "window_s"))
        off_g = med(values("off", "gen_s"))
        per = {k: med([window_delta(r, k) for r in rs]) for k in
               ("gpu_misses", "host_hits", "host_misses", "demand_ns", "verify_ns", "step_ns", "finish_ns")}
        extra = ""
        if arm == "fill":
            at = {p: med([r["stages"][p].get("fill_admitted", 0) for r in rs]) for p in ("gate", "warm")}
            extra = f" fill_admitted gate={at['gate']:.0f} warm={at['warm']:.0f}"
        print(f"DAY45 ARM {arm} window_door_ms_per_token={(med(values(arm, 'window_s')) - off_w) * 1000 / N_TOKENS:.2f}"
              f" gen_door_ms_per_token={(med(values(arm, 'gen_s')) - off_g) * 1000 / N_TOKENS:.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}"
              f" | per window token: gpu_misses={per['gpu_misses']:.1f} host_hits={per['host_hits']:.1f}"
              f" host_misses={per['host_misses']:.1f} demand={per['demand_ns'] / 1e6:.3f}"
              f" verify={per['verify_ns'] / 1e6:.3f} step={per['step_ns'] / 1e6:.3f}"
              f" finish={per['finish_ns'] / 1e6:.3f}{extra}")

    noise = max(iqr(values("fill", "window_s")), iqr(values("i9g", "window_s")))
    diffs = {o: med(values("fill", "window_s", o)) - med(values("i9g", "window_s", o)) for o in ("o1", "o2")}
    c1 = all(d < -noise for d in diffs.values())
    print(f"DAY45 CLAUSE (i) fill_minus_i9g window o1={diffs['o1']:+.3f} o2={diffs['o2']:+.3f} noise={noise:.3f}"
          f" rule < -noise both orders -> {'PASS' if c1 else 'FAIL'}")
    fr = [r for r in runs.values() if r["arm"] == "fill"]
    ratio = med([window_delta(r, "host_hits") / window_delta(r, "gpu_misses") for r in fr if window_delta(r, "gpu_misses")])
    c2 = ratio >= 0.9
    print(f"DAY45 CLAUSE (ii) fill host_hits/gpu_misses per window token median={ratio:.3f} rule >=0.9 -> "
          f"{'PASS' if c2 else 'FAIL'}")
    gen = med(values("fill", "gen_s")) - med(values("i9g", "gen_s"))
    print(f"DAY45 READING gen fill_minus_i9g={gen:+.3f} s -> {'gen_lower' if gen < 0 else 'gen_not_lower'}")
    print(f"DAY45 FILL rig={rig} integrity={integrity} clause_i={'PASS' if c1 else 'FAIL'}"
          f" clause_ii={'PASS' if c2 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
