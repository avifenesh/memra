#!/usr/bin/env python3
"""Day 47 reader for cell `pinned` (research/spill-c-20260919/DAY47.md section 1, registered before any I2 code).
Integrity, clause (i) I2 beats I1 in the window, clause (ii) the enqueue halves, and the copy_gpu reading.

usage: day47-pinned.py <cell-dir> [--rig NAME]
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
        check(r["fill_close"] is not None, f"{label} no fill close line")
        check((r["fill_close"] or {}).get("fill_refused", 1) == 0, f"{label} fill_refused != 0")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY47 PINNED CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def window_delta(r, key):
        return (r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS

    for arm in ("i1", "i2"):
        rs = [r for r in runs.values() if r["arm"] == arm]
        off_w = med(values("off", "window_s"))
        off_g = med(values("off", "gen_s"))
        per = {k: med([window_delta(r, k) for r in rs]) for k in
               ("gpu_misses", "host_hits", "demand_ns", "enqueue_ns", "copy_gpu_ns", "wait_ns", "retire_ns",
                "finish_ns", "alloc_ns", "step_ns", "miss_total_ns")}
        print(f"DAY47 ARM {arm} window_door_ms_per_token={(med(values(arm, 'window_s')) - off_w) * 1000 / N_TOKENS:.2f}"
              f" gen_door_ms_per_token={(med(values(arm, 'gen_s')) - off_g) * 1000 / N_TOKENS:.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}"
              f" | per window token: gpu_misses={per['gpu_misses']:.1f} host_hits={per['host_hits']:.1f} "
              + " ".join(f"{k[:-3]}={per[k] / 1e6:.3f}" for k in per if k.endswith("_ns")))

    noise = max(iqr(values("i2", "window_s")), iqr(values("i1", "window_s")))
    diffs = {o: med(values("i2", "window_s", o)) - med(values("i1", "window_s", o)) for o in ("o1", "o2")}
    c1 = all(d < -noise for d in diffs.values())
    print(f"DAY47 CLAUSE (i) i2_minus_i1 window o1={diffs['o1']:+.3f} o2={diffs['o2']:+.3f} noise={noise:.3f}"
          f" rule < -noise both orders -> {'PASS' if c1 else 'FAIL'}")
    enq = {arm: med([window_delta(r, "enqueue_ns") for r in runs.values() if r["arm"] == arm]) / 1e6
           for arm in ("i1", "i2")}
    c2 = enq["i2"] < 0.5 * enq["i1"]
    print(f"DAY47 CLAUSE (ii) enqueue per window token i1={enq['i1']:.3f} i2={enq['i2']:.3f} rule i2 < 0.5 x i1 -> "
          f"{'PASS' if c2 else 'FAIL'}")
    cp = {arm: med([window_delta(r, "copy_gpu_ns") for r in runs.values() if r["arm"] == arm]) / 1e6
          for arm in ("i1", "i2")}
    print(f"DAY47 READING copy_gpu per window token i1={cp['i1']:.3f} i2={cp['i2']:.3f} -> "
          f"{'copy_lower' if cp['i2'] < cp['i1'] else 'copy_not_lower'}")
    print(f"DAY47 PINNED rig={rig} integrity={integrity} clause_i={'PASS' if c1 else 'FAIL'}"
          f" clause_ii={'PASS' if c2 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
