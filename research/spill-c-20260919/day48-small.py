#!/usr/bin/env python3
"""Day 48 reader for cell `small` (research/spill-c-20260919/DAY48.md section 1, registered before any I8 or I5
code). Integrity, clauses (i) to (iv), and the window reading.

usage: day48-small.py <cell-dir> [--rig NAME]
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
        close = r["stages"].get("close", {})
        demands = close.get("host_hits", 0) + close.get("host_misses", 0)
        check(r["physical_reads"] == r["trace_misses"], f"{label} physical_reads != hit=false lines")
        check(r["trace_lines"] == demands, f"{label} trace lines != host demands")
        check(r["fill_close"] is not None, f"{label} no fill close line")
        check((r["fill_close"] or {}).get("fill_refused", 1) == 0, f"{label} fill_refused != 0")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY48 SMALL CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def window_delta(r, key):
        return (r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS

    per = {}
    for arm in ("i2", "i8", "i5"):
        rs = [r for r in runs.values() if r["arm"] == arm]
        off_w = med(values("off", "window_s"))
        per[arm] = {k: med([window_delta(r, k) for r in rs]) for k in
                    ("gpu_misses", "host_hits", "validate_ns", "trace_ns", "demand_ns", "enqueue_ns",
                     "retire_ns", "finish_ns", "miss_total_ns")}
        print(f"DAY48 ARM {arm} window_door_ms_per_token={(med(values(arm, 'window_s')) - off_w) * 1000 / N_TOKENS:.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}"
              f" | per window token: gpu_misses={per[arm]['gpu_misses']:.1f} host_hits={per[arm]['host_hits']:.1f} "
              + " ".join(f"{k[:-3]}={v / 1e6:.3f}" for k, v in per[arm].items() if k.endswith("_ns")))

    def no_regression(a, b):
        noise = max(iqr(values(a, "window_s")), iqr(values(b, "window_s")))
        diffs = {o: med(values(a, "window_s", o)) - med(values(b, "window_s", o)) for o in (None, "o1", "o2")}
        return all(d <= noise for d in diffs.values()), diffs, noise

    c1 = per["i8"]["validate_ns"] < 0.1 * per["i2"]["validate_ns"]
    print(f"DAY48 CLAUSE (i) validate per window token i2={per['i2']['validate_ns'] / 1e6:.3f}"
          f" i8={per['i8']['validate_ns'] / 1e6:.3f} rule i8 < 0.1 x i2 -> {'PASS' if c1 else 'FAIL'}")
    c2, d2, n2 = no_regression("i8", "i2")
    print(f"DAY48 CLAUSE (ii) i8_minus_i2 window pooled={d2[None]:+.3f} o1={d2['o1']:+.3f} o2={d2['o2']:+.3f}"
          f" noise={n2:.3f} rule <=noise -> {'PASS' if c2 else 'FAIL'}")
    c3 = per["i5"]["trace_ns"] < 0.1 * per["i8"]["trace_ns"]
    print(f"DAY48 CLAUSE (iii) trace per window token i8={per['i8']['trace_ns'] / 1e6:.3f}"
          f" i5={per['i5']['trace_ns'] / 1e6:.3f} rule i5 < 0.1 x i8 -> {'PASS' if c3 else 'FAIL'}")
    c4, d4, n4 = no_regression("i5", "i8")
    print(f"DAY48 CLAUSE (iv) i5_minus_i8 window pooled={d4[None]:+.3f} o1={d4['o1']:+.3f} o2={d4['o2']:+.3f}"
          f" noise={n4:.3f} rule <=noise -> {'PASS' if c4 else 'FAIL'}")
    order = [med(values(a, "window_s")) for a in ("i2", "i8", "i5")]
    print(f"DAY48 READING window i2={order[0]:.3f} i8={order[1]:.3f} i5={order[2]:.3f} -> "
          f"{'falls' if order[0] > order[1] > order[2] else 'not_monotone'}")
    print(f"DAY48 SMALL rig={rig} integrity={integrity} i8={'PASS' if c1 and c2 else 'FAIL'}"
          f" i5={'PASS' if c3 and c4 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
