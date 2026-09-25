#!/usr/bin/env python3
"""Day 60 reader for cell `gap` (research/spill-c-20260919/DAY60.md section 1, registered before the instrument's
code): integrity, R1 the gaps, R2 the instrument's cost, R3 the CPU brackets, R4 the split, and the verdict line.

usage: day60-gap.py <cell-dir> [--rig NAME]
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

N = 32
CLOCK = re.compile(r"\[moe-cache\] dispatch-clock phase=(\w+) (.*)")
KEYS = ["dispatch_calls", "dispatch_ns", "dispatch_hits", "dispatch_pending", "dispatch_sync", "prefetch_calls",
        "prefetch_ns", "prefetch_issued", "pf_reserve_ns", "pf_stage_ns", "pf_retire_ns", "pf_resident_ns",
        "pf_demand_ns"]


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
        clocks = {}
        for m in CLOCK.finditer(text):
            clocks[m.group(1)] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2))}
        run["clocks"] = clocks
        runs[log.stem] = run
    fails = []
    if len(runs) != 40:
        fails.append(f"runs={len(runs)} (40 expected)")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')} ({N} expected)")
        clocked = r["arm"] in ("refc", "onc")
        if clocked and not all(p in r["clocks"] for p in ("gate", "generate", "warm", "window")):
            fails.append(f"{label} dispatch-clock lines missing")
        if not clocked and r["clocks"]:
            fails.append(f"{label} a clock line without the flag")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY60 GAP CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def gap(a, b, key, order=None):
        return (med(values(a, key, order)) - med(values(b, key, order))) * 1000 / N

    for key, label in (("gen_s", "gen"), ("window_s", "window")):
        print(f"DAY60 R1 {label}_gap_ms_per_token on_minus_ref pooled={gap('on', 'ref', key):+.3f}"
              f" o1={gap('on', 'ref', key, 'o1'):+.3f} o2={gap('on', 'ref', key, 'o2'):+.3f}"
              f" | medians ref={med(values('ref', key)):.3f} refc={med(values('refc', key)):.3f}"
              f" on={med(values('on', key)):.3f} onc={med(values('onc', key)):.3f} (N=10 each)")
    window_gap = gap("on", "ref", "window_s")
    bound = min(0.1 * abs(window_gap), 0.1)
    cost_ref, cost_on = gap("refc", "ref", "window_s"), gap("onc", "on", "window_s")
    print(f"DAY60 R2 instrument_ms_per_token refc_minus_ref={cost_ref:+.3f} onc_minus_on={cost_on:+.3f}"
          f" bound={bound:.3f} -> {'within_bound' if cost_ref <= bound and cost_on <= bound else 'over_bound'}")

    def per_token(arm, lo, hi):
        rs = [r for r in runs.values() if r["arm"] == arm and lo in r["clocks"] and hi in r["clocks"]]
        return {k: med([(r["clocks"][hi].get(k, 0) - r["clocks"][lo].get(k, 0)) / N for r in rs]) for k in KEYS}

    verdicts = []
    for lo, hi, label, wall in (("warm", "window", "window", "window_s"), ("gate", "generate", "gen", "gen_s")):
        ref, on = per_token("refc", lo, hi), per_token("onc", lo, hi)
        ms = lambda d, k: d[k] / 1e6
        print(f"DAY60 R3 {label} per token: " + " ".join(
            f"{k}={ms(on, k) - ms(ref, k):+.3f}" for k in KEYS if k.endswith("_ns"))
              + " | refc " + " ".join(f"{k}={ref[k]:.1f}" for k in KEYS if not k.endswith("_ns"))
              + " | onc " + " ".join(f"{k}={on[k]:.1f}" for k in KEYS if not k.endswith("_ns")))
        cpu_gap = (ms(on, "dispatch_ns") + ms(on, "prefetch_ns")) - (ms(ref, "dispatch_ns") + ms(ref, "prefetch_ns"))
        wall_gap = gap("on", "ref", wall)
        residual = wall_gap - cpu_gap
        # R3's terms as registered: dispatch_ns, prefetch_ns and each pf_* (prefetch_ns contains the pf_* terms).
        parts = {k: ms(on, k) - ms(ref, k) for k in KEYS if k.endswith("_ns")}
        top = max(parts, key=lambda k: parts[k])
        side = "cpu_side" if wall_gap > 0 and cpu_gap >= 0.75 * wall_gap else "not_cpu_side"
        print(f"DAY60 R4 {label} wall_gap={wall_gap:+.3f} cpu_gap={cpu_gap:+.3f} residual={residual:+.3f}"
              f" top={top} ({parts[top]:+.3f}) -> {side}")
        verdicts.append(f"{label}: wall_gap={wall_gap:+.3f} cpu_gap={cpu_gap:+.3f} top={top} {side}")
    print(f"DAY60 GAP rig={rig} integrity={integrity} " + "; ".join(verdicts))
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
