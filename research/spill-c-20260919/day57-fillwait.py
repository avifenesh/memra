#!/usr/bin/env python3
"""Day 57 reader for cell `fillwait` (research/spill-c-20260919/DAY57.md section 1, registered before any I10 code).
Integrity, clause (i) physical_reads=0 on every I10 run, clause (ii) I10 beats I4 in gen-only decode, clause (iii)
no window regression, and the install reading.

usage: day57-fillwait.py <cell-dir> [--rig NAME]
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
FILL_DONE = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms: (fill_reads=.*)")


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
        fd = FILL_DONE.search(text)
        run["fill_done_ms"] = float(fd.group(1)) if fd else None
        run["fill_done"] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", fd.group(2))) if fd else None
        if "installed_t" in run and "q8rp_t" in run:
            run["install_s"] = run["installed_t"] - run["q8rp_t"]
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
        if r["arm"] == "i10":
            check(r["fill_done"] is not None and r["fill_done"].get("fill_refused", 1) == 0,
                  f"{label} no fill-complete line with fill_refused=0")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY57 FILLWAIT CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def window_delta(r, key):
        return (r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS

    for arm in ("i4", "i10"):
        rs = [r for r in runs.values() if r["arm"] == arm]
        off_w = med(values("off", "window_s"))
        off_g = med(values("off", "gen_s"))
        per = {k: med([window_delta(r, k) for r in rs]) for k in
               ("gpu_misses", "host_hits", "prefetches", "prefetch_hits", "demand_ns", "enqueue_ns", "copy_gpu_ns",
                "wait_ns", "retire_ns", "miss_total_ns")}
        print(f"DAY57 ARM {arm} window_door_ms_per_token={(med(values(arm, 'window_s')) - off_w) * 1000 / N_TOKENS:.2f}"
              f" gen_door_ms_per_token={(med(values(arm, 'gen_s')) - off_g) * 1000 / N_TOKENS:.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}"
              f" | per window token: gpu_misses={per['gpu_misses']:.1f} host_hits={per['host_hits']:.1f}"
              f" prefetches={per['prefetches']:.1f} prefetch_hits={per['prefetch_hits']:.1f} "
              + " ".join(f"{k[:-3]}={per[k] / 1e6:.3f}" for k in per if k.endswith("_ns")))

    reads = [r.get("physical_reads") for r in runs.values() if r["arm"] == "i10"]
    c1 = bool(reads) and all(x == 0 for x in reads)
    print(f"DAY57 CLAUSE (i) i10 physical_reads per run={reads} rule all 0 -> {'PASS' if c1 else 'FAIL'}")
    gnoise = max(iqr(values("i10", "gen_s")), iqr(values("i4", "gen_s")))
    gdiff = {o: med(values("i10", "gen_s", o)) - med(values("i4", "gen_s", o)) for o in ("o1", "o2")}
    c2 = all(d < -gnoise for d in gdiff.values())
    print(f"DAY57 CLAUSE (ii) i10_minus_i4 gen o1={gdiff['o1']:+.3f} o2={gdiff['o2']:+.3f} noise={gnoise:.3f}"
          f" rule < -noise both orders -> {'PASS' if c2 else 'FAIL'}")
    wnoise = max(iqr(values("i10", "window_s")), iqr(values("i4", "window_s")))
    wdiff = {o: med(values("i10", "window_s", o)) - med(values("i4", "window_s", o)) for o in (None, "o1", "o2")}
    c3 = all(d <= wnoise for d in wdiff.values())
    print(f"DAY57 CLAUSE (iii) i10_minus_i4 window pooled={wdiff[None]:+.3f} o1={wdiff['o1']:+.3f}"
          f" o2={wdiff['o2']:+.3f} noise={wnoise:.3f} rule <= noise pooled and both orders -> {'PASS' if c3 else 'FAIL'}")
    fills = [r["fill_done_ms"] for r in runs.values() if r["arm"] == "i10" and r.get("fill_done_ms") is not None]
    print(f"DAY57 READING install_s median i4={med(values('i4', 'install_s')):.2f} i10={med(values('i10', 'install_s')):.2f}"
          f" (i10 - i4 {med(values('i10', 'install_s')) - med(values('i4', 'install_s')):+.2f}) fill_complete_ms median={med(fills):.1f}"
          f" gen_s median off={med(values('off', 'gen_s')):.3f} i4={med(values('i4', 'gen_s')):.3f} i10={med(values('i10', 'gen_s')):.3f}")
    print(f"DAY57 FILLWAIT rig={rig} integrity={integrity} clause_i={'PASS' if c1 else 'FAIL'}"
          f" clause_ii={'PASS' if c2 else 'FAIL'} clause_iii={'PASS' if c3 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
