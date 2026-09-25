#!/usr/bin/env python3
"""Day 44 reader for cell `mapped` (research/spill-c-20260919/DAY44.md sections 1 and 1a, registered before any I9
code). Integrity, the census clause, clause (i) no regression, readings (ii) the install and (iii) the load.

usage: day44-mapped.py <cell-dir> [--rig NAME]
"""
import datetime
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
CENSUS = re.compile(r"\[experts-via-tier\] host expert storage: mmap=(\d+) \((\d+) bytes\) pinned=(\d+) \((\d+) "
                    r"bytes\) paged=(\d+) \((\d+) bytes\)")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def seconds(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    starts = {}
    for row in (ev / "marks.tsv").read_text().splitlines():
        stamp, label = row.split("\t")
        if label.endswith(" start"):
            t = datetime.datetime.strptime(stamp[:23], "%Y-%m-%dT%H:%M:%S.%f")
            starts[label[:-6]] = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        run["trace_lines"] = text.count("[expert-host-slru] key=")
        run["trace_misses"] = text.count(" hit=false ")
        census = CENSUS.search(text)
        run["census"] = tuple(int(x) for x in census.groups()) if census else None
        # The load ends at the q8rp mirrors line; run-gen prints `loaded ...` only after the
        # installer, so that line would count the install as load.
        for stamp, line in run["lines"]:
            if line.startswith("[q8rp] split-plane decode mirrors built"):
                run["loaded_s"] = seconds(stamp) - starts[log.stem]
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
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY44 MAPPED CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    censuses = {r["census"] for r in runs.values() if r["arm"] == "i9"}
    census_ok = len(censuses) == 1 and all(c is not None for c in censuses)
    if census_ok:
        mm, mb, pn, pb, pg, gb = next(iter(censuses))
        census_ok = pn == 0 and pg == 0 and mm > 0
        print(f"DAY44 CLAUSE census i9 mmap={mm} ({mb} bytes) pinned={pn} ({pb}) paged={pg} ({gb}) rule pinned=0 "
              f"paged=0 mmap>0 on every i9 run -> {'PASS' if census_ok else 'FAIL'}")
    else:
        print(f"DAY44 CLAUSE census i9 censuses={censuses} -> FAIL")

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    off = {o: med(values("off", "window_s", o)) for o in (None, "o1", "o2")}
    for arm in ("i6", "i9"):
        cost = {o: (med(values(arm, "window_s", o)) - off[o]) * 1000 / N_TOKENS for o in (None, "o1", "o2")}
        print(f"DAY44 ARM {arm} window_door_ms_per_token pooled={cost[None]:.2f} o1={cost['o1']:.2f} o2={cost['o2']:.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}"
              f" install_s median={med(values(arm, 'install_s')):.2f} iqr={iqr(values(arm, 'install_s')):.2f}"
              f" loaded_s median={med(values(arm, 'loaded_s')):.2f} iqr={iqr(values(arm, 'loaded_s')):.2f}")

    def diff(key, order=None):
        return med(values("i9", key, order)) - med(values("i6", key, order))

    noise = max(iqr(values("i9", "window_s")), iqr(values("i6", "window_s")))
    ok1 = all(diff("window_s", o) <= noise for o in (None, "o1", "o2"))
    print(f"DAY44 CLAUSE (i) no_regression i9_minus_i6 window pooled={diff('window_s'):+.3f} o1={diff('window_s', 'o1'):+.3f}"
          f" o2={diff('window_s', 'o2'):+.3f} noise={noise:.3f} rule <=noise pooled and both orders -> "
          f"{'PASS' if ok1 else 'FAIL'}")
    inoise = max(iqr(values("i9", "install_s")), iqr(values("i6", "install_s")))
    falls = all(diff("install_s", o) < -inoise for o in ("o1", "o2"))
    print(f"DAY44 READING (ii) install i9_minus_i6 o1={diff('install_s', 'o1'):+.2f} o2={diff('install_s', 'o2'):+.2f}"
          f" noise={inoise:.2f} -> {'install_falls' if falls else 'install_not_lower'}")
    lnoise = iqr(values("i6", "loaded_s"))
    grows = any(diff("loaded_s", o) > lnoise for o in ("o1", "o2"))
    print(f"DAY44 READING (iii) load i9_minus_i6 o1={diff('loaded_s', 'o1'):+.2f} o2={diff('loaded_s', 'o2'):+.2f}"
          f" noise={lnoise:.2f} -> {'load_grows' if grows else 'load_not_higher'}")
    print(f"DAY44 MAPPED rig={rig} integrity={integrity} census={'PASS' if census_ok else 'FAIL'}"
          f" no_regression={'PASS' if ok1 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
