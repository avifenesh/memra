#!/usr/bin/env python3
"""Day 49 reader for cell `install` (research/spill-c-20260919/DAY49.md section 1, registered before any I7 code).
Integrity with the installer identity values, clauses (i) records_ns, (ii) install_s and (iii) no window regression.

usage: day49-install.py <cell-dir> [--rig NAME]
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
IDENT = re.compile(r"catalog_sha256=([0-9a-f]+) records=(\d+) records_sha256=([0-9a-f]+)")
DAY40_IDENT = ("2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8", "30720",
               "8084706a52804d906afb7d9599d6ff44a7dd13c3b37af5fffb9caca72e0a459d")


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
        ident = IDENT.search(text)
        run["ident"] = ident.groups() if ident else None
        if r_inst := run.get("install"):
            run["records_s"] = r_inst.get("records_ns", 0) / 1e9
            run["sha_s"] = r_inst.get("sha_ns", 0) / 1e9
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
        check(r["ident"] == DAY40_IDENT, f"{label} installer identity {r['ident']} != day 40's")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY49 INSTALL CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    for arm in ("i5", "i7"):
        print(f"DAY49 ARM {arm} install_s median={med(values(arm, 'install_s')):.2f} iqr={iqr(values(arm, 'install_s')):.2f}"
              f" records_s median={med(values(arm, 'records_s')):.2f} sha_s median={med(values(arm, 'sha_s')):.2f}"
              f" window_s median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.3f}")
    rec = {a: med(values(a, "records_s")) for a in ("i5", "i7")}
    c1 = rec["i7"] < 0.25 * rec["i5"]
    print(f"DAY49 CLAUSE (i) records_s i5={rec['i5']:.2f} i7={rec['i7']:.2f} rule i7 < 0.25 x i5 -> {'PASS' if c1 else 'FAIL'}")
    inoise = max(iqr(values("i7", "install_s")), iqr(values("i5", "install_s")))
    idiff = {o: med(values("i7", "install_s", o)) - med(values("i5", "install_s", o)) for o in ("o1", "o2")}
    c2 = all(d < -inoise for d in idiff.values())
    print(f"DAY49 CLAUSE (ii) install_s i7_minus_i5 o1={idiff['o1']:+.2f} o2={idiff['o2']:+.2f} noise={inoise:.2f}"
          f" rule < -noise both orders -> {'PASS' if c2 else 'FAIL'}")
    wnoise = max(iqr(values("i7", "window_s")), iqr(values("i5", "window_s")))
    wdiff = {o: med(values("i7", "window_s", o)) - med(values("i5", "window_s", o)) for o in (None, "o1", "o2")}
    c3 = all(d <= wnoise for d in wdiff.values())
    print(f"DAY49 CLAUSE (iii) window i7_minus_i5 pooled={wdiff[None]:+.3f} o1={wdiff['o1']:+.3f} o2={wdiff['o2']:+.3f}"
          f" noise={wnoise:.3f} rule <=noise -> {'PASS' if c3 else 'FAIL'}")
    print(f"DAY49 INSTALL rig={rig} integrity={integrity} clause_i={'PASS' if c1 else 'FAIL'}"
          f" clause_ii={'PASS' if c2 else 'FAIL'} clause_iii={'PASS' if c3 else 'FAIL'}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
