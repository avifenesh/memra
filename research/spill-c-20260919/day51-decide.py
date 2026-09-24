#!/usr/bin/env python3
"""Day 51 reader (research/spill-c-20260919/DAY51.md section 1, registered before any improvement rung was
measured): G1 (hash lock), G2 (run-spec self-consistency under the tuned door), and the decide cell (G3 integrity and
the door_wins / door_flat / door_loses rule on gen-only decode, the window read the same way beside it).

usage: day51-decide.py hashlock|spec|decide <cell-dir> [--rig NAME]
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

DOOR_TAGS = ("[experts-via-tier]", "[expert-host-slru]", "[expert-gpu-slru]")
IDENT = re.compile(r"catalog_sha256=([0-9a-f]+) records=(\d+) records_sha256=([0-9a-f]+)")
DAY40_IDENT = ("2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8", "30720",
               "8084706a52804d906afb7d9599d6ff44a7dd13c3b37af5fffb9caca72e0a459d")


def lines(path):
    return [raw.split("\t", 1)[1] if "\t" in raw else raw for raw in path.read_text(errors="replace").splitlines()]


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def hashlock(ev, rig):
    door = lines(ev / "door.log")
    control = lines(ev / "control.log")
    door_exit = int((ev / "door.exit").read_text())
    control_exit = int((ev / "control.exit").read_text())
    last = [l for l in door if l.strip()][-1] if door else ""
    tags = sum(1 for l in door if any(l.startswith(t) for t in DOOR_TAGS))
    match = any("argmax" in l and l.strip().endswith("MATCH") for l in control)
    ok = (door_exit == 1 and last.strip() == 'Error: "experts-via-tier artifact SHA256 mismatch"' and tags == 0
          and control_exit == 0 and match)
    print(f"DAY51 G1 rig={rig} door_exit={door_exit} last_line={last.strip()!r} door_lines={tags}"
          f" control_exit={control_exit} control_match={match} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def spec_cell(ev, rig):
    ok = True
    for label in ("spec", "spec-pressure", "spec-exact8"):
        text = lines(ev / f"{label}.log")
        rc = int((ev / f"{label}.exit").read_text())
        verdict = any("=== SELF-CONSISTENCY PASS ===" in l for l in text)
        per_k = [l for l in text if "self-consistency:" in l]
        k_pass = sum(1 for l in per_k if "self-consistency: PASS (identical to plain target)" in l)
        installed = any(l.startswith("[experts-via-tier] installed") for l in text)
        refused = sum(1 for l in text if "fill refused" in l)
        good = rc == 0 and verdict and k_pass == len(per_k) and k_pass >= 8 and installed and refused == 0
        ok = ok and good
        print(f"DAY51 G2 {label} rig={rig} exit={rc} self_consistency_pass={verdict} k_lines={len(per_k)}"
              f" k_pass={k_pass} installed={installed} fill_refused_lines={refused} -> {'PASS' if good else 'FAIL'}")
    print(f"DAY51 G2 rig={rig} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def decide(ev, rig):
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        ident = IDENT.search(text)
        run["ident"] = ident.groups() if ident else None
        run["trace_lines"] = text.count("[expert-host-slru] key=")
        run["trace_misses"] = text.count(" hit=false ")
        fill = re.search(r"\[experts-via-tier\] fill (fill_reads=.*)", text)
        run["fill"] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", fill.group(1))) if fill else None
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
        if r["arm"] != "on":
            check(r.get("door_lines", 0) == 0, f"{label} a legacy arm printed a door line")
            continue
        check(r["ident"] == DAY40_IDENT, f"{label} installer identity {r['ident']}")
        check(r["fill"] is not None and r["fill"].get("fill_refused") == 0, f"{label} fill_refused")
        check(r.get("physical_reads") == r["trace_misses"], f"{label} physical_reads != hit=false lines")
        check(r["trace_lines"] >= r["trace_misses"] > 0, f"{label} trace lines {r['trace_lines']} misses {r['trace_misses']}")
        check(not r["stages"], f"{label} stage lines without the flag")
    check(len({r.get("tokens") for r in runs.values()}) == 1, "tapes differ")
    check(len({r.get("steady") for r in runs.values()}) == 1, "steady lines differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY51 G3 rig={rig} runs={len(runs)} integrity={integrity}" + ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def verdict(key):
        noise = max(iqr(values("on", key)), iqr(values("off", key)))
        diff = {o: med(values("on", key, o)) - med(values("off", key, o)) for o in (None, "o1", "o2")}
        if all(d < -noise for d in diff.values()):
            name = "door_wins"
        elif all(diff[o] > noise for o in ("o1", "o2")):
            name = "door_loses"
        else:
            name = "door_flat"
        ratio = {o: med(values("on", key, o)) / med(values("off", key, o)) for o in (None, "o1", "o2")}
        return name, diff, noise, ratio

    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        name, diff, noise, ratio = verdict(key)
        print(f"DAY51 DECIDE rig={rig} {label}: off={med(values('off', key)):.3f} on={med(values('on', key)):.3f}"
              f" ref={med(values('ref', key)):.3f} (N=10 each) on_minus_off pooled={diff[None]:+.4f}"
              f" o1={diff['o1']:+.4f} o2={diff['o2']:+.4f} noise={noise:.4f} ratio={ratio[None]:.3f}"
              f" (o1 {ratio['o1']:.3f}, o2 {ratio['o2']:.3f}) -> {name}")
    print(f"DAY51 READING rig={rig} on install_s median={med(values('on', 'install_s')):.2f}"
          f" ref_minus_off gen={med(values('ref', 'gen_s')) - med(values('off', 'gen_s')):+.4f}"
          f" ref_minus_on gen={med(values('ref', 'gen_s')) - med(values('on', 'gen_s')):+.4f}")
    name = verdict("gen_s")[0] if integrity == "ok" else "void"
    print(f"DAY51 VERDICT rig={rig} integrity={integrity} -> {name}")
    return 0 if integrity == "ok" else 1


def main():
    mode, cell = sys.argv[1], Path(sys.argv[2])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    return {"hashlock": hashlock, "spec": spec_cell, "decide": decide}[mode](cell / "ev", rig)


if __name__ == "__main__":
    sys.exit(main())
