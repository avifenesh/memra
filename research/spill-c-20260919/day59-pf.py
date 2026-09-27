#!/usr/bin/env python3
"""Day 59 reader (research/spill-c-20260919/DAY59.md section 1, registered before any cell): MEMRA_MOE_PREFETCH=1's
gates (pfgates: G1 tapes, G2 run-spec; pfserve: G3) and its timing cells (pftime, pfnaked) with the rule.

usage: day59-pf.py gates|serve|time <cell-dir> [--rig NAME] [--shape NAME]
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def gates(ev, rig):
    ok = True
    for slots in (9986, 512):
        runs = {arm: d40.parse_run(ev / f"tape-{arm}-s{slots}.log") for arm in ("off", "pf")}
        exits = {arm: int((ev / f"tape-{arm}-s{slots}.exit").read_text()) for arm in ("off", "pf")}
        match = all(r.get("match", "").endswith("MATCH") for r in runs.values())
        tapes = {r.get("tokens") for r in runs.values()}
        good = all(e == 0 for e in exits.values()) and match and len(tapes) == 1 and None not in tapes
        ok = ok and good
        print(f"DAY59 G1 rig={rig} slots={slots} exits={exits} match={match} one_tape={len(tapes) == 1}"
              f" -> {'PASS' if good else 'FAIL'}")
    text = (ev / "spec-pf.log").read_text(errors="replace")
    rc = int((ev / "spec-pf.exit").read_text())
    per_k = [l for l in text.splitlines() if "self-consistency:" in l]
    k_pass = sum("self-consistency: PASS (identical to plain target)" in l for l in per_k)
    good = rc == 0 and "=== SELF-CONSISTENCY PASS ===" in text and k_pass == len(per_k) >= 8
    ok = ok and good
    print(f"DAY59 G2 rig={rig} exit={rc} k_lines={len(per_k)} k_pass={k_pass} -> {'PASS' if good else 'FAIL'}")
    print(f"DAY59 GATES rig={rig} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def serve(ev, rig):
    names = [f"seq{i}" for i in range(3)] + [f"conc{i}" for i in range(4)]
    texts, errors = {}, []
    for arm in ("off", "pf"):
        for n in names:
            p = ev / f"serve-{arm}" / f"{n}.json"
            try:
                d = json.loads(p.read_text())
                texts[(arm, n)] = d["choices"][0]["text"]
            except Exception as err:
                errors.append(f"{arm}/{n}: {err!r}")
    equal = [n for n in names if (("off", n) in texts and texts.get(("off", n)) == texts.get(("pf", n)))]
    good = not errors and len(equal) == len(names)
    print(f"DAY59 G3 rig={rig} requests={len(names)} equal={len(equal)} errors={errors}"
          f" -> {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


def timing(ev, rig, shape):
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        runs[log.stem] = run
    fails = []
    if len(runs) != 20:
        fails.append(f"runs={len(runs)}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY59 CHECKS rig={rig} shape={shape} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    verdicts = {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        if not values("off", key):
            print(f"DAY59 TIME rig={rig} shape={shape} {label}: not printed by this shape")
            continue
        noise = max(iqr(values("pf", key)), iqr(values("off", key)))
        diff = {o: med(values("pf", key, o)) - med(values("off", key, o)) for o in (None, "o1", "o2")}
        if all(d < -noise for d in diff.values()):
            name = "pf_wins"
        elif all(diff[o] > noise for o in ("o1", "o2")):
            name = "pf_loses"
        else:
            name = "pf_flat"
        verdicts[key] = name
        print(f"DAY59 TIME rig={rig} shape={shape} {label}: off={med(values('off', key)):.3f}"
              f" pf={med(values('pf', key)):.3f} (N=10 each) pf_minus_off pooled={diff[None]:+.4f}"
              f" o1={diff['o1']:+.4f} o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}")
    name = verdicts.get("gen_s", "void") if integrity == "ok" else "void"
    print(f"DAY59 VERDICT rig={rig} shape={shape} integrity={integrity} -> {name}")
    return 0 if integrity == "ok" else 1


def main():
    mode, cell = sys.argv[1], Path(sys.argv[2])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    shape = sys.argv[sys.argv.index("--shape") + 1] if "--shape" in sys.argv else cell.name
    ev = cell / "ev"
    if mode == "gates":
        return gates(ev, rig)
    if mode == "serve":
        return serve(ev, rig)
    return timing(ev, rig, shape)


if __name__ == "__main__":
    sys.exit(main())
