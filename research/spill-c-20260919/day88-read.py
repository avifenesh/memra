#!/usr/bin/env python3
"""Day 88 reader for the cells `promo`, `promo-res` and `promo-spec` (research/spill-c-20260919/DAY88.md section 5,
registered before this script): each cell's integrity and admissibility, the registered readings (DAY61 section 2's
rule, gen-only primary, the window beside), the startup walls beside them, and the phase's verdict.

usage: day88-read.py <root holding promo/, promo-res/, promo-spec/> [--rig NAME]
"""
import datetime
import hashlib
import importlib.util
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d40 = load("day40_attrib", "day40-attrib.py")

N = 32
ADMISSIBLE_IQR = 0.005
SLOT = re.compile(r" slot=\d+")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
MARKS = {
    "qualified": re.compile(r"\[experts-via-tier\] qualified sha256=df27a780"),
    "installed": re.compile(r"\[experts-via-tier\] installed artifact_sha256=df27a780"),
    "registered": re.compile(r"host pinned pool bytes=.* registered$", re.M),
    "allocated": re.compile(r"host pinned pool bytes=.* (allocated|fill_reserve=\d+)$", re.M),
    "pf_on": re.compile(r"\[moe-prefetch\] default=on"),
    "pf_effective_off": re.compile(r"\[moe-prefetch\] default=\S+(?: \([^)]*\))? effective=off"),
    "off_rollback": re.compile(r"\[experts-via-tier\] off: MEMRA_EXPERTS_VIA_TIER=0"),
    "off_resident": re.compile(r"\[experts-via-tier\] off: the experts are resident"),
}
# What each arm must print and must not (DAY88 section 5's integrity).
EXPECT = {
    "naked": ({"qualified", "installed", "registered", "pf_on"}, {"off_rollback", "allocated", "pf_effective_off"}),
    "q22": ({"installed", "allocated"}, {"qualified", "registered", "off_rollback"}),
    "legacy": ({"off_rollback", "pf_on"}, {"installed", "qualified", "pf_effective_off"}),
    "alloc": ({"qualified", "installed", "allocated", "pf_on"}, {"registered", "off_rollback", "pf_effective_off"}),
    "nopf": ({"qualified", "installed", "registered", "pf_on", "pf_effective_off"}, {"off_rollback"}),
    "legnopf": ({"off_rollback", "pf_on", "pf_effective_off"}, {"installed", "qualified"}),
}
EXPECT_RES = {
    "naked": ({"off_resident", "pf_on", "qualified"}, {"installed", "off_rollback", "pf_effective_off"}),
    "legacy": ({"off_rollback", "pf_on"}, {"installed", "qualified", "pf_effective_off"}),
}
DOORS = ("naked", "q22", "alloc", "nopf")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def stamp_s(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_cell(cell):
    ev = cell / "ev"
    starts = {}
    marks_file = ev / "marks.tsv"
    if marks_file.exists():
        for line in marks_file.read_text().splitlines():
            when, _, what = line.partition("\t")
            label, _, kind = what.partition(" ")
            if kind == "start":
                t = datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
                starts[label] = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        body = "\n".join(line.partition("\t")[2] for line in text.splitlines())
        run["marks"] = {k for k, rx in MARKS.items() if rx.search(body)}
        run["fill_complete"] = bool(FILL.search(body))
        trace = [SLOT.sub("", line) for line in body.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()[:16] if trace else "-"
        run["trace_lines"] = len(trace)
        # Startup: the process start (the cell's mark) to the stamped prefill MATCH line.
        match_at = next((line.partition("\t")[0] for line in text.splitlines()
                         if "MATCH" in line and "argmax" in line), None)
        if match_at and log.stem in starts:
            run["to_prefill_s"] = (stamp_s(match_at) - starts[log.stem]) % 86400
        runs[log.stem] = run
    return runs


def values(runs, arm, key, order=None):
    return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]


def compare(runs, new, old, key, names=("improves", "regresses", "flat")):
    noise = max(iqr(values(runs, new, key)), iqr(values(runs, old, key)))
    diff = {o: med(values(runs, new, key, o)) - med(values(runs, old, key, o)) for o in (None, "o1", "o2")}
    if all(d < -noise for d in diff.values()):
        name = names[0]
    elif diff["o1"] > noise and diff["o2"] > noise:
        name = names[1]
    else:
        name = names[2]
    return f"pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}", name


def integrity(runs, arms, expect, rig, cell):
    fails = []
    got = sorted({r["arm"] for r in runs.values()})
    if got != sorted(arms) or any(len(values(runs, a, "exit")) != 10 for a in arms):
        fails.append(f"arms={got} runs={len(runs)}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH") or r.get("gen_n") != N:
            fails.append(f"{label} exit={r['exit']} match={r.get('match')} gen_n={r.get('gen_n')}")
        need, never = expect[r["arm"]]
        if not need <= r["marks"] or never & r["marks"]:
            fails.append(f"{label} lines: missing {sorted(need - r['marks'])}, unexpected {sorted(never & r['marks'])}")
        if r["arm"] in DOORS and cell == "promo":
            if not r["fill_complete"] or r.get("physical_reads") != 0 or r["trace_lines"] == 0:
                fails.append(f"{label} fill={r['fill_complete']} physical_reads={r.get('physical_reads')}"
                             f" trace_lines={r['trace_lines']}")
        elif r["trace_lines"] != 0:
            fails.append(f"{label} carries {r['trace_lines']} host demand lines")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if cell == "promo":
        # The door with its prefetch (naked, q22, alloc) is one program; without it (nopf) its own, constant.
        with_pf = {r["trace"] for r in runs.values() if r["arm"] in ("naked", "q22", "alloc")}
        if len(with_pf) != 1:
            fails.append(f"the door arms' host demand sequences differ: {sorted(with_pf)}")
        if len({r["trace"] for r in runs.values() if r["arm"] == "nopf"}) != 1:
            fails.append("nopf's host demand sequence differs within the arm")
        print(f"DAY88 {cell} host demand sequences: door {sorted(with_pf)}"
              f" nopf {sorted({r['trace'] for r in runs.values() if r['arm'] == 'nopf'})}")
    ok = not fails
    print(f"DAY88 {cell.upper()} CHECKS rig={rig} runs={len(runs)} integrity={'ok' if ok else 'FAIL'}"
          + ("" if ok else " failed=" + "; ".join(fails)))
    return ok


def admissibility(runs, arms, rig, cell):
    failing = [f"{a}:{k}={iqr(values(runs, a, k)):.4f}" for a in arms for k in ("gen_s", "window_s")
               if not iqr(values(runs, a, k)) <= ADMISSIBLE_IQR]
    worst = {k: max(iqr(values(runs, a, k)) for a in arms) for k in ("gen_s", "window_s")}
    print(f"DAY88 {cell.upper()} ADMISSIBILITY rig={rig} ceiling={ADMISSIBLE_IQR} max_iqr_gen={worst['gen_s']:.4f}"
          f" max_iqr_window={worst['window_s']:.4f} failing={failing}"
          f" -> {'admissible' if not failing else 'inadmissible'}")
    return not failing


def main():
    root = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    blocks = []
    # promo
    runs = read_cell(root / "promo")
    arms = list(EXPECT)
    ok = integrity(runs, arms, EXPECT, rig, "promo")
    adm = admissibility(runs, arms, rig, "promo") if ok else False
    step = {}
    if ok:
        for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
            print(f"DAY88 PROMO {label} medians (N=10 each): "
                  + " ".join(f"{a}={med(values(runs, a, key)):.3f}" for a in arms))
            line, step[("q22", key)] = compare(runs, "naked", "q22", key)
            print(f"DAY88 PROMO naked_vs_q22 {label}: {line}")
            line, step[("legacy", key)] = compare(runs, "naked", "legacy", key, ("beats", "loses", "matches"))
            print(f"DAY88 PROMO naked_vs_legacy {label}: {line}")
            line, _ = compare(runs, "alloc", "naked", key)
            print(f"DAY88 PROMO alloc_vs_naked {label} (the pool's rollback, recorded): {line}")
            line, _ = compare(runs, "naked", "nopf", key)
            print(f"DAY88 PROMO naked_vs_nopf {label} (the door's prefetch, recorded): {line}")
            line, _ = compare(runs, "legacy", "legnopf", key)
            print(f"DAY88 PROMO legacy_vs_legnopf {label} (the legacy's prefetch, recorded): {line}")
        print("DAY88 PROMO to_prefill_s medians (process start to the prefill MATCH line, beside): "
              + " ".join(f"{a}={med(values(runs, a, 'to_prefill_s')):.2f}" for a in arms))
    if not ok:
        blocks.append("promo integrity FAIL")
    elif not adm:
        blocks.append("promo inadmissible")
    else:
        if step[("q22", "gen_s")] == "regresses":
            blocks.append("naked regresses against q22")
        if step[("legacy", "gen_s")] == "loses":
            blocks.append("naked loses to legacy")
    # promo-res
    runs = read_cell(root / "promo-res")
    arms = list(EXPECT_RES)
    ok = integrity(runs, arms, EXPECT_RES, rig, "promo-res")
    adm = admissibility(runs, arms, rig, "promo-res") if ok else False
    if ok:
        for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
            print(f"DAY88 PROMO-RES {label} medians (N=10 each): "
                  + " ".join(f"{a}={med(values(runs, a, key)):.3f}" for a in arms))
            line, name = compare(runs, "naked", "legacy", key)
            print(f"DAY88 PROMO-RES naked_vs_legacy {label}: {line}")
            if key == "gen_s" and adm and name == "regresses":
                blocks.append("the resident shape regresses")
        print("DAY88 PROMO-RES to_prefill_s medians (beside): "
              + " ".join(f"{a}={med(values(runs, a, 'to_prefill_s')):.2f}" for a in arms))
    if not ok:
        blocks.append("promo-res integrity FAIL")
    elif not adm:
        blocks.append("promo-res inadmissible")
    # promo-spec
    ev = root / "promo-spec" / "ev"
    spec = {}
    for label in ("spec-naked", "spec-legacy"):
        log, rc = ev / f"{label}.log", ev / f"{label}.exit"
        text = log.read_text(errors="replace") if log.exists() else ""
        spec[label] = (rc.read_text().strip() if rc.exists() else "-",
                       "=== SELF-CONSISTENCY PASS ===" in text,
                       bool(MARKS["installed"].search(text)), bool(MARKS["off_rollback"].search(text)))
        print(f"DAY88 PROMO-SPEC {label} rc={spec[label][0]} self_consistency={'PASS' if spec[label][1] else 'not read'}"
              f" installed={spec[label][2]} off_rollback={spec[label][3]}")
    if not (spec["spec-naked"][:3] == ("0", True, True) and spec["spec-legacy"][0] == "0"
            and spec["spec-legacy"][1] and spec["spec-legacy"][3]):
        blocks.append("promo-spec")
    if any(b.endswith("FAIL") or b.endswith("inadmissible") for b in blocks):
        verdict = "void (" + "; ".join(blocks) + ")"
    elif blocks:
        verdict = "phase1_does_not_land (" + "; ".join(blocks) + ")"
    else:
        verdict = "phase1_lands"
    print(f"DAY88 VERDICT rig={rig} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
