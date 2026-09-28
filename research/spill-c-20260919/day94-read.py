#!/usr/bin/env python3
"""Day 94 reader for the cell `where285` (research/spill-c-20260919/DAY94.md section 1, registered before this
script). Integrity, admissibility (DAY64's clause), the registered readings (DAY61 section 2's rule through
day88-read.py's `compare`), the dispatch-clock twins' brackets beside them, and the verdict.

usage: day94-read.py <root holding where285/> [--rig NAME]
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d88 = load("day88_read", "day88-read.py")
d83 = load("day83_read", "day83-read.py")

DOOR_SEQ = "4bdc2610c3534e42"
DOOR_LINES = 22077
TIMED = ["wn", "wl", "pn", "pl", "wnc", "wlc", "pnc", "plc", "wo"]
DOOR = {"wn", "pn", "wnc", "pnc", "wo"}
LEGACY = {"wl", "pl", "wlc", "plc"}
NEED_DOOR = ({"qualified", "installed", "registered", "pf_on"}, {"off_rollback", "allocated", "pf_effective_off"})
NEED_LEGACY = ({"off_rollback", "pf_on"}, {"installed", "qualified", "pf_effective_off"})
DISPATCH_KEYS = ("dispatch_ns", "prefetch_ns", "pf_demand_ns", "pf_resident_ns", "pf_retire_ns", "pf_stage_ns")


def expand(cpus):
    out = set()
    for part in cpus.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    root = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    cell = root / "where285"
    ev = cell / "ev"
    lists = dict(line.split("=", 1) for line in (ev / "cpus.txt").read_text().split())
    wide, pcores = lists["wide"], lists["pcores"]
    non_p = sorted(expand(wide) - expand(pcores))
    print(f"DAY94 CORES rig={rig} wide={wide} pcores={pcores} wide_non_p={non_p or 'none'}")
    runs = d88.read_cell(cell)
    fails = []
    got = sorted({r["arm"] for r in runs.values()})
    if got != sorted(TIMED) or any(len(d88.values(runs, a, "exit")) != 10 for a in TIMED):
        fails.append(f"arms={got} runs={len(runs)}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH") or r.get("gen_n") != d88.N:
            fails.append(f"{label} exit={r['exit']} match={r.get('match')} gen_n={r.get('gen_n')}")
        need, never = NEED_DOOR if r["arm"] in DOOR else NEED_LEGACY
        if not need <= r["marks"] or never & r["marks"]:
            fails.append(f"{label} lines: missing {sorted(need - r['marks'])}, unexpected {sorted(never & r['marks'])}")
        want_cpus = pcores if r["arm"].startswith("p") else wide
        cpus_file = ev / f"{label}.cpus"
        if not cpus_file.exists() or cpus_file.read_text().strip() != want_cpus:
            fails.append(f"{label} ran on {cpus_file.read_text().strip() if cpus_file.exists() else '?'}, not {want_cpus}")
        if r["arm"] in DOOR and (not r["fill_complete"] or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_complete']} physical_reads={r.get('physical_reads')}")
        # I25's door writes no trace without --expert-bank-trace; I24's (wo) always did, and it is the door's.
        if r["arm"] == "wo":
            if r["trace"] != DOOR_SEQ or r["trace_lines"] != DOOR_LINES:
                fails.append(f"{label} host demand sequence {r['trace']} ({r['trace_lines']} lines)")
        elif r["trace_lines"] != 0:
            fails.append(f"{label} carries {r['trace_lines']} host demand lines")
    # The traced twins of pn: the door's host demand sequence.
    twins = sorted(ev.glob("t-pnt-r[12].log"))
    if len(twins) != 2:
        fails.append(f"traced twins {len(twins)}")
    tapes = {r.get("tokens") for r in runs.values()}
    for log in twins:
        run = d88.d40.parse_run(log)
        body = "\n".join(line.partition("\t")[2] for line in log.read_text(errors="replace").splitlines())
        trace = [d88.SLOT.sub("", line) for line in body.splitlines() if "[expert-host-slru] key=" in line]
        seq = d88.hashlib.sha256("\n".join(trace).encode()).hexdigest()[:16] if trace else "-"
        rc = (ev / f"{log.stem}.exit").read_text().strip()
        print(f"DAY94 TRACED {log.stem} rc={rc} host demand sequence={seq} lines={len(trace)}")
        if rc != "0" or seq != DOOR_SEQ or len(trace) != DOOR_LINES:
            fails.append(f"{log.stem} rc={rc} sequence={seq} lines={len(trace)}")
        tapes.add(run.get("tokens"))
    if len(tapes) != 1:
        fails.append("tapes differ")
    ok = not fails
    print(f"DAY94 CHECKS rig={rig} runs={len(runs)} integrity={'ok' if ok else 'FAIL'}"
          + ("" if ok else " failed=" + "; ".join(fails)))
    adm = d88.admissibility(runs, TIMED, rig, "where285") if ok else False
    print("DAY94 gen-only decode medians (N=10 each): "
          + " ".join(f"{a}={d88.med(d88.values(runs, a, 'gen_s')):.3f}" for a in TIMED))
    print("DAY94 steady window medians (N=10 each): "
          + " ".join(f"{a}={d88.med(d88.values(runs, a, 'window_s')):.3f}" for a in TIMED))
    verdict = "void (integrity)" if not ok else "void (inadmissible)" if not adm else None
    names = ("beats", "loses", "matches")
    read = {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        for new, old, what, nm in (("wn", "wl", "GAP wide (naked_vs_legacy)", names),
                                   ("pn", "pl", "GAP pinned (naked_vs_legacy)", names),
                                   ("pn", "wn", "PIN door (pinned_vs_wide, beside)", None),
                                   ("pl", "wl", "PIN legacy (pinned_vs_wide, beside)", None),
                                   ("wn", "wo", "I25_vs_I24 (beside)", None)):
            line, name = d88.compare(runs, new, old, key) if nm is None else d88.compare(runs, new, old, key, nm)
            read[(what, key)] = name
            print(f"DAY94 {what} {label}: {line}")
    # The dispatch-clock twins, deciding nothing: each bracket's median in us per generated token, per phase.
    for phase, a, b in (("generate", "gate", "generate"), ("window", "warm", "window")):
        for arm in ("wnc", "wlc", "pnc", "plc"):
            per = [d83.terms(d83.parse(log), a, b) for log in sorted(ev.glob(f"o[12]-{arm}-r*.log"))]
            med = {k: d83.med([p[k] for p in per]) for k in DISPATCH_KEYS}
            print(f"DAY94 CLOCK {arm} {phase} N={len(per)} (us per token, medians): "
                  + " ".join(f"{k[:-3]}={d83.us_tok(med[k])}" for k in DISPATCH_KEYS))
    if verdict is None:
        gap_w, gap_p = read[("GAP wide (naked_vs_legacy)", "gen_s")], read[("GAP pinned (naked_vs_legacy)", "gen_s")]
        if not non_p:
            verdict = f"no_non_p_cores_in_wide (gap_wide={gap_w} gap_pinned={gap_p})"
        elif gap_w != "loses":
            verdict = f"gap_not_reproduced (gap_wide={gap_w} gap_pinned={gap_p})"
        elif gap_p in ("matches", "beats"):
            verdict = f"pinning_closes (gap_wide={gap_w} gap_pinned={gap_p})"
        else:
            verdict = f"pinning_does_not (gap_wide={gap_w} gap_pinned={gap_p})"
    print(f"DAY94 VERDICT rig={rig} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
