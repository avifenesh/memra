#!/usr/bin/env python3
"""Day 56 reader (research/spill-c-20260919/DAY56.md section 1; the rule of DAY19.md Task 3): the identity gate's
drafter arm, door OFF (cell identity-dspark-off) against door ON (identity-dspark-on).

Rule: ALL GREEN both arms, the same verdict lines, equal `[prefix-host] demote:` bytes, under ON one receipt per tail
plane per draft layer (the `contracts door tail bound:` line with 2L Role::Tail segments for L draft layers), a
`DSPARK restore:` line in both arms with the same text, and no `refused (contracts door)` line under ON.

usage: day56-reading.py <receipts-root> [--rig NAME]
"""
import re
import sys
from pathlib import Path

DEMOTE = re.compile(r"\[prefix-host\] demote: (\d+) tokens, ([0-9.]+)MB in")
TAIL = re.compile(r"\[prefix-host\] contracts door tail bound: (\d+) draft layers, (\d+) Role::Tail segments \((\d+) B")
RESTORE = re.compile(r"\[prefix-cache\] DSPARK restore: .*")
REFUSAL = re.compile(r"refused[a-z ]*\(contracts door\)", re.IGNORECASE)


def arm(root, cell):
    out = root / cell
    gate = (out / "gate.log").read_text(errors="replace") if (out / "gate.log").exists() else ""
    log = out / "ev" / "host-on-server.log"
    server = log.read_text(errors="replace") if log.exists() else ""
    checks = [l.strip() for l in gate.splitlines() if l.strip().startswith(("ok:", "FAIL:"))]
    verdict = [l for l in gate.splitlines() if "IDENTITY GATE:" in l]
    return {
        "exit": (out / "gate.exit").read_text().strip() if (out / "gate.exit").exists() else "missing",
        "green": any("ALL GREEN" in l for l in verdict),
        "verdict": verdict[-1].strip() if verdict else "missing",
        "checks": checks,
        "demotes": DEMOTE.findall(server),
        "tails": [tuple(int(x) for x in m) for m in TAIL.findall(server)],
        "restores": RESTORE.findall(server),
        "refusals": REFUSAL.findall(server),
    }


def main():
    root = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    off, on = arm(root, "identity-dspark-off"), arm(root, "identity-dspark-on")
    for name, a in (("off", off), ("on", on)):
        print(f"DAY56 ARM {name} rig={rig} exit={a['exit']} verdict={a['verdict']!r} checks={len(a['checks'])} "
              f"demotes={a['demotes']} tails={a['tails']} restores={len(a['restores'])} refusals={len(a['refusals'])}")
    # The ON arm carries the drafter arm's door-only checks; every other check line must match.
    door_only = ("door ON",)
    common_on = [c for c in on["checks"] if not any(d in c for d in door_only)]
    terms = {
        "all_green_both": off["green"] and on["green"],
        "same_verdict_lines": off["checks"] == common_on,
        "equal_demote_bytes": bool(off["demotes"]) and off["demotes"] == on["demotes"],
        "tail_receipt_2L": bool(on["tails"]) and all(seg == 2 * layers for layers, seg, _ in on["tails"]),
        "restore_both_same_text": bool(off["restores"]) and off["restores"] == on["restores"],
        "no_refusal_on": not on["refusals"],
    }
    for k, v in terms.items():
        print(f"DAY56 TERM {k} -> {'PASS' if v else 'FAIL'}")
    verdict = "PASS" if all(terms.values()) else "FAIL"
    print(f"DAY56 DFLASH TAIL rig={rig} -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
