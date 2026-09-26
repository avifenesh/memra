#!/usr/bin/env python3
"""WP-A design L' (DAY63.md section 4) reader, written before its sitting runs.

usage: l2-reading.py R
L's own reader (l-reading.py, sections 1 and 2's clauses, with the changed fault gate among the 11), then the gate
change's red arm: R/gates-red/contract-fault.exit must be 1 and R/gates-red/contract-fault.log must carry both FAIL lines
(`span-refusal: the staging set filled once ..` and `promote-span-refusal: the staging set filled once ..`); the redgate
binary must carry the red arm's marker (R/markers.txt).
Verdict: ADOPT when L's reader reads ADOPT and the red arm fails as required; a red arm that passes refutes the gate
change (and L' with it); otherwise L's reader's verdict stands.
"""
import os
import subprocess
import sys


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return ""


def main():
    root = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, os.path.join(here, "l-reading.py"), root], capture_output=True, text=True).stdout
    print(out.rstrip())
    last = out.strip().splitlines()[-1] if out.strip() else ""
    rc = rd(os.path.join(root, "gates-red", "contract-fault.exit"))
    log = rd(os.path.join(root, "gates-red", "contract-fault.log"))
    fails = [n for n in ("FAIL: span-refusal: the staging set filled once",
                         "FAIL: promote-span-refusal: the staging set filled once") if n in log]
    marker = "redgate marker: 0" not in rd(os.path.join(root, "markers.txt"))
    red_ok = rc == "1" and len(fails) == 2 and marker
    print(f"L2 GATE RED ARM rc={rc or 'missing'} staging-fill FAILs={len(fails)} of 2 marker={marker} -> "
          f"{'caught (as required)' if red_ok else 'NOT caught'}")
    if not red_ok:
        v = "REFUTED (the gate change's red arm was not caught): L' and the gate change revert"
    elif last.startswith("L VERDICT -> ADOPT"):
        v = "ADOPT (L' is the naked program; the gate change stands)"
    else:
        v = last.replace("L VERDICT", "L' VERDICT (per L's reader)")
    print(f"L2 VERDICT -> {v}")


if __name__ == "__main__":
    main()
