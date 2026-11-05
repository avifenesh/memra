#!/usr/bin/env python3
"""OWED 18 reader for the Max-Q kit (PREREG.md): the registered pairs reader per cell, then one
overall VERDICT line.

Usage: maxq-verdict.py RECEIPT_DIR [--label LABEL]

RECEIPT_DIR holds the collector cells handoff-1g/ and handoff-8g/ (visits/cycle-NN/cycle.json),
as run-sitting.sh step 7 and run.sh lay them out. For each cell present, the registered reader
research/spill-f-20260919/m1-handoff-pairs.py runs unchanged and its stdout and stderr go to
RECEIPT_DIR/<cell>.pairs.log, exactly as run-sitting.sh wrote them. Then one line:

  M1-HANDOFF-VERDICT <label> export_ms direct/buffered: <winner|loser|flat|no-verdict> \
      handoff-1g=<verdict>@<median_ratio>/<pairs> handoff-8g=... default=<direct|buffered|none> ...

Rule (PREREG.md, fixed before any Max-Q cycle): a cell counts only with 10 pairs, no failed
cycle and no malformed pair. default=direct when export_ms reads winner in both cells;
default=buffered when any counted cell reads flat or loser; otherwise none (no verdict, owed).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "research/spill-f-20260919/m1-handoff-pairs.py"
CELLS = ("handoff-1g", "handoff-8g")
REGISTERED_PAIRS = 10


def read_cell(root, cell):
    d = root / cell
    if not ((d / "visits").is_dir() or any(d.glob("round-[0-9][0-9]/visits"))):
        note = root / f"{cell}.SKIPPED"
        why = note.read_text().strip() if note.exists() else "cell absent"
        return {"state": "absent", "why": why}
    proc = subprocess.run([sys.executable, str(PAIRS), str(d)], capture_output=True, text=True)
    (root / f"{cell}.pairs.log").write_text(proc.stdout + proc.stderr)
    sys.stdout.write(proc.stdout + proc.stderr)
    try:
        out = json.loads((d / "handoff-pairs.json").read_text())
    except (OSError, ValueError) as error:
        return {"state": "unreadable", "why": f"pairs reader rc={proc.returncode}: {error}"}
    if out["failed_cycles"] or out["malformed_pairs"]:
        return {"state": "failed", "why": f"failed_cycles={out['failed_cycles']} malformed_pairs={out['malformed_pairs']}",
                "pairs": out["pairs"]}
    if out["pairs"] != REGISTERED_PAIRS:
        return {"state": "incomplete", "why": f"pairs={out['pairs']} of {REGISTERED_PAIRS}", "pairs": out["pairs"]}
    m = out["metrics"]["export_ms"]
    if m["verdict"] not in ("winner", "loser", "flat"):
        return {"state": "incomplete", "why": f"export_ms {m['verdict']}", "pairs": out["pairs"]}
    return {"state": "counted", "verdict": m["verdict"], "ratio": m["median_ratio"], "pairs": m["n_pairs"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--label", default="pro6000-maxq")
    a = ap.parse_args()
    cells = {c: read_cell(a.receipt, c) for c in CELLS}
    counted = {c: r for c, r in cells.items() if r["state"] == "counted"}
    if any(r["verdict"] in ("flat", "loser") for r in counted.values()):
        word = "loser" if any(r["verdict"] == "loser" for r in counted.values()) else "flat"
        default = "buffered"
    elif len(counted) == len(CELLS):
        word, default = "winner", "direct"
    else:
        word, default = "no-verdict", "none"
    parts = []
    for c, r in cells.items():
        if r["state"] == "counted":
            parts.append(f"{c}={r['verdict']}@{r['ratio']}/{r['pairs']}")
        else:
            parts.append(f"{c}={r['state']}")
    why = "; ".join(f"{c}: {r['why']}" for c, r in cells.items() if r["state"] != "counted")
    extra = []
    stop = a.receipt / "THERMAL-STOP.txt"
    if stop.exists():
        extra.append("thermal_stop=yes")
    card = a.receipt / "card.json"
    if card.exists():
        info = json.loads(card.read_text())
        extra.append(f"card_index={info.get('index')} power_limit_w={info.get('power_limit_w')} "
                     f"card=\"{info.get('name')}\"")
    line = (f"M1-HANDOFF-VERDICT {a.label} export_ms direct/buffered: {word} {' '.join(parts)} "
            f"default={default}" + (f" ({why})" if why else "") + (" " + " ".join(extra) if extra else ""))
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
