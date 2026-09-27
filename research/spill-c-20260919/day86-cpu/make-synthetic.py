#!/usr/bin/env python3
"""DAY86: a synthetic `slow86` cell for the reader's mechanics dry check, from DAY84's BOX31 `i21` receipts (a bimodal
host): `ref` as it is, `i15` as `i20`, `i20` as `i21`, `i21` as `i22`, `i21c` as `i22r`, with invented vmstat
sections in each boundary snapshot (a slow boot's pgmigrate_fail large). The reading means nothing.
usage: make-synthetic.py <day84-9950x-cell-dir> <out-cell-dir>
"""
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
names = {"ref": "ref", "i15": "i20", "i20": "i21", "i21": "i22", "i21c": "i22r"}
GEN = re.compile(r"generated 32 tokens in ([0-9.]+)s")
counter = 1000
for log in sorted(src.glob("o[12]-*-r[1-5].log")):
    order, arm, rep = log.stem.split("-")
    label = f"{order}-{names[arm]}-{rep}"
    shutil.copy(log, out / f"{label}.log")
    shutil.copy(src / f"{log.stem}.exit", out / f"{label}.exit")
    gen = float(GEN.search(log.read_text(errors="replace")).group(1))
    fail = 70000 if gen > 0.28 else 12
    for side, add in (("before", 0), ("after", fail)):
        base = (src / f"{log.stem}.{side}.snap").read_text()
        vm = "\n".join(f"{k} {counter + (add if k == 'pgmigrate_fail' else add // 7)}" for k in (
            "compact_isolated", "compact_migrate_scanned", "compact_fail", "compact_stall", "pgmigrate_fail",
            "pgmigrate_success", "thp_fault_alloc", "thp_fault_fallback"))
        (out / f"{label}.{side}.snap").write_text(base + "## vmstat\n" + vm + "\n## buddyinfo\nNode 0, zone Normal 1 2 3\n")
    counter += fail + 5
print(f"synthetic cell {out.parent}")
