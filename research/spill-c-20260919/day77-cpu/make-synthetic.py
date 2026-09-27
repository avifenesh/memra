#!/usr/bin/env python3
"""DAY77: synthetic `i17` cells for the reader's mechanics dry check, from DAY75's target-card `i16` receipts.
Mode `as-i16`: its `i16`/`i16c` runs relabelled `i17`/`i17c` (their host demand sequence differs from I15's, so the
equality check must void). Mode `as-i15`: its `i15` runs copied as `i17` and its `i16c` runs as `i17c` with their
trace lines replaced by the I15 runs' (the check passes; the reading means nothing).
usage: make-synthetic.py <i16-cell-dir> <sqlite-dir> <out-cell-dir> as-i16|as-i15
"""
import shutil
import sys
from pathlib import Path

src, db, out, mode = Path(sys.argv[1]) / "ev", Path(sys.argv[2]), Path(sys.argv[3]) / "ev", sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
rename = {"i16": "i17", "i16c": "i17c"} if mode == "as-i16" else {"i15": "i17", "i16c": "i17c"}
keep_as_is = {"ref", "i15"}
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    parts = lab.split("-")
    if parts[0] in ("o1", "o2"):
        if parts[1] in keep_as_is:
            marks.append(line)
        if parts[1] in rename:
            marks.append(f"{t}\t{parts[0]}-{rename[parts[1]]}-{parts[2]} {rest}")
    else:
        marks.append(line)
i15_trace = [l for l in (src / "o1-i15-r1.log").read_text(errors="replace").splitlines() if "[expert-host-slru] key=" in l]
for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    parts = stem.split("-")
    if len(parts) == 3 and parts[0] in ("o1", "o2"):
        if parts[1] in keep_as_is:
            shutil.copy(f, out / f.name)
        if parts[1] in rename:
            target = out / f"{parts[0]}-{rename[parts[1]]}-{parts[2]}.{ext}"
            if mode == "as-i15" and parts[1] == "i16c" and ext == "log":
                lines = f.read_text(errors="replace").splitlines()
                others = [l for l in lines if "[expert-host-slru] key=" not in l]
                target.write_text("\n".join(others + i15_trace) + "\n")
            else:
                shutil.copy(f, target)
    else:
        shutil.copy(f, out / f.name)
for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
    shutil.copy(db / f"{label}.sqlite", out / f"{label}.sqlite")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic cell {out.parent} ({mode})")
