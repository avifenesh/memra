#!/usr/bin/env python3
"""DAY75: a synthetic `i16` cell for the reader's mechanics dry check, from DAY72's target-card `gap15` receipts: its
`ref` runs as `ref`, its `on` runs as both `i15` and `i16`, its `onc` runs as `i16c`, its four profiled runs with their
SQLite exports (from wherever they are kept) as Part B. The arms' meanings are invented; the reading means nothing.
usage: make-synthetic.py <gap15-cell-dir> <sqlite-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

src, db, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]), Path(sys.argv[3]) / "ev"
out.mkdir(parents=True, exist_ok=True)
names = {"ref": ["ref"], "on": ["i15", "i16"], "onc": ["i16c"]}
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    parts = lab.split("-")
    if parts[0].startswith("o") and parts[1] in names:
        for arm in names[parts[1]]:
            marks.append(f"{t}\t{parts[0]}-{arm}-{parts[2]} {rest}")
    elif lab.startswith("p-"):
        marks.append(line)
for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    parts = stem.split("-")
    if len(parts) == 3 and parts[0] in ("o1", "o2") and parts[1] in names:
        for arm in names[parts[1]]:
            shutil.copy(f, out / f"{parts[0]}-{arm}-{parts[2]}.{ext}")
    elif not (len(parts) == 3 and parts[0] in ("o1", "o2")):
        shutil.copy(f, out / f.name)
for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
    shutil.copy(db / f"{label}.sqlite", out / f"{label}.sqlite")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic cell {out.parent}: from {src}")
