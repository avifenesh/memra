#!/usr/bin/env python3
"""DAY79: a synthetic `i18` cell for the reader's mechanics dry check, from DAY77's target-card `i17` receipts, the arms
relabelled one step (`i15` as `i17`, `i17` as `i18`, `i17c` as `i18c`); the four profiled runs with their SQLite exports
from where they are kept. The reading means nothing. usage: make-synthetic.py <i17-cell-dir> <sqlite-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

src, db, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]), Path(sys.argv[3]) / "ev"
out.mkdir(parents=True, exist_ok=True)
rename = {"i15": "i17", "i17": "i18", "i17c": "i18c"}


def relabel(label):
    parts = label.split("-")
    if len(parts) == 3 and parts[0] in ("o1", "o2") and parts[1] in rename:
        parts[1] = rename[parts[1]]
    return "-".join(parts)


for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    shutil.copy(f, out / f"{relabel(stem)}.{ext}" if ext else out / relabel(stem))
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    marks.append(f"{t}\t{relabel(lab)} {rest}")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
    shutil.copy(db / f"{label}.sqlite", out / f"{label}.sqlite")
print(f"synthetic cell {out.parent}")
