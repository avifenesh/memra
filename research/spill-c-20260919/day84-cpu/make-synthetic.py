#!/usr/bin/env python3
"""DAY84: a synthetic `i21` cell for the reader's mechanics dry check, from DAY82's BOX39 `i20` receipts: `ref` and `i15` as they are,
`i18` as `i20`, `i20` as `i21`, `i20c` as `i21c`, the profiled runs with their SQLite exports. The
reading means nothing. usage: make-synthetic.py <i20-cell-dir> <sqlite-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

src, db, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]), Path(sys.argv[3]) / "ev"
out.mkdir(parents=True, exist_ok=True)
names = {"ref": ["ref"], "i15": ["i15"], "i18": ["i20"], "i20": ["i21"], "i20c": ["i21c"]}


def targets(stem):
    parts = stem.split("-")
    if len(parts) == 3 and parts[0] in ("o1", "o2"):
        return [f"{parts[0]}-{a}-{parts[2]}" for a in names.get(parts[1], [])]
    return None


for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    t = targets(stem)
    if t is None:
        shutil.copy(f, out / f.name)
    for target in t or []:
        shutil.copy(f, out / f"{target}.{ext}")
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    tg = targets(lab)
    for target in (tg if tg is not None else [lab]):
        marks.append(f"{t}\t{target} {rest}")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
    shutil.copy(db / f"{label}.sqlite", out / f"{label}.sqlite")
print(f"synthetic cell {out.parent}")
