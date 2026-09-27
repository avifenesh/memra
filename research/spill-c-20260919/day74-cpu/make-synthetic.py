#!/usr/bin/env python3
"""DAY74: a synthetic `induce` cell for the reader's mechanics dry check, relabelled from machine b's DAY71 receipts:
per order, REF runs 1 to 3 as `ref`, 4 to 6 as `refi`, door runs 1 to 3 as `i15`, 4 to 6 as `i15i`, with its sampler
rows and an invented inducer.txt. The arms' meanings are invented; the reading means nothing.
usage: make-synthetic.py <day71-b-cell-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
names = {}
for o in (1, 2):
    for i in range(1, 7):
        names[f"o{o}-ref-r{i}"] = f"o{o}-{'ref' if i <= 3 else 'refi'}-r{(i - 1) % 3 + 1}"
        names[f"o{o}-i15-r{i}"] = f"o{o}-{'i15' if i <= 3 else 'i15i'}-r{(i - 1) % 3 + 1}"
for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    if stem in names:
        shutil.copy(f, out / f"{names[stem]}.{ext}")
    elif not stem.startswith("o"):
        shutil.copy(f, out / f.name)
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    if lab in names:
        marks.append(f"{t}\t{names[lab]} {rest}")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
(out / "inducer.txt").write_text("memfree_gib=80 F_gib=64 induce=1\n")
print(f"synthetic cell {out.parent}: 24 runs relabelled from {src}")
