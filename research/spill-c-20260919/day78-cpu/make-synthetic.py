#!/usr/bin/env python3
"""DAY78: a synthetic `pages` cell for the reader's mechanics dry check, from DAY76's BOX32 `chunk` receipts: `refi` and
`di` as they are, `dci` relabelled `dpi` (its pool line made the pageable form), and six census runs made of copies of
the first run of each arm with a page census file copied from the local dry check. The reading means nothing.
usage: make-synthetic.py <chunk-cell-dir> <census.tsv> <out-cell-dir>
"""
import re
import shutil
import sys
from pathlib import Path

src, census, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]), Path(sys.argv[3]) / "ev"
out.mkdir(parents=True, exist_ok=True)
POOL = re.compile(rb"( chunk_bytes=\d+ allocations=\d+)")
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    marks.append(line.replace("-dci-", "-dpi-"))
for f in src.iterdir():
    name = f.name.replace("-dci-", "-dpi-")
    data = f.read_bytes()
    if "-dci-" in f.name and f.suffix == ".log":
        data = POOL.sub(b" pageable", data)
    (out / name).write_bytes(data)
for order, arms in (("c1", ("refi", "di", "dpi")), ("c2", ("dpi", "di", "refi"))):
    for arm in arms:
        for ext in (".log", ".exit"):
            shutil.copy(out / f"o1-{arm}-r1{ext}", out / f"{order}-{arm}-r1{ext}")
        shutil.copy(census, out / f"{order}-{arm}-r1.pages.tsv")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic cell {out.parent}")
