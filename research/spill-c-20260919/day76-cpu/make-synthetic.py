#!/usr/bin/env python3
"""DAY76: a synthetic `chunk` cell for the reader's mechanics dry check, relabelled from DAY74's induce-b receipts on
BOX29: per order, REF runs as `refi`, the door's induced runs and one uninduced one as `di`, the door's uninduced runs
and one induced one as `dci` (two sources used twice) (their pool line given an invented chunk field). The arms' meanings are invented; the
reading means nothing. usage: make-synthetic.py <induce-b-cell-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
names = {}
for o in ("o1", "o2"):
    plan = {"refi": [("refi", 1), ("refi", 2), ("refi", 3), ("ref", 1)],
            "di": [("i15i", 1), ("i15i", 2), ("i15i", 3), ("i15", 1)],
            "dci": [("i15", 2), ("i15", 3), ("i15i", 1), ("i15", 1)]}
    for arm, sources in plan.items():
        for k, (a, i) in enumerate(sources, 1):
            names.setdefault(f"{o}-{a}-r{i}", []).append(f"{o}-{arm}-r{k}")
for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    for target in names.get(stem, []):
        text = f.read_bytes()
        if target.split("-")[1] == "dci" and ext == "log":
            text = text.replace(b"fill_reserve=", b"chunk_marker_", 1)
            lines = text.split(b"\n")
            for n, line in enumerate(lines):
                if b"host pinned pool" in line:
                    lines[n] = line.replace(b"chunk_marker_", b"fill_reserve=") + b" chunk_bytes=268435456 allocations=102"
            text = b"\n".join(lines)
        (out / f"{target}.{ext}").write_bytes(text)
    if stem not in names and not stem.startswith("o"):
        shutil.copy(f, out / f.name)
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    for target in names.get(lab, []):
        marks.append(f"{t}\t{target} {rest}")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic cell {out.parent}: {sum(len(v) for v in names.values())} runs relabelled from {src}")
