#!/usr/bin/env python3
"""DAY80: synthetic cells for the two readers' mechanics dry checks.
`regtime`: from DAY75's target `i16` receipts, `ref` as `ref`, `i15` as both `d` and `dr` (the `dr` copies' pool line
given the ` registered` ending). `regpool`: from DAY78's BOX34 `pages` receipts, `dpi` relabelled `dri` (its pool line's
` pageable` ending made ` registered`), census runs c1-di and c1-dpi (as c1-dri) kept. The readings mean nothing.
usage: make-synthetic.py regtime|regpool <src-cell-dir> <out-cell-dir>
"""
import shutil
import sys
from pathlib import Path

mode, src, out = sys.argv[1], Path(sys.argv[2]) / "ev", Path(sys.argv[3]) / "ev"
out.mkdir(parents=True, exist_ok=True)
if mode == "regtime":
    names = {"ref": ["ref"], "i15": ["d", "dr"]}
else:
    names = {"refi": ["refi"], "di": ["di"], "dpi": ["dri"]}


def targets(stem):
    parts = stem.split("-")
    if len(parts) == 3 and parts[0] in ("o1", "o2", "c1", "c2"):
        if mode == "regpool" and (parts[0] == "c2" or (parts[0] == "c1" and parts[1] == "refi")):
            return []
        return [f"{parts[0]}-{a}-{parts[2]}" for a in names.get(parts[1], [])]
    return None


for f in src.iterdir():
    stem, _, ext = f.name.partition(".")
    t = targets(stem)
    if t is None:
        if not (mode == "regtime" and f.suffix in (".sqlite", ".tsv") and stem.startswith("p-")):
            shutil.copy(f, out / f.name)
        continue
    for target in t:
        data = f.read_bytes()
        if ext == "log" and target.split("-")[1] in ("dr", "dri"):
            lines = data.split(b"\n")
            for n, line in enumerate(lines):
                if b"host pinned pool" in line:
                    lines[n] = line.replace(b" pageable", b"") + b" registered"
            data = b"\n".join(lines)
        (out / f"{target}.{ext}").write_bytes(data)
marks = []
for line in (src / "marks.tsv").read_text().splitlines():
    t, m = line.split("\t")
    lab, _, rest = m.partition(" ")
    for target in targets(lab) or []:
        marks.append(f"{t}\t{target} {rest}")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic {mode} cell {out.parent}")
