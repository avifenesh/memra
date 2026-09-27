#!/usr/bin/env python3
"""Day 52 views (research/spill-c-20260919/DAY52.md section 1): each improvement rung's registered reader, unchanged,
on the target card's ladder cell. For every day 43 to 50 this builds a view directory holding only that day's arms
(symlinks to the ladder's run logs and exit files under that day's arm names, and the ladder's marks for those runs
renamed the same way), runs the day's reader on it with --rig, and writes its output beside the ladder.

usage: day52-views.py <ladder-cell-dir> [--rig NAME] [--days 44,45]
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# day -> (reader, {day's arm name: ladder arm})
VIEWS = {
    43: ("day43-resid.py", {"off": "off", "base": "base", "i6d": "i6d", "i6g": "i6g"}),
    44: ("day44-mapped.py", {"off": "off", "i6": "i6g", "i9": "i9g"}),
    45: ("day45-fill.py", {"off": "off", "i9g": "i9g", "fill": "fill"}),
    46: ("day46-nodrain.py", {"off": "off", "fill": "fill", "i1": "i1"}),
    47: ("day47-pinned.py", {"off": "off", "i1": "i1", "i2": "i2"}),
    48: ("day48-small.py", {"off": "off", "i2": "i2", "i8": "i8", "i5": "i5"}),
    49: ("day49-install.py", {"off": "off", "i5": "i5", "i7": "i7"}),
    50: ("day50-prefetch.py", {"off": "off", "i7": "i7", "i4": "i4"}),
    57: ("day57-fillwait.py", {"off": "off", "i4": "i4", "i10": "i10"}),
}


def build_view(ladder_ev, view_ev, arms):
    back = {ladder: day for day, ladder in arms.items()}
    marks = []
    for row in (ladder_ev / "marks.tsv").read_text().splitlines():
        stamp, label = row.split("\t", 1)
        run, rest = label.split(" ", 1)
        order, arm, rep = run.split("-")
        if arm in back:
            marks.append(f"{stamp}\t{order}-{back[arm]}-{rep} {rest}")
    (view_ev / "marks.tsv").write_text("\n".join(marks) + "\n")
    for log in sorted(ladder_ev.glob("o[12]-*-r[1-5].log")):
        order, arm, rep = log.stem.split("-")
        if arm not in back:
            continue
        stem = f"{order}-{back[arm]}-{rep}"
        os.symlink(log.resolve(), view_ev / f"{stem}.log")
        os.symlink((ladder_ev / f"{log.stem}.exit").resolve(), view_ev / f"{stem}.exit")


def main():
    ladder = Path(sys.argv[1]).resolve()
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    out = ladder / "views"
    out.mkdir(exist_ok=True)
    worst = 0
    # DAY52 section 10: `--days 44,45` reads only those views (the ladder-b cell).
    days = set(int(d) for d in sys.argv[sys.argv.index("--days") + 1].split(",")) if "--days" in sys.argv else None
    for day, (reader, arms) in VIEWS.items():
        if days is not None and day not in days:
            continue
        with tempfile.TemporaryDirectory(prefix=f"day52-view{day}-") as tmp:
            view_ev = Path(tmp) / "ev"
            view_ev.mkdir()
            build_view(ladder / "ev", view_ev, arms)
            res = subprocess.run([sys.executable, str(HERE / reader), tmp, "--rig", rig],
                                 capture_output=True, text=True)
        text = (f"DAY52 VIEW day={day} reader={reader} arms="
                + ",".join(f"{d}<-{l}" for d, l in arms.items()) + f" rc={res.returncode}\n"
                + res.stdout + res.stderr)
        (out / f"day{day}.log").write_text(text)
        sys.stdout.write(text)
        worst = max(worst, res.returncode)
    return worst


if __name__ == "__main__":
    sys.exit(main())
