#!/usr/bin/env python3
"""memra#1255: copy the lane's receipts into the worktree's research dir, gzipping logs and
telemetry (> 20 kB). Idempotent: a destination is rewritten from its source each run.
usage: assemble.py SRC_RUN_DIR:DEST_SUBDIR ...   (DEST relative to the research dir)"""
import gzip
import shutil
import sys
from pathlib import Path

R = Path.home() / ".cache/memra-1255/wt/research/same-pass-admission-20261112"
SKIP = {"server.pgid", "cell.pgid"}
GZ_MIN = 20_000


def copy_run(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.name in SKIP:
            continue
        data = f.read_bytes()
        if f.suffix in (".log", ".csv") and len(data) > GZ_MIN:
            out = dest / (f.name + ".gz")
            (dest / f.name).unlink(missing_ok=True)
            with gzip.GzipFile(out, "wb", mtime=0, compresslevel=9) as g:
                g.write(data)
        else:
            out = dest / f.name
            shutil.copyfile(f, out)
        total += out.stat().st_size
    return total


def main() -> None:
    grand = 0
    for spec in sys.argv[1:]:
        src, dest = spec.split(":", 1)
        n = copy_run(Path(src), R / dest)
        grand += n
        print(f"{dest}: {n / 1e3:.0f} kB")
    print(f"copied {grand / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
