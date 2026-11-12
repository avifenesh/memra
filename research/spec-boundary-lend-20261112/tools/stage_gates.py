#!/usr/bin/env python3
"""stage_gates.py SRC DST: stage the exactness gates' runs as receipts (memra#1221 part 2).
Verdicts, plans, summaries, rows and guards are copied; every log is gzipped; a gate's texts
directory becomes texts.tar.gz. The home directory becomes `~` in every file."""
import gzip
import io
import pathlib
import sys
import tarfile

HOME = str(pathlib.Path.home()).encode()
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)


def scrub(data: bytes) -> bytes:
    return data.replace(HOME, b"~")


texts_dirs = sorted(p for p in src.rglob("texts") if p.is_dir())
for t in texts_dirs:
    out = dst / t.relative_to(src).with_name("texts.tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz", compresslevel=9) as tar:
        for f in sorted(t.iterdir()):
            data = scrub(f.read_bytes())
            info = tarfile.TarInfo(f"texts/{f.name}")
            info.size = len(data)
            info.mtime = int(f.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))

for p in sorted(src.rglob("*")):
    if p.is_dir() or any(t in p.parents for t in texts_dirs):
        continue
    out = dst / p.relative_to(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = scrub(p.read_bytes())
    if p.suffix == ".log":
        with gzip.open(out.with_name(p.name + ".gz"), "wb", 9) as g:
            g.write(data)
    else:
        out.write_bytes(data)

total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
print(f"staged {src} -> {dst}: {total / 1e6:.2f} MB")
