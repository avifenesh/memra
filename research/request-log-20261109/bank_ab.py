#!/usr/bin/env python3
"""Copy one ab.py output directory into this lane's receipts: summary, per-boot results, log
lines and the server log, with the home directory shown as `~`, and the raw response bodies as
one `raw-bodies.tar.gz` (SSE bodies end in a blank line by protocol; an archive keeps their
bytes exact without tripping `git diff --check`).

Usage: bank_ab.py SRC_DIR DEST_DIR
"""
import gzip
import io
import re
import sys
import tarfile
from pathlib import Path

HOME = str(Path.home())
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{8,}")


def scrub(text: str) -> str:
    return GPU_UUID.sub("GPU-<uuid>", text.replace(HOME, "~"))


def main() -> int:
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=False)
    bodies = io.BytesIO()
    with tarfile.open(fileobj=bodies, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(src.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            if path.suffix == ".body":
                data = path.read_bytes()
                info = tarfile.TarInfo(str(rel))
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(scrub(path.read_text()))
    (dest / "raw-bodies.tar.gz").write_bytes(gzip.compress(bodies.getvalue(), mtime=0))
    print(f"banked {src} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
