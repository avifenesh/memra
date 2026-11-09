#!/usr/bin/env python3
"""Copy one ab.py output directory into this lane's receipts: summary, per-boot results, log
lines, raw bodies and the server log, with the home directory shown as `~`.

Usage: bank_ab.py SRC_DIR DEST_DIR
"""
import re
import shutil
import sys
from pathlib import Path

HOME = str(Path.home())
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{8,}")


def scrub(text: str) -> str:
    return GPU_UUID.sub("GPU-<uuid>", text.replace(HOME, "~"))


def main() -> int:
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=False)
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".json", ".jsonl", ".log"):
            out.write_text(scrub(path.read_text()))
        else:
            shutil.copyfile(path, out)
    print(f"banked {src} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
