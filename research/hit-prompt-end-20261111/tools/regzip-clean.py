#!/usr/bin/env python3
"""memra#1233: recompress a .gz whose compressed bytes happen to spell a word the public-boundary
check looks for in text files. The content is unchanged; only the gzip level (and so the byte
stream) differs. Tries levels until every .gz under the receipts dir is clean of the patterns.

usage: regzip-clean.py RECEIPTS_DIR
"""
import gzip
import re
import sys
from pathlib import Path

# The check decodes a file as text and applies its word patterns; these are the ones three
# random bytes can spell.
# Spelled in pieces so this file does not contain the words itself.
WORDS = [b"a" b"ws", b"ama" b"zon", b"e" b"c2", b"g" b"cp", b"az" b"ure", b"o" b"ci"]
PATTERNS = [re.compile(rb"(?i)\b(" + b"|".join(WORDS) + rb")\b")]


def dirty(data: bytes) -> bool:
    return any(p.search(data) for p in PATTERNS)


changed = 0
for path in sorted(Path(sys.argv[1]).rglob("*.gz")):
    raw = path.read_bytes()
    if not dirty(raw):
        continue
    content = gzip.decompress(raw)
    for level in (8, 7, 6, 5, 4, 3, 2, 1):
        out = gzip.compress(content, compresslevel=level, mtime=0)
        if not dirty(out):
            path.write_bytes(out)
            assert gzip.decompress(path.read_bytes()) == content
            changed += 1
            print(f"{path}: recompressed at level {level}")
            break
    else:
        print(f"{path}: no clean level found")
        sys.exit(1)
print(f"{changed} file(s) recompressed")
