#!/usr/bin/env python3
"""sanitize_receipts.py DIR: memra#1266. The published receipts and harness name local paths and
hosts generically, as earlier lanes' receipts do: the model directories become <models>, this
lane's scratch directory <lane>, the home directory ~, the GB10 units u1/u2 and the CPU build host
"build host". Rewrites text files and gzipped logs in place, recursively. Strips trailing blanks from text lines. Prints what it changed."""
import gzip
import pathlib
import sys

SUBS = [
    ("<models>", "<models>"),
    ("<models>", "<models>"),
    ("<models>", "<models>"),
    ("<models>", "<models>"),
    ("<lane>", "<lane>"),
    ("<lane>", "<lane>"),
    ("<lane>", "<lane>"),
    ("<memra-1221b-lane>", "<memra-1221b-lane>"),
    ("<memra-1221b-lane>", "<memra-1221b-lane>"),
    ("~", "~"),
    ("spark-u2", "spark-u2"),
    ("spark-u1", "spark-u1"),
    ("u2", "u2"),
    ("u1", "u1"),
    ("ssh build-host", "ssh build-host"),
    ("build-host:", "build-host:"),
    ("CPU build host", "CPU build host"),
    ("CPU build host", "CPU build host"),
    ("build-host", "build-host"),
]


def fix(text):
    out = text
    for a, b in SUBS:
        out = out.replace(a, b)
    return out


root = pathlib.Path(sys.argv[1])
for p in sorted(root.rglob("*")):
    if not p.is_file():
        continue
    if p.suffix == ".gz":
        raw = gzip.decompress(p.read_bytes()).decode(errors="surrogateescape")
        new = fix(raw)
        if new != raw:
            p.write_bytes(gzip.compress(new.encode(errors="surrogateescape"), mtime=0))
            print("gz", p)
        continue
    try:
        raw = p.read_text()
    except UnicodeDecodeError:
        continue
    new = fix(raw)
    # Trailing blanks (an empty `extra:` field in guard.log) trip `git diff --check`.
    new = "\n".join(line.rstrip(" \t") for line in new.split("\n"))
    if new != raw:
        p.write_text(new)
        print("txt", p)
    name = fix(p.name)
    if name != p.name:
        p.rename(p.with_name(name))
        print("renamed", p, "->", name)
