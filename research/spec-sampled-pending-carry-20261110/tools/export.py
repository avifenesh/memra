#!/usr/bin/env python3
"""Copy this lane's receipts from the scratch run directories into the research directory
and scrub them for the public tree: the home directory becomes "~", device lease lines are
dropped, the gate's scratch output path becomes its name. Texts and response bodies are not
copied (rows carry a sha of each text).

usage: export.py SCRATCH_RUN_DIR DEST_DIR   (e.g. .../run/gb10 research/.../gb10)
"""
import pathlib
import re
import shutil
import sys

SRC, DST = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
KEEP = re.compile(r"(\.jsonl|\.log|\.csv|stops.*\.json|\.txt|\.rc)$")
SKIP_DIRS = {"texts"}
home = str(pathlib.Path.home())
n = 0
for p in sorted(SRC.rglob("*")):
    rel = p.relative_to(SRC)
    if not p.is_file() or SKIP_DIRS & set(rel.parts) or ".old." in str(rel) or not KEEP.search(p.name):
        continue
    if p.suffix == ".json" and not p.name.startswith("stops"):
        continue
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        t = p.read_text()
    except UnicodeDecodeError:
        shutil.copyfile(p, out)
        n += 1
        continue
    t = t.replace(home, "~")
    t = re.sub(r"/home/[A-Za-z0-9_.-]+", "~", t)
    t = re.sub(r"(?m)^(gpulock|sparklock): .*\n", "", t)
    t = re.sub(r"(?m)^(receipts: )\S*/([^/\s]+)$", r"\1\2/", t)
    t = re.sub(r"(?m)[ \t]+$", "", t)
    out.write_text(t)
    n += 1
print("copied", n, "files to", DST)
