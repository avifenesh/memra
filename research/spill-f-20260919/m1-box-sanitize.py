#!/usr/bin/env python3
"""Replace the rented volume id in mirrored receipts (run from the box27 directory); originals
are copied to the private directory first and every change is appended to EXPORT-MANIFEST.json."""
import hashlib
import json
import shutil
from pathlib import Path

P = Path("/home/avifenesh/.local/share/memra-lane-f-private/box27/originals")
TOKEN, REPL = b"V.52617309", b"V.<volume-id>"
m = json.loads(Path("EXPORT-MANIFEST.json").read_text())
n = 0
for f in sorted(Path(".").rglob("*")):
    if not f.is_file():
        continue
    data = f.read_bytes()
    if TOKEN not in data:
        continue
    dest = P / f
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest)
    f.write_bytes(data.replace(TOKEN, REPL))
    m["exported"].append({"path": str(f), "original_sha256": hashlib.sha256(data).hexdigest(),
                          "export_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                          "change": "rented volume id replaced by V.<volume-id>"})
    n += 1
Path("EXPORT-MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
print(f"{n} sanitized, {len(m['exported'])} total")
