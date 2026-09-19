#!/usr/bin/env python3
"""Record only explicit cell arguments/files, never environment or credential files."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

out, head, dry, cell, status, log, *argv = sys.argv[1:]
raw = Path(log).read_bytes()
row = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "source_head": head,
    "dry_run": dry == "1",
    "cell": cell,
    "status": "BLOCKED" if argv and argv[0] == "BLOCKED" else ("STUB" if dry == "1" else "EXECUTED"),
    "exit": int(status),
    "argv": argv,
    "log": str(Path(log).name),
    "log_sha256": hashlib.sha256(raw).hexdigest(),
    "binary_sha256": None,
}
binary = argv[3] if argv[:3] == ["env", "-u", "MEMRA_SPEC_K"] else (argv[0] if argv else "")
if dry != "1" and binary and Path(binary).is_file():
    h = hashlib.sha256()
    with Path(binary).open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    row["binary_sha256"] = h.hexdigest()
with (Path(out) / "runs.jsonl").open("a") as destination:
    destination.write(json.dumps(row, sort_keys=True) + "\n")
