#!/usr/bin/env python3
"""CPU checks only; a Linux cargo check is not a linked Linux test or GPU result."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "day3-checks"
OUT.mkdir(exist_ok=True)
checks = [
    ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
    ("check-mac", ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets"]),
    ("check-linux", ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"]),
    ("test", ["cargo", "test", "-p", "memra-kv", "-p", "memra-tier", "--offline"]),
    ("clippy", ["cargo", "clippy", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("diff", ["git", "diff", "--check"]),
    ("flags", ["bash", "tools/check-flags.sh"]),
    ("patch", ["git", "apply", "--check", "research/spill-b-20260919/HOSTPREFIX-PATCH.diff"]),
    ("shell", ["bash", "-n", "research/spill-b-20260919/rig-cells-b.sh", "research/spill-b-20260919/rig-stub-b.sh"]),
    ("runner", ["python3", "research/spill-b-20260919/test-rig-cells-b.py"]),
    ("frozen", ["git", "diff", "--exit-code", "98e558dc", "--", "crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts"]),
    ("runtime-unapplied", ["git", "diff", "--exit-code", "98e558dc", "--", "crates/memra-server/src/worker.rs", "crates/memra-server/src/admit_memory.rs", "crates/memra-server/src/worker/host_glm.rs"]),
]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
rows = []
for name, command in checks:
    path = OUT / f"{name}.log"
    with path.open("wb") as raw:
        result = subprocess.run(command, cwd=ROOT, stdout=raw, stderr=subprocess.STDOUT, check=False)
    # Raw file is complete before any parsing/hash/summary.
    data = path.read_bytes()
    archive = path.with_suffix(".log.gz")
    archive.write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    row = {"check": name, "command": command, "exit": result.returncode, "head": head, "raw": str(archive.relative_to(HERE)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "uncompressed_sha256": hashlib.sha256(data).hexdigest()}
    rows.append(row)
    print(f"{name}: exit {result.returncode}", flush=True)
    if result.returncode:
        print(data.decode(errors="replace"), flush=True)
(OUT / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
sys.exit(int(any(row["exit"] for row in rows)))
