#!/usr/bin/env python3
"""CPU checks with exact source/raw hashes; never labels them native GPU evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "day5-checks")
if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode:
    raise SystemExit("refuse: commit tracked source before exact-tip verification")
OUT.mkdir(exist_ok=False)
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
rows = []

def run(name, command):
    path = OUT / f"{name}.log"
    with path.open("wb") as raw:
        result = subprocess.run(command, cwd=ROOT, stdout=raw, stderr=subprocess.STDOUT, check=False)
    data = path.read_bytes()  # Raw complete before hashes/summary.
    archive = path.with_suffix(".log.gz")
    archive.write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    rows.append({"check": name, "command": command, "exit": result.returncode,
                 "head": HEAD, "raw": str(archive.relative_to(HERE)),
                 "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                 "uncompressed_sha256": hashlib.sha256(data).hexdigest()})
    print(f"{name}: exit {result.returncode}", flush=True)
    if result.returncode:
        print(data.decode(errors="replace"), flush=True)
    return result.returncode

for name, cmd in [
    ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
    ("check-mac", ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets"]),
    ("check-linux", ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"]),
    ("test", ["cargo", "test", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--no-fail-fast"]),
    ("clippy", ["cargo", "clippy", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("diff", ["git", "diff", "--check"]),
    ("lane-diff", ["git", "diff", "--check", "cc2a7df638fc4ba84d5545c7b0685a60a10229cf"]),
    ("flags", ["bash", "tools/check-flags.sh"]),
    ("patch", ["git", "apply", "--check", "research/spill-b-20260919/HOSTPREFIX-PATCH.diff"]),
    ("runner", ["python3", "research/spill-b-20260919/test-rig-cells-b.py"]),
    ("runtime-unapplied", ["git", "diff", "--exit-code", "fb870849", "--", "crates/memra-server/src/worker.rs", "crates/memra-server/src/admit_memory.rs", "crates/memra-server/src/worker/host_glm.rs"]),
    ("shared-unchanged", ["git", "diff", "--exit-code", "fb870849", "--", "Cargo.toml", "Cargo.lock", "crates/memra-engine/Cargo.toml", "crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts", "crates/memra-tier/src/peer"]),
]:
    run(name, cmd)
with tempfile.TemporaryDirectory(prefix="memra-b-cli-") as temp:
    binary = str(Path(temp) / "cli-tests")
    if run("gate-cli-build", ["rustc", "--edition", "2024", "--test", "crates/memra-engine/src/bin/kv_tier_gate/cli.rs", "-o", binary]) == 0:
        run("gate-cli-test", [binary])
(OUT / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
sys.exit(int(any(row["exit"] for row in rows)))
