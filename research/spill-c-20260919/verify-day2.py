#!/usr/bin/env python3
"""Bounded CPU verification. No shell evaluation, environment dump, GPU or network."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[2]
out = root / "research/spill-c-20260919/raw/day2"
out.mkdir(parents=True, exist_ok=True)
commands = [
    ["cargo", "fmt", "--all", "--", "--check"],
    ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets"],
    ["cargo", "test", "-p", "memra-tier", "--offline"],
    ["cargo", "clippy", "-p", "memra-tier", "--offline", "--all-targets", "--", "-D", "warnings"],
    ["git", "diff", "--check"],
    ["bash", "tools/check-flags.sh"],
    ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"],
]
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
rows = []
for index, cmd in enumerate(commands):
    name = f"verify-{index + 1}.log"
    with (out / name).open("wb") as log:
        result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    # Only surplus terminal newlines normalized; diagnostics retained verbatim.
    log = out / name
    log.write_bytes(log.read_bytes().rstrip(b"\n") + b"\n")
    rows.append({"argv": cmd, "exit": result.returncode, "log": name})
    print(f"exit={result.returncode} {' '.join(cmd)}")
files = sorted((root / "crates/memra-tier/src/bank").glob("*.rs"))
files += sorted((root / "crates/memra-tier/tests/bank").glob("*.rs"))
files += [root / p for p in [
    "crates/memra-tier/src/contracts.rs", "crates/memra-tier/src/lib.rs",
    "crates/memra-tier/Cargo.toml", "crates/memra-engine/src/banked_residency.rs",
    "research/spill-c-20260919/fixtures/ple-ngram-synthetic.json",
]]
receipt = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "head": sha, "commands": rows,
    "files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
(out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(1 if any(r["exit"] for r in rows) else 0)
