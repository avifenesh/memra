#!/usr/bin/env python3
"""CPU and syntax verification only. Never builds engine or contacts a rig."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[2]
out = root / "research/spill-c-20260919/raw/day3"
out.mkdir(parents=True, exist_ok=True)
commands = [
    ["cargo", "fmt", "--all", "--", "--check"],
    ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets"],
    ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"],
    ["cargo", "test", "-p", "memra-tier", "--offline"],
    ["cargo", "clippy", "-p", "memra-tier", "--offline", "--all-targets", "--", "-D", "warnings"],
    ["git", "diff", "--check"],
    ["git", "diff", "--cached", "--check"],
    ["bash", "tools/check-flags.sh"],
    ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"],
    ["bash", "-n", "research/spill-c-20260919/rig-cells-c.sh"],
    ["python3", "research/spill-c-20260919/test-rig-cells.py"],
    ["python3", "research/spill-c-20260919/trace-policy.py", "--check"],
    ["git", "apply", "--check", "research/spill-c-20260919/HY3-DISPATCH-PATCH.diff"],
]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
rows = []
for i, argv in enumerate(commands):
    path = out / f"verify-{i+1}.log"
    with path.open("wb") as log:
        result = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    # Preserve diagnostic contents; normalize surplus terminal blank lines only.
    path.write_bytes(path.read_bytes().rstrip(b"\n") + b"\n")
    rows.append({"argv": argv, "exit": result.returncode, "log": path.name,
                 "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    print(f"exit={result.returncode} {' '.join(argv)}")
files = sorted((root / "crates/memra-tier/src/bank").glob("*.rs"))
files += sorted((root / "crates/memra-tier/tests/bank").glob("*.rs"))
files += sorted((root / "research/spill-c-20260919").glob("*.py"))
files += sorted((root / "research/spill-c-20260919/fixtures").glob("*.json"))
files += [root / p for p in ["research/spill-c-20260919/rig-cells-c.sh",
    "research/spill-c-20260919/HY3-DISPATCH-PATCH.diff", "crates/memra-tier/src/contracts.rs",
    "crates/memra-tier/src/tier/governor.rs", "crates/memra-tier/src/io/transfer.rs",
    "crates/memra-engine/src/hybrid.rs", "crates/memra-engine/src/hybrid_forward.rs"]]
receipt = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_head": head,
    "scope": "CPU-only; native engine unchanged; Linux cross-target is check only, not Linux execution",
    "commands": rows, "files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(1 if any(row["exit"] for row in rows) else 0)
