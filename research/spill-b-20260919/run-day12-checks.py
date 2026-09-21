#!/usr/bin/env python3
"""Run the day-11 CPU battery under the owner's CPU quota and seal exit codes and log hashes.

No GPU. Writes research/spill-b-20260919/day12-checks/final/{check-NN.log,checks.json}.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "day12-checks" / "final"
QUOTA = ["systemd-run", "--user", "--scope", "-q", "-p", "CPUQuota=1200%", "-p", "MemoryMax=28G"]
CHECKS = [
    ["cargo", "fmt", "--all", "--", "--check"],
    ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--all-targets", "--offline"],
    ["cargo", "test", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--no-fail-fast"],
    ["cargo", "clippy", "-p", "memra-engine", "-p", "memra-tier", "-p", "memra-kv", "--offline",
     "--all-targets", "--", "-D", "warnings"],
    ["python3", "research/spill-b-20260919/test-day12.py"],
    ["python3", "research/spill-b-20260919/verify-day12.py", "--require-complete"],
    ["git", "diff", "--check"],
    ["bash", "tools/check-flags.sh"],
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quota = QUOTA if "--no-quota" not in sys.argv else []
    checks = []
    for index, command in enumerate(CHECKS):
        log = OUT / f"check-{index:02d}.log"
        with log.open("wb") as handle:
            code = subprocess.run(quota + command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        checks.append({"command": command, "exit": code, "log": log.name,
                       "sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        print(code, " ".join(command), flush=True)
    source = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "checks.json").write_text(json.dumps({"source": source, "cpu_quota": " ".join(QUOTA) if quota else "none",
                                                "checks": checks}, indent=2) + "\n")
    sys.exit(int(any(c["exit"] for c in checks)))


if __name__ == "__main__":
    main()
