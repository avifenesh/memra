#!/usr/bin/env python3
"""Exercise dry-run success, partial status, failure propagation and reuse hygiene."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parents[2]
lane = root / "research/spill-c-20260919"
raw = lane / "raw"
raw.mkdir(exist_ok=True)
runner = "research/spill-c-20260919/rig-cells-c.sh"


def run(label, extra, expected):
    before = set(raw.glob(label + "-*"))
    result = subprocess.run(["bash", runner, "--dry-run", "--host-label", label, *extra],
                            cwd=root, capture_output=True, timeout=30)
    assert result.returncode == expected, result.stderr.decode()
    created = set(raw.glob(label + "-*")) - before
    assert len(created) == 1, created
    directory = created.pop()
    (directory / "driver.log").write_bytes(result.stdout + result.stderr)
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    assert rows and all(row["dry_run"] for row in rows)
    assert all(row["status"] in ("STUB", "BLOCKED") for row in rows)
    assert rows[0]["cell"] == "build"
    assert rows[1]["argv"][1] in ("/tmp/memra-5090.lock", "/tmp/memra-gpu.lock")
    return directory, rows


first, rows = run("stub-test5090", [], 0)
assert next(r for r in rows if r["cell"] == "hy3-pro-pair")["exit"] == 3
second, _ = run("stub-test5090", [], 0)
assert first != second
_, rows = run("stub-testpro", ["--rig", "pro-pair", "--hy3-artifact", "/fixture/hy3",
                               "--hy3-manifest", "/fixture/SHA256SUMS"], 0)
assert [r["cell"] for r in rows].index("ple-tiny") < [r["cell"] for r in rows].index("hy3-gen")
assert next(r for r in rows if r["cell"] == "hy3-spec")["argv"][:3] == ["env", "-u", "MEMRA_SPEC_K"]
_, rows = run("stub-testred", ["--stub-fail", "ple-tiny"], 23)
assert rows[-1]["cell"] == "ple-tiny" and rows[-1]["exit"] == 23
assert "injected stub failure" in (raw / sorted(p.name for p in raw.glob("stub-testred-*"))[-1] / "ple-tiny.log").read_text()
# No confirmation => refusal BEFORE even a build; no rig or flock accessed.
refusal = subprocess.run(["bash", runner], cwd=root, capture_output=True, timeout=10)
assert refusal.returncode == 2 and b"non-serving-confirmed" in refusal.stderr
print("rig-cells-c: dry-run 5090 + PRO branches, repeated invocation, injected exit 23, and non-serving refusal PASS; no GPU commands executed")
