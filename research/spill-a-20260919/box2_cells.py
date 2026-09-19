#!/usr/bin/env python3
"""One development-overlay cell per invocation; sync its receipt before the next.

Runs on the box, not an SSH launcher. No endpoint or rental identity belongs here.
The collector, never this launcher, owns the canonical GPU campaign lock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LABEL = "overlay, development, not spill speed"
WORKER = "spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read"


def plan(cell, scratch):
    cargo = ["cargo", "test", "--release", "-p", "memra-tier", "--offline", "-j", "16"]
    fixed = {
        "build": ["cargo", "build", "--release", "-p", "memra-engine", "--bin", "storage-bench", "-j", "16"],
        "build-worker": ["cargo", "test", "--release", "-p", "memra-engine", "--lib", "--no-run", "-j", "16"],
        "storage-tests": cargo + ["--test", "storage", "--", "--nocapture"],
        "gc": cargo + ["--test", "storage", "review_", "--", "--nocapture"],
        "gc-upgrade": cargo + ["--lib", "review_gc_upgrade_window", "--", "--nocapture"],
        "catalog": cargo + ["--test", "storage", "day4::sharded_catalog_200gb_metadata_only_constant_touch_cost", "--", "--exact", "--nocapture"],
        "pinned": ["cargo", "test", "--release", "-p", "memra-engine", "--lib", "-j", "16", WORKER, "--", "--ignored", "--exact", "--nocapture"],
    }
    if cell in fixed:
        return fixed[cell], cell == "pinned"
    parts = cell.split("-")
    if len(parts) != 3 or parts[0] not in ("buffered", "uncached", "direct") or parts[1] not in ("roundtrip", "restore") or parts[2] not in ("264", "4097", "1048576", "4194568"):
        raise ValueError("unknown cell")
    backend, operation, size = parts
    return ["target/release/storage-bench", operation, str(scratch / (backend + "-" + size)), size, backend], True


def command_for(cell, scratch, out):
    command, collected = plan(cell, scratch)
    if collected:
        # Keep nested Cargo -- separators away from collector argparse.REMAINDER.
        return ["python3", "tools/tier-battery.py", "--rig", "rtx5090", "--timeout", "300",
                "--out", str(out / "collector"), "--run-id", cell, "--execute", "bash", "-c", shlex.join(command)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="owned persistent scratch directory (overlay)")
    parser.add_argument("--out", type=Path, required=True, help="new receipt directory, one per cell")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--approved-non-serving", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scratch, out = args.directory.absolute(), args.out.absolute()
    command = command_for(args.cell, scratch, out)
    if args.dry_run:
        print(json.dumps({"cell": args.cell, "command": command, "storage_label": LABEL,
                          "qualification": False, "executed": False}))
        return
    if platform.system() != "Linux" or not args.approved_non_serving:
        raise ValueError("requires Linux and explicit non-serving approval")
    for executable in ("cargo", "python3", "nvidia-smi", "findmnt"):
        if shutil.which(executable) is None:
            raise ValueError("missing prerequisite: " + executable)
    os.chdir(ROOT)
    subprocess.run(["git", "diff", "--exit-code", "--quiet"], check=True)
    subprocess.run(["git", "diff", "--cached", "--exit-code", "--quiet"], check=True)
    # Refuse an accidental NVMe/other filesystem claim in this explicitly overlay mode.
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(scratch.parent)], text=True).strip()
    if filesystem != "overlay":
        raise ValueError("--box2 mode requires overlay; use the NVMe launcher for proven NVMe")
    scratch.mkdir(parents=False, exist_ok=True)
    marker = scratch / ".spill-a-owned"
    if not marker.is_file():
        if any(scratch.iterdir()):
            raise ValueError("refusing nonempty scratch without A ownership marker")
        marker.write_text("WP-A development scratch; remove after receipts are synced\n")
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # The collector supports private operator fields; public lane receipts don't.
    for key in ("RUNPOD_POD_ID", "CONTAINER_ID"):
        env.pop(key, None)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {"runtime_commit": head, "storage_label": LABEL, "cell": args.cell,
                "command": command, "qualification": False, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    binary = ROOT / "target/release/storage-bench"
    if binary.is_file():
        metadata["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Record mount TYPES/TARGETS only: source/host identifiers are private.
    with (out / "mounts.log").open("xb") as log:
        subprocess.run(["findmnt", "-rn", "-o", "TARGET,FSTYPE"], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "power-before.log").open("xb") as log:
        subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,power.max_limit,power.draw",
                        "--format=csv"], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "compute-before.log").open("xb") as log:
        state = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(state.stdout)
    collected = plan(args.cell, scratch)[1]
    if collected:
        with (out / "processes-before.log").open("xb") as log:
            subprocess.run(["pgrep", "-fa", "run-gen|run-spec|qwen4exp|storage-bench|pp-transport"],
                           stdout=log, stderr=subprocess.STDOUT, check=False)
        if state.returncode or state.stdout.strip():
            raise ValueError("GPU busy or compute-app inventory failed; no cell started")
    started = time.monotonic_ns()
    with (out / "command.log").open("xb") as log:
        try:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800)
            code = run.returncode
        except subprocess.TimeoutExpired:
            log.write(b"ERROR: cell timed out after 1800 seconds\n")
            code = 124
    # Parse/report only after raw stdout/stderr is retained.
    result = dict(metadata, exit_code=code, wall_ns=time.monotonic_ns()-started,
                  status="executed-not-qualified" if code == 0 else "failed")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("SYNC THIS CELL NOW before running another. " + LABEL)
    sys.exit(code if 0 <= code <= 125 else 2)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("BLOCKED: " + str(error), file=sys.stderr)
        sys.exit(2)
