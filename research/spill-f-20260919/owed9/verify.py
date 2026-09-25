#!/usr/bin/env python3
"""OWED 9 CPU gate: storage-bench publishes io_ns and a stage line; the collector still joins it.

Usage: verify.py <storage-bench binary> <scratch dir on the filesystem under test> <out dir>
Runs roundtrip then restore for every (size, mode); raw stdout+stderr of every invocation is
written before parsing. Timings are development I/O characterization, never spill speed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tools/tier-battery.py").exists())
spec = importlib.util.spec_from_file_location("battery_owed9", ROOT / "tools/tier-battery.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
SIZES = [264, 4097, 1048576, 116654080]
MODES = ["buffered", "uncached", "direct"]
STAGE = re.compile(r"^\[storage-bench\] stages put_ns=(\d+) commit_ns=(\d+) lease_ns=(\d+) "
                   r"read_ns=(\d+) verify_ns=(\d+) total_ns=(\d+)$", re.M)


def main():
    binary, scratch, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    rows, envelopes, results = [], [], []
    for size in SIZES:
        for mode in MODES:
            obj = scratch / f"{mode}-{size}"
            for phase in ("roundtrip", "restore"):
                run_id = f"{phase}-{mode}-{size}"
                raw = out / f"{run_id}.log"
                with raw.open("xb") as log:
                    p = subprocess.run([str(binary), phase, str(obj), str(size), mode],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    log.write(p.stdout + b"\n--- stderr ---\n" + p.stderr)
                text_out, text_err = p.stdout.decode(), p.stderr.decode()
                assert p.returncode == 0, f"{run_id}: exit {p.returncode}: {text_err}"
                sample = json.loads(text_out.strip().splitlines()[-1])
                stage = STAGE.search(text_err)
                assert stage, f"{run_id}: no stage line"
                put, commit, lease, read, verify, total = map(int, stage.groups())
                assert sample["io_ns"] == read, f"{run_id}: io_ns {sample['io_ns']} != read_ns {read}"
                assert sample["total_ns"] == total and read + verify <= total
                assert sample["status"] == "byte-exact" and sample["fallbacks"] == 0
                assert (put == 0 and commit == 0) == (phase == "restore")
                rows.append({"run_id": run_id})
                envelopes.append({"run_id": run_id, "sample": sample})
                results.append({"run_id": run_id, "backend_actual": sample["backend_actual"],
                                "valid_bytes": sample["valid_bytes"], "io_ns": read,
                                "put_ns": put, "commit_ns": commit, "verify_ns": verify,
                                "total_ns": total, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()})
    joined = B.join_storage(rows, envelopes)
    assert len(joined) == len(rows) and all(j["samples"][0]["io_ns"] is not None for j in joined)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    shutil.rmtree(scratch)
    print(f"OWED9 VERIFY PASS: {len(rows)} invocations (2 phases x {len(MODES)} modes x {len(SIZES)} sizes), "
          f"io_ns == stage read_ns in every sample, collector join_storage accepted all {len(joined)}")


if __name__ == "__main__":
    main()
