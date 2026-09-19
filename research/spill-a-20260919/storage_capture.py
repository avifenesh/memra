#!/usr/bin/env python3
"""Hash-bound StorageSample/CELL join. Diagnostic capture, never a forced-pair gate.

The battery passes its own evidence/schema helpers; no duplicated StorageSample schema.
Empty or short GPU CSVs stay diagnostic, never become invented 250 ms tier counters.
"""
import json
import shlex
from pathlib import Path


def validate_storage_cell(journal, battery, envelopes=None):
    root = journal.parent
    require = battery.require
    rows, torn = battery.read_cell_journal(journal)
    require(not torn and len(rows) == 2, "storage cell needs one complete start/end attempt")
    start, end = rows
    require(start.get("event") == "start" and end.get("event") == "end", "storage cell event order")
    for key in ("kind", "run_id", "command", "started_utc", "qualification", "resume_from"):
        require(key in start and start[key] == end.get(key), "storage cell identity mismatch")
    require(start["kind"] == "CELL" and start["qualification"] is False, "not a diagnostic CELL")
    rid = start["run_id"]
    require(isinstance(rid, str) and bool(rid), "empty storage run id")
    require(type(end.get("exit_code")) is int and end["exit_code"] == 0
            and end.get("status") == "executed-not-qualified", "failed storage cell")
    capture_path = battery.evidence(root, end["capture"])
    capture = json.loads(capture_path.read_text())
    captured_command = start["command"]
    command = captured_command
    if isinstance(command, list) and len(command) == 3 and command[:2] == ["bash", "-c"]:
        command = shlex.split(command[2])
        require(shlex.join(command) == captured_command[2], "noncanonical shell wrapper")
    require(isinstance(command, list) and len(command) == 5
            and all(isinstance(x, str) for x in command)
            and Path(command[0]).name == "storage-bench"
            and command[1] in ("roundtrip", "restore")
            and command[4] in ("buffered", "uncached", "direct"), "not an exact storage-bench command")
    require(capture.get("schema_version") == 1 and capture.get("kind") == "subprocess-capture"
            and capture.get("command") == captured_command and capture.get("qualification") is False,
            "capture command/class mismatch")
    require(type(capture.get("exit_code")) is int and capture["exit_code"] == 0
            and capture.get("timed_out") is False and capture.get("parse_error") is None
            and capture.get("status") == "executed-not-qualified", "failed storage capture")
    first, last = capture.get("started_monotonic_ns"), capture.get("ended_monotonic_ns")
    require(type(first) is int and type(last) is int and 0 <= first < last
            and end.get("duration_ns") == last - first, "invalid capture window")
    lock = json.loads((root / "lock.json").read_text())
    require(lock.get("rig") in battery.LOCKS and lock.get("rig") != "cpu"
            and lock.get("lock") == battery.LOCKS[lock["rig"]]
            and lock.get("acquired") is True, "missing canonical collector lock")
    raw = battery.evidence(root, capture["raw_log"])
    lines = raw.read_text().splitlines()
    require(len(lines) == 1, "storage stdout must contain exactly one sample")
    sample = json.loads(lines[0])
    expected = [{"run_id": rid, "sample": sample}]
    joined = battery.join_storage([{"run_id": rid}], expected)
    if envelopes is not None:
        battery.join_storage([{"run_id": rid}], envelopes)
        require(envelopes == expected, "storage envelope differs from captured stdout")
    require(0 < sample["valid_bytes"] == int(command[3]) <= 1 << 30,
            "storage command/sample length mismatch")
    require(sample["backend_requested"] == command[4], "storage requested backend mismatch")
    require(sample["total_ns"] <= last - first, "sample exceeds collector window")
    # Preserve unknown physical/transfer counters and every status/fallback. A
    # successful subprocess does not turn a non-byte-exact sample into a gate pass.
    telemetry = capture["gpu_telemetry"]
    diagnostics = {}
    for key in ("raw_csv", "stderr"):
        if key in telemetry:
            descriptor = telemetry[key]
            # Collector's short cells may produce zero-byte CSV/stderr. Verify
            # these bytes without relaxing the positive gate evidence helper.
            if descriptor.get("bytes") == 0:
                require(set(descriptor) == {"path", "bytes", "sha256"}, "diagnostic descriptor fields")
                rel = Path(descriptor["path"])
                path = (root / rel).resolve()
                require(not rel.is_absolute() and ".." not in rel.parts
                        and path.is_relative_to(root.resolve()) and path.is_file()
                        and path.stat().st_size == 0 and battery.digest(path) == descriptor["sha256"],
                        "empty diagnostic hash/path mismatch")
            else:
                battery.evidence(root, descriptor)
            diagnostics[key] = descriptor
    for snapshot in capture.get("compute_apps", {}).values():
        battery.evidence(root, snapshot["raw_log"])
    result = joined[0]
    result.update(schema_version=1, kind="storage-cell-join", capture=end["capture"],
                  raw_log=capture["raw_log"], rig=lock["rig"], lock=lock["lock"],
                  gpu_telemetry_status=telemetry["status"], diagnostics=diagnostics,
                  gpu_telemetry=telemetry, started_monotonic_ns=first, ended_monotonic_ns=last,
                  duration_ns=last-first,
                  storage_label="filesystem development characterization, not spill speed",
                  status="capture-matched-not-qualified")
    return result
