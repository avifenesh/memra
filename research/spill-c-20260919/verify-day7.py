#!/usr/bin/env python3
"""Replay day-seven receipt identity and ON/OFF correctness (not a perf gate)."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "rented-5090-20260919/day7"
CASES = ("baseline-gen", "baseline-spec3", "experts-gen", "experts-spec", "device-publish2", "final-gen-off", "final-gen-on", "final-spec-off", "final-spec-on", "device-integrated")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(root, value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            rel = Path(value["path"])
            require(not rel.is_absolute() and ".." not in rel.parts, "escaping receipt")
            raw = (root / rel).read_bytes()
            require(len(raw) == value["bytes"], "receipt length mismatch")
            require(hashlib.sha256(raw).hexdigest() == value["sha256"], "receipt hash mismatch")
        for nested in value.values():
            hashes(root, nested)
    elif isinstance(value, list):
        for nested in value:
            hashes(root, nested)


def tokens(text):
    matches = re.findall(r"^\s*tokens: (\[[0-9, ]+\])$", text, re.M)
    require(len(matches) == 1, "missing/ambiguous token tape")
    ids = json.loads(matches[0])
    require(len(ids) == 32, "short token tape")
    return ids


def main():
    logs = {}
    for case in CASES:
        root = RAW / case
        capture = json.loads((root / "command.capture.json").read_text())
        hashes(root, capture)
        require(capture["exit_code"] == 0 and not capture["timed_out"], case + " failed")
        require(capture["qualification"] is False, "collector is not qualification")
        require(capture["gpu_telemetry"]["interval_ms"] == 250, "telemetry cadence")
        require(capture["gpu_power_limits"] == [{"device": "0", "power.limit": "400.00 W", "power.max_limit": "600.00 W"}], "power regime changed")
        require(json.loads((root / "lock.json").read_text()) == {"rig": "rtx5090", "lock": "/tmp/memra-5090.lock", "acquired": True}, "missing canonical collector lock receipt")
        logs[case] = (root / "command.log").read_text()
    for baseline, banked in [("baseline-gen", "experts-gen"), ("baseline-spec3", "experts-spec"), ("final-gen-off", "final-gen-on"), ("final-spec-off", "final-spec-on")]:
        require(tokens(logs[baseline]) == tokens(logs[banked]), "ON/OFF token tape differs")
        require(re.search(r"physical_reads=[1-9][0-9]* owner_close=Ok\(\(\)\)", logs[banked]), "bank did not engage/drain")
    for case in ["baseline-gen", "experts-gen", "final-gen-off", "final-gen-on"]:
        require("prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH" in logs[case], "argmax verdict changed")
    acceptance = []
    for case in ["baseline-spec3", "experts-spec", "final-spec-off", "final-spec-on"]:
        require("=== SELF-CONSISTENCY PASS ===" in logs[case], "spec failed")
        require(re.findall(r"\[generate_spec K=(\d)\]", logs[case]) == list("12345678"), "K ladder incomplete")
        rows = re.findall(r"acceptance: ([^\n]+)", logs[case])
        require(len(rows) == 8 and all("self-consistency: PASS" in row for row in rows), "spec row failed")
        acceptance.append(rows)
    require(all(rows == acceptance[0] for rows in acceptance), "acceptance changed")
    require("values=6144 tier_calls=16 forced_read_chunks=144 device_uploads=16 exclusive_handback=true budget_drained=true" in logs["device-integrated"], "native row publication did not engage every case")
    require("BIT-IDENTICAL PLE outputs + convolution state" in logs["device-integrated"], "row outputs differ")
    source = {name: (RAW / name).read_text().strip() for name in ["baseline-source.txt", "expert-source.txt", "spec-source.txt", "device-source.txt", "handoff-source.txt", "device-integrated-source.txt"]}
    require(source["handoff-source.txt"].startswith("18b2f092") and source["device-integrated-source.txt"].startswith("efd3fbeb"), "final source identity changed")
    require(all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in source.values()), "invalid source identity")
    print(json.dumps({"receipt_replay": "PASS", "rig": "rented RTX 5090 development", "power_watts": [400, 600], "argmax_on_off": "MATCH", "gen_and_spec_tapes": "32 tokens each, byte-identical within each ON/OFF pair", "spec": "K=1..8 PASS, acceptance unchanged", "rows": "16 native device uploads; bit-identical 6144 output/state values", "source_commits": source, "production_qualified": False}, indent=2))


if __name__ == "__main__":
    main()
