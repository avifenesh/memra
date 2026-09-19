#!/usr/bin/env python3
"""Four-tier qualification scaffold: plan and strict byte-receipt comparison, NOT GPU runner.

A validation pass proves supplied byte evidence agrees; it does not prove every required
model/cell ran, qualify a numeric program, or replace step-pro and the standard battery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

LOCKS = {"rtx5090": "/tmp/memra-5090.lock", "pro-pair": "/tmp/memra-gpu.lock", "pro-four": "/tmp/memra-gpu.lock", "cpu": None}
ROUTES = {"local", "pcie-p2p", "host-bounce", "host", "nvme"}
IDENTITY = ("runtime_commit", "binary_sha256", "artifact_sha256", "plan_sha256", "layout_sha256", "prompt_sha256", "numeric_class", "context_tokens", "requests", "rig", "kind")
CASES = {
    "D1": ["peer-bytes", "grant-failure", "link-downgrade", "timeout", "late-completion", "destination-reuse", "source-free", "graph-address-stability"],
    "D2": ["step37-pp-ladder", "qwen-peer-blocks", "boundary", "churn", "spec-rollback", "cancel"],
    "D3": ["four-tier-pressure", "tenant-purge", "corrupt-active", "pool-smaller-than-object", "all-directed-routes", "shared-fabric-30min"],
    "D4": ["placement-predicted-vs-peak", "owner-replica-accounting", "capacity-refusal"],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evidence(root, record):
    require(set(record) == {"path", "sha256", "bytes"}, "invalid evidence descriptor")
    require(isinstance(record["path"], str), "invalid evidence path")
    rel = Path(record["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, "evidence path escapes bundle")
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "missing/escaping evidence")
    require(type(record["bytes"]) is int and record["bytes"] > 0, "empty evidence")
    require(path.stat().st_size == record["bytes"], "evidence size mismatch")
    require(isinstance(record["sha256"], str) and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "invalid evidence hash")
    require(digest(path) == record["sha256"], "evidence hash mismatch")
    return path


def validate_row(row, root):
    fields = {"schema_version", "cell", "pair_id", "run_id", "arm", "kind", "rig", "lock", "route", "direct_path_proven", "migrated_bytes", "state", "logits", "tokens", "raw_log", "telemetry", "telemetry_interval_ms", "exit_code", "status", *IDENTITY}
    require(set(row) == fields, "missing/unknown receipt fields")
    require(type(row["schema_version"]) is int and row["schema_version"] == 1, "unknown receipt schema")
    require(row["cell"] in sum(CASES.values(), []), "unknown cell")
    require(row["arm"] in {"off", "on"}, "arm must explicitly be off/on")
    require(row["kind"] in {"cpu-fixture", "gpu"}, "unknown evidence class")
    require(row["rig"] in LOCKS and row["lock"] == LOCKS[row["rig"]], "noncanonical rig lock")
    require((row["kind"] == "cpu-fixture") == (row["rig"] == "cpu"), "fixture cannot claim hardware")
    require(row["route"] in ROUTES, "unknown route")
    require(type(row["direct_path_proven"]) is bool, "invalid direct-path evidence")
    if row["route"] == "pcie-p2p":
        require(row["kind"] == "gpu" and row["rig"] in {"pro-pair", "pro-four"} and row["direct_path_proven"], "P2P needs hardware direct-path evidence")
    else:
        require(not row["direct_path_proven"], "host/local route mislabeled direct P2P")
    require(row["status"] == "pass" and type(row["exit_code"]) is int and row["exit_code"] == 0, "failed/refused/skipped run is not a pass")
    for key in ("run_id", "pair_id", "numeric_class"):
        require(isinstance(row[key], str) and bool(row[key]), "empty identity")
    require(isinstance(row["runtime_commit"], str) and re.fullmatch("[0-9a-f]{40}", row["runtime_commit"]), "invalid runtime revision")
    for key in ("binary_sha256", "artifact_sha256", "plan_sha256", "layout_sha256", "prompt_sha256"):
        require(isinstance(row[key], str) and re.fullmatch("[0-9a-f]{64}", row[key]), "invalid identity hash")
    for key in ("context_tokens", "requests"):
        require(type(row[key]) is int and row[key] > 0, "empty workload")
    require(type(row["migrated_bytes"]) is int and row["migrated_bytes"] >= 0, "invalid movement count")
    require(row["migrated_bytes"] > 0 if row["arm"] == "on" else row["migrated_bytes"] == 0, "forced arm did not engage")
    for key in ("state", "logits", "tokens", "raw_log"):
        evidence(root, row[key])
    require(row["logits"]["bytes"] % 4 == 0 and row["tokens"]["bytes"] % 4 == 0, "logits/tokens must be f32/u32 LE bytes")
    if row["kind"] == "gpu":
        require(row["telemetry_interval_ms"] == 250, "GPU telemetry must be 250 ms")
        telemetry = evidence(root, row["telemetry"])
        samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
        require(len(samples) >= 2, "empty/vacuous telemetry")
        # Full telemetry fields and time-window coverage get frozen with the GPU runner.
        require(all(isinstance(s, dict) and "monotonic_ns" in s for s in samples), "invalid telemetry samples")
    else:
        require(row["telemetry"] is None and row["telemetry_interval_ms"] is None, "CPU fixture cannot invent GPU telemetry")


def validate_rows(rows, root):
    require(bool(rows), "empty receipt bundle")
    seen, pairs = set(), {}
    for row in rows:
        validate_row(row, root)
        require(row["run_id"] not in seen, "duplicate run id")
        seen.add(row["run_id"])
        key = (row["cell"], row["pair_id"])
        pair = pairs.setdefault(key, {})
        require(row["arm"] not in pair, "duplicate arm")
        pair[row["arm"]] = row
    for pair in pairs.values():
        require(set(pair) == {"off", "on"}, "missing forced arm")
        a, b = pair["off"], pair["on"]
        require(all(a[k] == b[k] for k in IDENTITY), "not same-program controls")
        for key in ("state", "logits", "tokens"):
            require((a[key]["sha256"], a[key]["bytes"]) == (b[key]["sha256"], b[key]["bytes"]), f"{key} identity mismatch")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate", type=Path, metavar="RUNS_JSONL")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"schema_version": 1, "status": "pending-gpu-adapters", "cases": CASES, "standard_gates": ["kernel-check", "run-gen argmax", "run-spec K=1..8 / manifest refusals", "step-pro"], "locks": LOCKS, "performance": {"AB_pairs": 5, "BA_pairs": 5, "telemetry_interval_ms": 250}, "warning": "Scaffold only. No GPU cell executed; pair is four tiers, not four-card qualification."}, indent=2))
        return
    rows = [json.loads(line) for line in args.validate.read_text().splitlines() if line.strip()]
    pairs = validate_rows(rows, args.validate.parent)
    print(f"BYTE-RECEIPTS MATCH: {len(rows)} records / {pairs} forced pairs; not full battery qualification")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
