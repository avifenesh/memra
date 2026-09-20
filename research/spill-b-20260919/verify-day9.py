#!/usr/bin/env python3
"""Replay source-bound day-9 receipts. This does not execute CUDA or qualify serving."""
import hashlib
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent
RAW = LANE / "rented-5090-20260919"
SURFACES = ("prefix-state.tsv", "final-state.tsv", "tokens.u32le", "logits.tsv",
            "final-logits.f32le", "prompt.u32le", "plan.debug")
CONTROL = "ACTIVE-8K copy/restore bit-identical, no reclaim — not G1 PASS"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def integer_fields(path):
    return {k: int(v) for k, v in fields(path).items() if v.lstrip("-").isdigit()}


def classify_vmm(metrics, identical, context):
    require(identical, "continuation differs from frozen baseline")
    released = metrics["vmm_released_chunk_bytes"]
    require(released > 0, "no VMM chunk was released")
    require(metrics["demote_count"] == metrics["reload_count"] == 32, "incomplete engagement")
    require(metrics["device_charged_after_demote"] == 0, "device still charged")
    require(metrics["pinned_charged_after_demote"] == metrics["logical_d2h_bytes"] > 0,
            "host accounting mismatch")
    require(metrics["free_after_demote_bytes"] - metrics["free_before_bytes"] == released,
            "driver reclaim differs from released chunk bytes")
    require(metrics["free_after_demote_bytes"] - metrics["free_after_restore_bytes"] == released,
            "driver reacquisition differs from released chunk bytes")
    require(metrics["reclaimed_bytes"] == metrics["reacquired_bytes"] == released,
            "reported deltas disagree")
    return f"ACTIVE-{context // 1024}K G1 PASS"


def capture(folder, baseline, source):
    require((folder / "source.commit").read_text().strip() == source, "wrong native source")
    require((folder / "collector.exit").read_text().strip() == "0", "collector failed")
    cap = json.loads((folder / "collector/command.capture.json").read_text())
    require(cap["exit_code"] == 0 and not cap["timed_out"], "native subprocess failed")
    require(cap["qualification"] is False, "collector must not invent production qualification")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, "wrong telemetry interval")
    require(sha(folder / "collector/command.log") == cap["raw_log"]["sha256"], "raw log mismatch")
    require(sha(folder / "collector/command.gpu.csv") == cap["gpu_telemetry"]["raw_csv"]["sha256"],
            "telemetry mismatch")
    identity = fields(folder / "receipt/identity.txt")
    require(identity["binary_sha256"] == (folder / "binary.sha256").read_text().split()[0],
            "binary identity mismatch")
    frozen = fields(baseline / "identity.txt")
    for key in ("artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "context"):
        require(identity[key] == frozen[key], f"baseline identity mismatch: {key}")
    for name in SURFACES:
        require((folder / "receipt" / name).read_bytes() == (baseline / name).read_bytes(),
                f"baseline surface mismatch: {name}")
    require((folder / "receipt/restored-prefix-state.tsv").read_bytes()
            == (baseline / "prefix-state.tsv").read_bytes(), "restored prefix mismatch")
    return integer_fields(folder / "receipt/active-reclaim.txt")


def verify_vmm(folder, context, source):
    metrics = capture(folder, RAW / f"day6-baseline-{context}/receipt", source)
    rows = (folder / "receipt/vmm-planes.tsv").read_text().splitlines()[1:]
    require(len(rows) == 32, "missing per-plane VMM census")
    released = 0
    physical = 0
    addresses = set()
    for row in rows:
        _, role, address, valid, capacity, granularity = row.split("\t")
        address, valid, capacity, granularity = map(int, (address, valid, capacity, granularity))
        require(role in ("Key", "Value"), "unknown VMM plane")
        require(granularity > 0 and granularity & (granularity - 1) == 0, "invalid granularity")
        require(granularity == metrics["vmm_granularity_bytes"], "granularity mismatch")
        require(address % granularity == 0 and address not in addresses, "invalid/aliased VA")
        addresses.add(address)
        require(capacity >= valid and capacity % granularity == 0, "invalid physical capacity")
        released += valid // granularity * granularity
        physical += capacity
    require(metrics["vmm_released_chunk_bytes"] == released, "chunk math mismatch")
    require(metrics["source_physical_bytes"] == physical, "physical census mismatch")
    require(metrics["vmm_retained_edge_and_capacity_bytes"] == physical - released,
            "retained edge/capacity mismatch")
    flags = fields(folder / "receipt/active-reclaim.txt")
    require(flags["vmm_fixed_va_restored"] == flags["reclaim_observed"] == "true",
            "fixed VA/reclaim proof absent")
    verdict = classify_vmm(metrics, True, context)
    # Each red arm must reject physical/accounting/continuation substitutions.
    for key in ("vmm_released_chunk_bytes", "free_after_demote_bytes",
                "free_after_restore_bytes", "demote_count", "device_charged_after_demote"):
        bad = metrics.copy()
        bad[key] += 1
        try:
            classify_vmm(bad, True, context)
        except ValueError:
            continue
        raise ValueError(f"negative verdict arm unexpectedly passed: {key}")
    try:
        classify_vmm(metrics, False, context)
    except ValueError:
        pass
    else:
        raise ValueError("nonidentical continuation unexpectedly passed")
    return verdict


def main():
    manifest = json.loads((LANE / "day9-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        require(sha(LANE / name) == expected, f"receipt integrity: {name}")
    diag = RAW / "day9-diag-8192"
    metrics = capture(diag, RAW / "day6-baseline-8192/receipt", manifest["diagnostic_source"])
    info = integer_fields(diag / "receipt/reclaim-diagnosis.txt")
    require((diag / "receipt/reclaim-diagnosis.txt").read_text().splitlines()[0]
            == "RECLAIM-DIAG: freed but not observable", "wrong diagnosis")
    require(info["source_slice_drops"] == info["source_owners_before_release"] == 32,
            "source destruction not observed")
    require(info["source_owners_after_release"] == info["device_registry_after_demote"] == 0,
            "source remains owned")
    require(info["free_before_trim_bytes"] == info["free_after_trim_bytes"], "trim reclaimed")
    require(info["pool_before_used"] - info["pool_demoted_used"]
            == metrics["source_allocation_bytes"] - 32, "pool used/placeholder mismatch")
    require(metrics["reclaimed_bytes"] == metrics["reacquired_bytes"] == 0, "control reclaimed")
    print("RECLAIM-DIAG: freed but not observable")
    print(CONTROL)
    for context in manifest["vmm_contexts"]:
        print(verify_vmm(RAW / f"day9-vmm-{context}", context, manifest["vmm_source"]))
    for directory in ("day9-checks", "day9-vmm-checks"):
        checks = json.loads((LANE / directory / "commands.json").read_text())
        require(all(check["exit"] == 0 for check in checks["checks"]), f"failed Mac check: {directory}")
    print(f"PASS: {len(manifest['files'])} hashed files; native execution and CPU checks remain distinct")


if __name__ == "__main__":
    main()
