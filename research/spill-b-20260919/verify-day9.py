#!/usr/bin/env python3
"""Replay source-bound day-9 receipts. This does not execute CUDA or qualify serving."""
import csv
import gzip
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


def verify_collector(folder, power=400):
    require((folder / "collector.exit").read_text().strip() == "0", "collector failed")
    lock = json.loads((folder / "collector/lock.json").read_text())
    require(lock == {"rig": "rtx5090", "lock": "/tmp/memra-5090.lock", "acquired": True},
            "canonical collector lock missing")
    cap = json.loads((folder / "collector/command.capture.json").read_text())
    require(cap["exit_code"] == 0 and not cap["timed_out"], "native subprocess failed")
    cells = [json.loads(line) for line in (folder / "collector/CELL.jsonl").read_text().splitlines()]
    require([row["event"] for row in cells] == ["start", "end"], "incomplete CELL journal")
    require(cells[-1]["exit_code"] == 0 and cells[-1]["qualification"] is False,
            "unsuccessful or promoted CELL")
    require(cells[-1]["capture"]["sha256"] == sha(folder / "collector/command.capture.json"),
            "CELL capture hash mismatch")
    require(cap["gpu_power_limits"] == [{"device": "0", "power.limit": f"{power:.2f} W",
                                         "power.max_limit": "600.00 W"}], "power envelope mismatch")
    with (folder / "collector/command.gpu.csv").open() as data:
        telemetry = list(csv.DictReader(data))
    require(telemetry, "empty telemetry")
    for sample in telemetry:
        sample = {key.strip(): value.strip() for key, value in sample.items()}
        require(sample["power.limit [W]"] == f"{power:.2f} W"
                and sample["power.max_limit [W]"] == "600.00 W", "power cap changed inside cell")
    require(cap["qualification"] is False, "collector must not invent production qualification")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, "wrong telemetry interval")
    require(sha(folder / "collector/command.log") == cap["raw_log"]["sha256"], "raw log mismatch")
    require(sha(folder / "collector/command.gpu.csv") == cap["gpu_telemetry"]["raw_csv"]["sha256"],
            "telemetry mismatch")
    for phase in ("before", "after"):
        snapshot = folder / f"collector/command.{phase}.log"
        require(len(snapshot.read_text().splitlines()) == 1, f"co-tenant in {phase} snapshot")
        require(sha(snapshot) == cap["compute_apps"][phase]["raw_log"]["sha256"],
                "compute-app snapshot hash mismatch")
    require(not any(message in (folder / "collector/command.log").read_text()
                    for message in ("VMM cleanup", "VMM VA cleanup", "CUDA_ERROR_")),
            "CUDA cleanup or runtime failure in raw log")


def capture(folder, baseline, source, power=400):
    require((folder / "source.commit").read_text().strip() == source, "wrong native source")
    verify_collector(folder, power)
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


def verify_plane_census(receipt, metrics):
    rows = (receipt / "vmm-planes.tsv").read_text().splitlines()[1:]
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

def verify_vmm(folder, context, source):
    metrics = capture(folder, RAW / f"day6-baseline-{context}/receipt", source,
                      power=600 if context == 32768 else 400)
    verify_plane_census(folder / "receipt", metrics)
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


def bounded_no_leak(m):
    gain = m["free_after_demote_bytes"] - m["free_before_bytes"]
    return (m["vmm_released_chunk_bytes"] > 0 and m["vmm_granularity_bytes"] > 0 and gain > 0
            and gain >= max(0, m["vmm_released_chunk_bytes"] - m["vmm_granularity_bytes"])
            and m["free_after_restore_bytes"] == m["free_before_bytes"]
            and m["free_after_demote_bytes"] - m["free_after_restore_bytes"] == gain)


def probe_class(m, probe):
    residual = m["vmm_released_chunk_bytes"] - m["reclaimed_bytes"]
    after = m["free_after_demote_bytes"]
    va = probe["free_after_spare_va_free_bytes"]
    sync = probe["free_after_context_sync_bytes"]
    if residual == 0:
        return "none"
    if va - after == residual and sync == va:
        return "spare-VA-release-sensitive-driver-accounting"
    if sync - va == residual and va == after:
        return "deferred-driver-release-completed-by-context-sync"
    return "unclassified"


def verify_residual(manifest):
    original = RAW / "day9-vmm-32768-retry"
    m = capture(original, RAW / "day6-baseline-32768/receipt", manifest["vmm_source"], power=600)
    verify_plane_census(original / "receipt", m)
    require(bounded_no_leak(m), "original 32k does not meet bounded no-leak criterion")
    require(m["vmm_released_chunk_bytes"] - m["reclaimed_bytes"] == m["vmm_granularity_bytes"],
            "original 32k residual changed")
    require(fields(original / "receipt/active-reclaim.txt")["reclaim_observed"] == "false",
            "original strict equality result rewritten")
    folder = RAW / "day9-residual"
    verify_collector(folder, power=600)
    require((folder / "source.commit").read_text().strip() == manifest["residual_source"],
            "residual diagnostic source mismatch")
    require((folder / "residual-cell.sh").read_bytes() == (LANE / "residual-cell.sh").read_bytes(),
            "diagnostic script changed")
    archived = folder / "receipt-16384/final-logits.f32le"
    archive = json.loads(archived.with_suffix(".f32le.manifest.json").read_text())
    decoded = gzip.decompress(archived.with_suffix(".f32le.gz").read_bytes())
    require(len(decoded) == archive["decoded_bytes"]
            and hashlib.sha256(decoded).hexdigest() == archive["decoded_sha256"],
            "lossless 16k logits archive differs")
    native = RAW / "day9-residual-build"
    require((native / "source.commit").read_text().strip() == manifest["residual_source"],
            "final native build source mismatch")
    for check in ("build", "clippy"):
        require((native / f"{check}.exit").read_text().strip() == "0", f"final native {check} failed")
    results = {}
    for context in (32768, 16384):
        receipt = folder / f"receipt-{context}"
        identity = fields(receipt / "identity.txt")
        require(identity["binary_sha256"] == (folder / "binary.sha256").read_text().split()[0],
                "diagnostic binary mismatch")
        require(identity["context"] == str(context), "diagnostic context mismatch")
        require((receipt / "prefix-state.tsv").read_bytes() == (receipt / "restored-prefix-state.tsv").read_bytes(),
                "diagnostic restored prefix differs")
        if context == 32768:
            baseline = RAW / "day6-baseline-32768/receipt"
            for name in SURFACES:
                require((receipt / name).read_bytes() == (baseline / name).read_bytes(),
                        f"diagnostic 32k baseline mismatch: {name}")
        current = integer_fields(receipt / "active-reclaim.txt")
        flags = fields(receipt / "active-reclaim.txt")
        probe = integer_fields(receipt / "residual-diagnostic.txt")
        probe_flags = fields(receipt / "residual-diagnostic.txt")
        require(probe_flags["spare_va_free_success"] == probe_flags["context_sync_success"] == "true",
                "residual diagnostic operation failed")
        verify_plane_census(receipt, current)
        require(current["demote_count"] == current["reload_count"] == 32, "diagnostic incomplete engagement")
        require(current["device_charged_after_demote"] == 0 and
                current["pinned_charged_after_demote"] == current["logical_d2h_bytes"] > 0,
                "diagnostic accounting mismatch")
        residual = current["vmm_released_chunk_bytes"] - current["reclaimed_bytes"]
        require(current["residual_bytes"] == probe["residual_bytes"] == residual, "residual arithmetic mismatch")
        klass = probe_class(current, probe)
        require(flags["residual_class"] == probe_flags["residual_class"] == klass, "unsupported residual class")
        bounded = bounded_no_leak(current)
        require(flags["reclaim_observed"] == str(bounded).lower(), "bounded criterion flag mismatch")
        exact = current["reclaimed_bytes"] == current["reacquired_bytes"] == current["vmm_released_chunk_bytes"] > 0
        require(flags["reclaim_exact_equal"] == str(exact).lower(), "raw equality flag mismatch")
        require(flags["g1_reclaim_qualified"] == str(bounded and klass != "unclassified").lower(),
                "unclassified residual promoted")
        print(f"Residual diagnostic {context}: {residual} bytes, class={klass}")
        results[context] = (residual, klass, bounded)
    residual, klass, bounded = results[32768]
    if residual == m["vmm_granularity_bytes"] and klass != "unclassified" and bounded:
        return f"ACTIVE-32K G1 PASS (one-granule driver residual, classified: {klass})"
    return "ACTIVE-32K physical reclaim/restore bit-identical, one-granule residual unclassified — not G1 PASS"


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
    require(fields(diag / "receipt/reclaim-diagnosis.txt")["trim_api_success"] == "true",
            "checked trim did not succeed")
    require(info["free_before_trim_bytes"] == info["free_after_trim_bytes"], "trim reclaimed")
    require(info["pool_before_reserved"] == info["pool_demoted_reserved"] == info["pool_trimmed_reserved"],
            "driver pool reservation changed in control")
    require(info["pool_before_used"] - info["pool_demoted_used"]
            == metrics["source_allocation_bytes"] - 32, "pool used/placeholder mismatch")
    require(metrics["reclaimed_bytes"] == metrics["reacquired_bytes"] == 0, "control reclaimed")
    print("RECLAIM-DIAG: freed but not observable")
    print(CONTROL)
    interrupted = RAW / "day9-vmm-32768-interrupted"
    failure = json.loads((interrupted / "failure.json").read_text())
    require(failure["verdict"] == "died, cause: host stop" and failure["exit_code"] is None,
            "interrupted capture reclassified")
    require(not (interrupted / "collector.exit").exists()
            and not (interrupted / "collector/command.capture.json").exists(), "interrupted capture gained completion")
    cells = [json.loads(line) for line in (interrupted / "collector/CELL.jsonl").read_text().splitlines()]
    require([row["event"] for row in cells] == ["start"], "unexpected interrupted CELL completion")
    print("32k initial capture: died, cause: host stop (lead-confirmed; incomplete raw capture)")
    for context in manifest["vmm_contexts"]:
        suffix = "-retry" if context == 32768 else ""
        print(verify_vmm(RAW / f"day9-vmm-{context}{suffix}", context, manifest["vmm_source"]))
    if "residual_source" in manifest:
        print(verify_residual(manifest))
    native = RAW / "day9-vmm-build"
    require((native / "source.commit").read_text().strip() == manifest["vmm_source"], "native check source mismatch")
    for check in ("build", "clippy", "tests-retry"):
        require((native / f"{check}.exit").read_text().strip() == "0", f"native {check} did not pass")
    require((native / "tests.exit").read_text().strip() == "101", "initial native test failure was erased")
    require("Err` value: Busy" in (native / "tests.log").read_text(), "initial native failure quote missing")
    for directory in ("day9-checks", "day9-vmm-checks", "day9-residual-checks"):
        checks = json.loads((LANE / directory / "commands.json").read_text())
        require(all(check["exit"] == 0 for check in checks["checks"]), f"failed Mac check: {directory}")
    print(f"PASS: {len(manifest['files'])} hashed files; native execution and CPU checks remain distinct")


if __name__ == "__main__":
    main()
