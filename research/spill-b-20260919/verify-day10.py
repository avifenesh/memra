#!/usr/bin/env python3
"""Offline byte/collector replay, NOT CUDA execution or serving qualification."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "pro-single-day10"
SURFACES = ("prefix-state.tsv", "final-state.tsv", "tokens.u32le", "logits.tsv",
            "final-logits.f32le", "prompt.u32le", "plan.debug")
EXPECTED = ("baseline-8192", "baseline-32768", "vmm-8192", "vmm-32768",
            "pooled-8192", "diagnostic-32768", "injected-8192")
spec = importlib.util.spec_from_file_location("day9", HERE / "verify-day9.py")
day9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day9)
require = day9.require
fields = day9.fields


def data(path):
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def collector(folder):
    require((folder / "collector.exit").read_text().strip() == "0", "collector failed")
    attempt = int((folder / "successful-attempt").read_text())
    c = folder / f"collector-{attempt}"
    require(json.loads((c / "lock.json").read_text()) ==
            {"rig": "pro-single", "lock": "/tmp/memra-gpu.lock", "acquired": True}, "wrong rig/lock")
    cap = json.loads((c / "command.capture.json").read_text())
    require(cap["exit_code"] == 0 and not cap["timed_out"] and cap["qualification"] is False,
            "unsuccessful/promoted capture")
    require(cap["gpu_power_limits"] == [{"device": "0", "power.limit": "600.00 W",
                                         "power.max_limit": "600.00 W"}], "power envelope")
    cells = [json.loads(x) for x in (c / "CELL.jsonl").read_text().splitlines()]
    require([x["event"] for x in cells] == ["start", "end"], "incomplete journal")
    require(cells[-1]["exit_code"] == 0 and cells[-1]["qualification"] is False, "bad CELL end")
    require(cells[-1]["capture"]["sha256"] == sha(c / "command.capture.json"), "capture hash")
    require(cap["raw_log"]["sha256"] == sha(c / "command.log"), "raw log hash")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, "telemetry interval")
    require(cap["gpu_telemetry"]["raw_csv"]["sha256"] == sha(c / "command.gpu.csv"), "telemetry hash")
    samples = list(csv.DictReader((c / "command.gpu.csv").open()))
    require(samples, "empty telemetry")
    for sample in samples:
        sample = {k.strip(): v.strip() for k, v in sample.items()}
        require(sample["power.limit [W]"] == sample["power.max_limit [W]"] == "600.00 W", "power changed")
    for phase in ("before", "after"):
        path = c / f"command.{phase}.log"
        require(len(path.read_text().splitlines()) == 1, "co-tenant snapshot")
        require(sha(path) == cap["compute_apps"][phase]["raw_log"]["sha256"], "snapshot hash")
    require(not any(x in (c / "command.log").read_text() for x in
                    ("VMM cleanup", "VMM VA cleanup", "CUDA_ERROR_")), "native cleanup/runtime error")
    identity = fields(folder / "receipt/identity.txt")
    require(identity["binary_sha256"] == (folder / "binary.sha256").read_text().split()[0], "binary hash")
    source = (folder / "source.commit").read_text().strip()
    builds = [p for p in RAW.glob("*build*") if p.is_dir() and (p / "source.txt").exists()]
    require(any((p / "source.txt").read_text().strip() == source and
                (p / "binary.sha256").read_text().split()[0] == identity["binary_sha256"] and
                (p / "exit").read_text().strip() == "0" for p in builds), "no matching successful native build")
    command = cap["command"]
    require(command[command.index("--context") + 1] == identity["context"], "command context mismatch")
    require(command[command.index("--tiers") + 1] == "host" and "--same-program" in command,
            "wrong program/storage route")
    r = folder / "receipt"
    require(identity["plan_debug_sha256"] == decoded_sha(r / "plan.debug"), "plan hash")
    require(identity["prompt_sha256"] == decoded_sha(r / "prompt.u32le"), "prompt hash")
    require(len(data(r / "tokens.u32le")) == 128 * 4, "continuation token count")
    require(len(data(r / "prompt.u32le")) == (int(identity["context"]) - 128) * 4, "prompt count")
    return identity


def verdict(m, flags, context):
    require(m["demote_count"] == m["reload_count"] == 32, "engagement")
    require(m["device_charged_after_demote"] == 0 and
            m["pinned_charged_after_demote"] == m["logical_d2h_bytes"] > 0, "accounting")
    gain = m["free_after_demote_bytes"] - m["free_before_bytes"]
    reacquired = m["free_after_demote_bytes"] - m["free_after_restore_bytes"]
    require(m["reclaimed_bytes"] == gain and m["reacquired_bytes"] == reacquired, "reported deltas")
    tag = f"ACTIVE-{context // 1024}K"
    if m["vmm_granularity_bytes"] == 0:
        require(flags["g1_reclaim_qualified"] == flags["vmm_fixed_va_restored"] ==
                flags["residual_class"] == "not-applicable-pooled", "pooled falsely qualified")
        return f"{tag} pooled control: not-applicable-pooled"
    residual = m["vmm_released_chunk_bytes"] - gain
    require(m["residual_bytes"] == residual, "residual math")
    require(flags["vmm_fixed_va_restored"] == "true", "fixed VA absent")
    bounded = day9.bounded_no_leak(m)
    require(flags["reclaim_observed"] == str(bounded).lower(), "bounded flag mismatch")
    exact = m["vmm_released_chunk_bytes"] > 0 and gain == m["vmm_released_chunk_bytes"] and gain == reacquired
    require(flags["reclaim_exact_equal"] == str(exact).lower(), "exact flag mismatch")
    qualified = bounded and residual == 0
    # No mapped-range diagnostic exists on the earlier card: never promote a one-card class.
    require(flags["g1_reclaim_qualified"] == str(qualified).lower(), "unsupported G1 promotion")
    if qualified:
        require(flags["residual_class"] == "none", "zero residual class")
        return f"{tag} G1 PASS"
    return (f"{tag} physical reclaim/restore bit-identical, residual {residual} B, "
            f"class {flags['residual_class']} — not G1 PASS")


def check_probe(receipt, m, flags):
    probe = fields(receipt / "residual-diagnostic.txt")
    rows = list(csv.DictReader((receipt / "mapped-va-probe.tsv").open(), delimiter="\t"))
    require(len(rows) == 32 and {int(r["plane"]) for r in rows} == set(range(32)), "probe census")
    va = sum(int(r["after_va_free"]) - int(r["after_unmap"]) for r in rows)
    unmap = sum(int(r["after_unmap"]) - int(r["before"]) for r in rows)
    equal = all(r["before"] == r["after_remap"] for r in rows)
    equal &= probe["free_after_mapped_va_probe_bytes"] == probe["free_after_context_sync_bytes"]
    require(int(probe["mapped_va_release_delta_bytes"]) == va and
            int(probe["mapped_unmap_delta_bytes"]) == unmap and
            probe["mapped_va_roundtrip_equal"] == str(equal).lower(), "probe accounting")
    residual = m["residual_bytes"]
    klass = "none" if residual == 0 else "va-reservation-page-table" if va == residual and unmap == 0 and equal else day9.probe_class(m, {k: int(v) for k, v in probe.items() if v.lstrip('-').isdigit()})
    require(flags["residual_class"] == probe["residual_class"] == klass, "unproven residual class")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", action="store_true", help="seal newly copied raw bytes, never replace frozen baseline hashes")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    frozen_path = HERE / "BOX3-BASELINES.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {
        "rig": "one RTX PRO 6000 Blackwell 96 GB", "power_w": [600, 600],
        "warning": "Separate frozen bundles; never mix with RTX 5090 baselines.", "baselines": {}}
    result = {}
    for context in (8192, 32768):
        folder = RAW / f"baseline-{context}"
        if not (folder / "collector.exit").exists():
            continue
        identity = collector(folder)
        row = {"identity": identity, "source": (folder / "source.commit").read_text().strip(),
               "surfaces_sha256": {x: decoded_sha(folder / "receipt" / x) for x in SURFACES}}
        key = str(context)
        if key in frozen["baselines"]:
            require(frozen["baselines"][key] == row, "frozen baseline changed")
        else:
            require(args.seal, "baseline not frozen; use --seal for first import")
            frozen["baselines"][key] = row
        result[folder.name] = "BASELINE_CAPTURED (not G1)"
    for label in EXPECTED[2:]:
        folder = RAW / label
        if not (folder / "collector.exit").exists():
            continue
        identity = collector(folder)
        context = int(identity["context"])
        frozen_row = frozen["baselines"][str(context)]
        for key in ("artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "context"):
            require(identity[key] == frozen_row["identity"][key], f"wrong baseline identity: {key}")
        receipt = folder / "receipt"
        for name in SURFACES:
            require(decoded_sha(receipt / name) == frozen_row["surfaces_sha256"][name], f"baseline mismatch: {name}")
        require(decoded_sha(receipt / "restored-prefix-state.tsv") == frozen_row["surfaces_sha256"]["prefix-state.tsv"], "restored prefix differs")
        flags = fields(receipt / "active-reclaim.txt")
        m = {k: int(v) for k, v in flags.items() if v.lstrip('-').isdigit()}
        if m["vmm_granularity_bytes"]:
            day9.verify_plane_census(receipt, m)
        if label.startswith("diagnostic-"):
            check_probe(receipt, m, flags)
        result[label] = verdict(m, flags, context)
    if args.require_complete:
        require(set(result) == set(EXPECTED), f"pending cells: {set(EXPECTED) - set(result)}")
    manifest = {str(p.relative_to(RAW)): sha(p) for p in sorted(RAW.rglob("*")) if p.is_file()}
    mp = HERE / "day10-raw-manifest.json"
    if args.seal:
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
        mp.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        require(json.loads(mp.read_text()) == manifest, "raw archive membership/hash mismatch")
    print(json.dumps({"checked": result, "pending": sorted(set(EXPECTED) - set(result)),
                      "cuda_executed_by_replay": False}, indent=2))


if __name__ == "__main__":
    main()
