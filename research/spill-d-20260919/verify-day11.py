#!/usr/bin/env python3
"""Offline replay of the day-11 fault-arm cells: collector receipts, the pure fault verdict over
the recorded checks, and the continuation surfaces against the frozen target-card bundle.
Byte replay only, NOT CUDA execution, NOT qualification."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("cancel-demote", "cancel-restore", "corrupt-host", "missing-host",
        "host-budget-short", "device-short", "require-resident")
# Mirror of `fault_contract::Arm` (kv_tier_gate/fault_contract.rs); the Rust test harness
# replays the same receipts, this file replays them without a Rust toolchain.
REQUIRED = {
    "cancel-demote": ("cancel", "take-after-cancel", "retire", "acknowledge", "pinned-after-cancel",
                      "source-returned", "budget-zero", "restored-identical"),
    "cancel-restore": ("cancel", "ready-view-after-cancel", "with-destination-after-cancel",
                       "take-after-cancel", "retire", "acknowledge", "retake-demoted-copy", "budget-zero"),
    "corrupt-host": ("write-under-live-ticket", "write-sole-owner", "restore-integrity",
                     "device-registry-unchanged", "budget-zero"),
    "missing-host": ("completion-checksum", "remove", "pinned-released", "retake-removed-copy", "budget-zero"),
    "host-budget-short": ("whole-state-admission", "layers-resident", "pinned-charged", "budget-zero",
                          "restored-identical"),
    "device-short": ("restore-admission", "device-registry-unchanged", "host-copy-intact", "budget-zero",
                     "restored-identical"),
    "require-resident": ("budget-zero", "restored-identical"),
}
REFUSAL = {
    "cancel-restore": "REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to "
                      "recover the H2D source after cancellation; no tokens, budget drained",
    "require-resident": "REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a "
                        "suspended cache, decode_step_h unwraps a suspended layer, and tier "
                        "RestoreDecision::RequireState is a load-versus-recompute rule",
}
RESTORES = {arm: arm not in ("cancel-restore", "corrupt-host", "missing-host") for arm in ARMS}
CONTINUES = {arm: RESTORES[arm] and arm not in REFUSAL for arm in ARMS}
# Fault injection points as the arms document them; printed in the table for the issue comment.
INJECTION = {
    "cancel-demote": "TransferEngine::cancel after D2H synchronize, before take_destination",
    "cancel-restore": "TransferEngine::cancel after H2D synchronize, before ready_view",
    "corrupt-host": "CudaPinnedLease::write (Busy under the live D2H ticket; Ok after its drain), then restore",
    "missing-host": "untaken D2H ticket retired and acknowledged; take_destination at restore",
    "host-budget-short": "governor pinned capacity = whole demoted bytes minus 1; admit_whole_state",
    "device-short": "competing tenant reserves the governor device dimension; restore alloc_device",
    "require-resident": "Cache::ensure_usable asked while every full-history plane is demoted",
}
SURFACES = ("prefix-state.tsv", "final-state.tsv", "tokens.u32le", "logits.tsv",
            "final-logits.f32le", "prompt.u32le", "plan.debug")
IDENTITY_SHARED = ("artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "mode",
                   "context", "prompt_tokens", "generate", "requested_tiers")


def require(condition, message):
    if not condition:
        raise SystemExit(f"REFUSED: {message}")


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def data(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + ".gz").read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def verdict(arm, checks):
    """The pure rule of fault_contract::verdict, replayed in Python."""
    names = [c["check"] for c in checks]
    missing = [name for name in REQUIRED[arm] if name not in names]
    failed = []
    for c in checks:
        if c["expected"] != c["observed"] and c["check"] not in failed:
            failed.append(c["check"])
    if missing or failed:
        return False, (f"fault arm {arm} did not prove its contract; "
                       f"missing=[{','.join(missing)}] failed=[{','.join(failed)}]")
    if arm in REFUSAL:
        return False, REFUSAL[arm]
    return True, f"FAULT-ARM PASS {arm}"


def collector(folder, arm, build):
    exit_code = int((folder / "collector.exit").read_text().strip())
    attempt = int((folder / "launched-attempt").read_text())
    c = folder / f"collector-{attempt}"
    require(json.loads((c / "lock.json").read_text()) ==
            {"rig": "pro-single", "lock": "/tmp/memra-gpu.lock", "acquired": True}, f"{arm}: wrong rig/lock")
    cap = json.loads((c / "command.capture.json").read_text())
    require(cap["qualification"] is False and not cap["timed_out"], f"{arm}: promoted or timed-out capture")
    expected_status = "refused" if arm in REFUSAL else "executed-not-qualified"
    expected_exit = 2 if arm in REFUSAL else 0
    require(cap["exit_code"] == expected_exit and cap["status"] == expected_status and exit_code == expected_exit,
            f"{arm}: exit {cap['exit_code']} status {cap['status']} (collector exit {exit_code})")
    require(cap["gpu_power_limits"] == [{"device": "0", "power.limit": "600.00 W",
                                         "power.max_limit": "600.00 W"}], f"{arm}: power envelope")
    cells = [json.loads(x) for x in (c / "CELL.jsonl").read_text().splitlines()]
    require([x["event"] for x in cells] == ["start", "end"], f"{arm}: incomplete journal")
    require(cells[-1]["exit_code"] == expected_exit and cells[-1]["qualification"] is False, f"{arm}: bad CELL end")
    require(cells[-1]["capture"]["sha256"] == sha(c / "command.capture.json"), f"{arm}: capture hash")
    require(cap["raw_log"]["sha256"] == sha(c / "command.log"), f"{arm}: raw log hash")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, f"{arm}: telemetry interval")
    require(cap["gpu_telemetry"]["raw_csv"]["sha256"] == sha(c / "command.gpu.csv"), f"{arm}: telemetry hash")
    samples = list(csv.DictReader((c / "command.gpu.csv").open()))
    require(samples, f"{arm}: empty telemetry")
    for sample in samples:
        sample = {k.strip(): v.strip() for k, v in sample.items()}
        require(sample["power.limit [W]"] == sample["power.max_limit [W]"] == "600.00 W", f"{arm}: power changed")
    for phase in ("before", "after"):
        path = c / f"command.{phase}.log"
        require(len(path.read_text().splitlines()) == 1, f"{arm}: co-tenant snapshot {phase}")
        require(sha(path) == cap["compute_apps"][phase]["raw_log"]["sha256"], f"{arm}: snapshot hash")
    log = (c / "command.log").read_text()
    require(not any(x in log for x in ("CUDA_ERROR_", "panicked at", "kv-tier-gate: fault arm")),
            f"{arm}: runtime error or failed arm in the raw log")
    command = cap["command"]
    for flag, value in (("--case", "active"), ("--kv-allocator", "pooled"), ("--context", "8192"),
                        ("--tiers", "host"), ("--fault", arm)):
        require(command[command.index(flag) + 1] == value, f"{arm}: command {flag} is not {value}")
    require("--same-program" in command and "--reclaim-diagnostic" not in command, f"{arm}: program flags")
    require((folder / "source.commit").read_text().strip() == build["source"], f"{arm}: source commit")
    require((folder / "binary.sha256").read_text().split()[0] == build["binary"], f"{arm}: binary hash")
    validate = json.loads((folder / "validate.json").read_text())
    require(validate["kind"] == "capture-integrity" and validate["cells"] == 1 and
            validate["failed_commands"] == 0 and validate["qualification"] is False and
            validate["refused_commands"] == (1 if arm in REFUSAL else 0), f"{arm}: collector --validate")
    return log.splitlines()[-1], samples


def receipt(folder, arm, baseline, build):
    r = folder / "receipt"
    identity = fields(r / "identity.txt")
    require(identity["binary_sha256"] == build["binary"], f"{arm}: identity binary hash")
    require(identity["context"] == "8192" and identity["requested_tiers"] == "host", f"{arm}: identity context")
    for key in IDENTITY_SHARED:
        require(identity[key] == baseline["identity"][key], f"{arm}: identity {key} differs from the frozen bundle")
    summary = fields(r / "FAULT-ARM.txt")
    rows = list(csv.DictReader((r / "fault-checks.tsv").open(), delimiter="\t"))
    require(rows and all(set(row) == {"check", "expected", "observed", "ok"} for row in rows), f"{arm}: checks header")
    for row in rows:
        require(row["ok"] == str(row["expected"] == row["observed"]).lower(), f"{arm}: ok column {row}")
        require(row["check"] in REQUIRED[arm], f"{arm}: unexpected check {row['check']}")
    ok, line = verdict(arm, rows)
    require(summary["arm"] == arm and summary["line"] == line, f"{arm}: recorded line differs from the pure verdict")
    require(summary["verdict"] == ("PASS" if ok else "REFUSED"), f"{arm}: verdict {summary['verdict']}")
    require(summary["continues"] == str(CONTINUES[arm]).lower() and
            summary["restores_cache"] == str(RESTORES[arm]).lower(), f"{arm}: shape flags")
    require(summary["faulted_role"] == "Key" and int(summary["faulted_valid_bytes"]) > 0 or arm == "host-budget-short",
            f"{arm}: faulted plane")
    prefix = sha(r / "prefix-state.tsv")
    require(prefix == baseline["surfaces_sha256"]["prefix-state.tsv"], f"{arm}: suspended prefix differs from the bundle")
    if RESTORES[arm]:
        require(sha(r / "restored-prefix-state.tsv") == prefix, f"{arm}: restored prefix differs")
    else:
        require(not (r / "restored-prefix-state.tsv").exists(), f"{arm}: incomplete cache must not be captured")
    if CONTINUES[arm]:
        active = (r / "ACTIVE.txt").read_text().splitlines()
        require(active[0] == line and "committed=8192" in active, f"{arm}: ACTIVE.txt header")
        for surface in SURFACES:
            require(decoded_sha(r / surface) == baseline["surfaces_sha256"][surface],
                    f"{arm}: {surface} differs from the frozen bundle")
        require(len(data(r / "tokens.u32le")) == 128 * 4, f"{arm}: token count")
    else:
        for name in ("tokens.u32le", "ACTIVE.txt", "logits.tsv", "output.txt", "final-state.tsv"):
            require(not (r / name).exists() and not Path(str(r / name) + ".gz").exists(), f"{arm}: {name} written")
    if arm in REFUSAL:
        require((r / "REFUSED.txt").read_text().strip() == line, f"{arm}: REFUSED.txt")
        require(summary["finding"] == line, f"{arm}: finding")
    else:
        require(not (r / "REFUSED.txt").exists(), f"{arm}: REFUSED.txt on a passing arm")
    return summary, rows, line


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE / "pro-single-day11")
    p.add_argument("--baselines", type=Path, default=HERE.parent / "spill-b-20260919/BOX3-BASELINES.json")
    a = p.parse_args()
    bundle = json.loads(a.baselines.read_text())
    baseline = bundle["baselines"]["8192"]
    # Every native build attempt is retained; the cells must bind to one whose build, clippy and
    # tests all exited 0 on a clean tree (the first day-11 attempt kept a clippy failure).
    builds = []
    for build_dir in sorted(a.root.glob("build*")):
        exits = {name: (build_dir / name).read_text().strip() for name in ("exit", "clippy.exit", "tests.exit")}
        record = {"dir": build_dir.name, "exits": exits,
                  "source": (build_dir / "source.txt").read_text().strip(),
                  "binary": (build_dir / "binary.sha256").read_text().split()[0],
                  "dirty": (build_dir / "dirty.txt").read_text().strip(),
                  "green": all(v == "0" for v in exits.values()) and (build_dir / "dirty.txt").read_text().strip() == ""}
        builds.append(record)
    cell_binary = (a.root / "binary.sha256").read_text().split()[0]
    cell_source = (a.root / "source.commit").read_text().strip()
    green = [b for b in builds if b["green"] and b["binary"] == cell_binary and b["source"] == cell_source]
    require(len(green) == 1, "the cells do not bind to exactly one green native build receipt")
    build = green[0]
    overall = json.loads((a.root / "validate.json").read_text())
    require(overall["cells"] == len(ARMS) and overall["failed_commands"] == 0 and
            overall["refused_commands"] == len(REFUSAL) and overall["qualification"] is False, "root --validate")
    table = []
    temps, powers = [], []
    for arm in ARMS:
        folder = a.root / arm
        last, samples = collector(folder, arm, build)
        summary, rows, line = receipt(folder, arm, baseline, build)
        # An incomplete arm stops at the committed prompt (context minus the 128 it never generates).
        committed = "8192" if CONTINUES[arm] else baseline["identity"]["prompt_tokens"]
        generated = "128" if CONTINUES[arm] else "0"
        expected_last = line if arm in REFUSAL else f"{line} committed={committed} generated={generated}"
        require(last == expected_last, f"{arm}: last console line {last!r}")
        require(last == (folder / "last-line.txt").read_text().rstrip("\n"), f"{arm}: last-line receipt")
        for s in samples:
            s = {k.strip(): v.strip() for k, v in s.items()}
            temps.append(int(s["temperature.gpu"]))
            powers.append(float(s["power.draw [W]"].split()[0]))
        table.append({"arm": arm, "injection": INJECTION[arm],
                      "expected": "typed refusal (missing seam)" if arm in REFUSAL else
                      ("PASS, intact-resident, continuation matches the frozen bundle" if CONTINUES[arm]
                       else "PASS, no publish, no token"),
                      "printed": last, "checks": len(rows),
                      "failed": [r["check"] for r in rows if r["ok"] == "false"],
                      "faulted_layer": summary["faulted_layer"], "faulted_valid_bytes": summary["faulted_valid_bytes"],
                      "observations": {k[len("observation."):]: v for k, v in summary.items()
                                       if k.startswith("observation.")},
                      "receipt": str(folder.relative_to(a.root.parent.parent))})
    print(json.dumps({"kind": "day11-fault-arms-replay", "arms": len(table), "source": build["source"],
                      "binary_sha256": build["binary"], "build_receipt": build["dir"],
                      "build_attempts": [{"dir": b["dir"], "exits": b["exits"], "green": b["green"]} for b in builds],
                      "frozen_bundle_source": baseline["source"],
                      "telemetry": {"samples": len(temps), "gpu_temperature_c": [min(temps), max(temps)],
                                    "power_draw_w": [min(powers), max(powers)], "power_limit_w": 600},
                      "qualification": False, "table": table}, indent=2))
    print("DAY11 FAULT-ARM REPLAY MATCH: 7 arms; 5 PASS, 2 typed refusals; N=1 each; NOT qualification")


if __name__ == "__main__":
    main()
