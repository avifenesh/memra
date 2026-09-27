#!/usr/bin/env python3
"""Offline replay of the day-12 cells: the seven fault arms through lane A's two rule seams
(`recover_source`, `suspend_layer`/`resume_layer`) and the two tier-transfer-gate cases that
carry A's day-11 rule lines natively. Collector receipts, the pure fault verdict over the
recorded checks, and the continuation surfaces against the frozen target-card bundle.
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
TRANSFER_CASES = ("conformance", "roundtrip")
# Mirror of `fault_contract::Arm::required_checks` (kv_tier_gate/fault_contract.rs, day 12); the
# Rust harness replays the same receipts, this file replays them without a Rust toolchain.
REQUIRED = {
    "cancel-demote": ("cancel", "take-after-cancel", "retire", "acknowledge", "pinned-after-cancel",
                      "source-returned", "budget-zero", "restored-identical"),
    "cancel-restore": ("cancel", "ready-view-after-cancel", "with-destination-after-cancel",
                       "take-after-cancel", "retire-holds-source", "retire-source-holds", "recover-source",
                       "recovered-source-checksum", "recovered-source-intact", "recover-source-once",
                       "cancel-after-recovery", "retire", "acknowledge", "pinned-held-by-recovered-lease",
                       "retake-demoted-copy", "budget-zero", "restored-identical"),
    "corrupt-host": ("write-under-live-ticket", "write-sole-owner", "restore-integrity",
                     "device-registry-unchanged", "budget-zero"),
    "missing-host": ("completion-checksum", "remove", "pinned-released", "retake-removed-copy", "budget-zero"),
    "host-budget-short": ("whole-state-admission", "layers-resident", "pinned-charged", "budget-zero",
                          "restored-identical"),
    "device-short": ("restore-admission", "device-registry-unchanged", "host-copy-intact", "budget-zero",
                     "restored-identical"),
    "require-resident": ("suspended-register", "continuation-gate-on-suspended-cache",
                         "continuation-gate-asked-twice", "continuation-gate-after-partial-resume",
                         "register-empty-after-resume", "continuation-gate-after-resume", "budget-zero",
                         "restored-identical"),
}
RESTORES = {arm: arm not in ("corrupt-host", "missing-host") for arm in ARMS}
CONTINUES = dict(RESTORES)
# Fault injection points as the arms document them; printed in the table for the issue comment.
INJECTION = {
    "cancel-demote": "TransferEngine::cancel after D2H synchronize, before take_destination",
    "cancel-restore": "TransferEngine::cancel after H2D synchronize, before ready_view; rule 1: retire holds, "
                      "recover_source hands the demoted copy back once, same plane restored",
    "corrupt-host": "CudaPinnedLease::write (Busy under the live D2H ticket; Ok after its drain), then restore",
    "missing-host": "untaken D2H ticket retired and acknowledged; take_destination at restore",
    "host-budget-short": "governor pinned capacity = whole demoted bytes minus 1; admit_whole_state",
    "device-short": "competing tenant reserves the governor device dimension; restore alloc_device",
    "require-resident": "Cache::suspend_layer for every plane; Cache::ensure_usable asked while suspended and after "
                        "a partial resume (rule 2: ContinuationRefused naming the layers), Ok after resume_layer",
}
# The two day-11 rule lines A could not run natively, plus the drain line every conformance run ends on.
RULE_LINES = ("PASS rule cancelled-restore-recovers-source native CUDA",
              "PASS rule cancel-refused-after-source-consumed native CUDA",
              "PASS native governor zero after controlled drain")
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
    """The pure rule of fault_contract::verdict, replayed in Python. Day 12: no refusal branch."""
    names = [c["check"] for c in checks]
    missing = [name for name in REQUIRED[arm] if name not in names]
    failed = []
    for c in checks:
        if c["expected"] != c["observed"] and c["check"] not in failed:
            failed.append(c["check"])
    if missing or failed:
        return False, (f"fault arm {arm} did not prove its contract; "
                       f"missing=[{','.join(missing)}] failed=[{','.join(failed)}]")
    return True, f"FAULT-ARM PASS {arm}"


def collector(folder, label, binary, source):
    """Collector receipts of one cell: canonical rig and lock, envelope, journal, hashes, no co-tenant,
    binding to the green build. Returns the capture, the raw log lines and the telemetry samples."""
    exit_code = int((folder / "collector.exit").read_text().strip())
    attempt = int((folder / "launched-attempt").read_text())
    c = folder / f"collector-{attempt}"
    require(json.loads((c / "lock.json").read_text()) ==
            {"rig": "pro-single", "lock": "/tmp/memra-gpu.lock", "acquired": True}, f"{label}: wrong rig/lock")
    cap = json.loads((c / "command.capture.json").read_text())
    require(cap["qualification"] is False and not cap["timed_out"], f"{label}: promoted or timed-out capture")
    require(cap["exit_code"] == 0 and cap["status"] == "executed-not-qualified" and exit_code == 0,
            f"{label}: exit {cap['exit_code']} status {cap['status']} (collector exit {exit_code})")
    require(cap["gpu_power_limits"] == [{"device": "0", "power.limit": "600.00 W",
                                         "power.max_limit": "600.00 W"}], f"{label}: power envelope")
    cells = [json.loads(x) for x in (c / "CELL.jsonl").read_text().splitlines()]
    require([x["event"] for x in cells] == ["start", "end"], f"{label}: incomplete journal")
    require(cells[-1]["exit_code"] == 0 and cells[-1]["qualification"] is False, f"{label}: bad CELL end")
    require(cells[-1]["capture"]["sha256"] == sha(c / "command.capture.json"), f"{label}: capture hash")
    require(cap["raw_log"]["sha256"] == sha(c / "command.log"), f"{label}: raw log hash")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, f"{label}: telemetry interval")
    require(cap["gpu_telemetry"]["raw_csv"]["sha256"] == sha(c / "command.gpu.csv"), f"{label}: telemetry hash")
    samples = list(csv.DictReader((c / "command.gpu.csv").open()))
    require(samples, f"{label}: empty telemetry")
    for sample in samples:
        sample = {k.strip(): v.strip() for k, v in sample.items()}
        require(sample["power.limit [W]"] == sample["power.max_limit [W]"] == "600.00 W", f"{label}: power changed")
    for phase in ("before", "after"):
        path = c / f"command.{phase}.log"
        require(len(path.read_text().splitlines()) == 1, f"{label}: co-tenant snapshot {phase}")
        require(sha(path) == cap["compute_apps"][phase]["raw_log"]["sha256"], f"{label}: snapshot hash")
    log = (c / "command.log").read_text()
    require(not any(x in log for x in ("CUDA_ERROR_", "panicked at", "kv-tier-gate: fault arm", "REFUSED:")),
            f"{label}: runtime error, failed arm or refusal in the raw log")
    require((folder / "source.commit").read_text().strip() == source, f"{label}: source commit")
    require((folder / "binary.sha256").read_text().split()[0] == binary, f"{label}: binary hash")
    require(cap["command"][0].endswith(cap["command"][0].rsplit("/", 1)[-1]), f"{label}: command shape")
    validate = json.loads((folder / "validate.json").read_text())
    require(validate["kind"] == "capture-integrity" and validate["cells"] == 1 and
            validate["failed_commands"] == 0 and validate["refused_commands"] == 0 and
            validate["qualification"] is False, f"{label}: collector --validate")
    lines = log.splitlines()
    require(lines and lines[-1] == (folder / "last-line.txt").read_text().rstrip("\n"), f"{label}: last-line receipt")
    return cap, lines, samples


def arm_cell(folder, arm, baseline, build):
    cap, lines, samples = collector(folder, arm, build["binary"], build["source"])
    command = cap["command"]
    for flag, value in (("--case", "active"), ("--kv-allocator", "pooled"), ("--context", "8192"),
                        ("--tiers", "host"), ("--fault", arm)):
        require(command[command.index(flag) + 1] == value, f"{arm}: command {flag} is not {value}")
    require("--same-program" in command and "--reclaim-diagnostic" not in command, f"{arm}: program flags")
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
        # Every recorded row must hold (ok column); extra rows are evidence, the required set is
        # enforced by verdict(): the same rule as the Rust replay.
        require(row["ok"] == str(row["expected"] == row["observed"]).lower(), f"{arm}: ok column {row}")
    ok, line = verdict(arm, rows)
    require(ok, f"{arm}: {line}")
    require(summary["arm"] == arm and summary["line"] == line, f"{arm}: recorded line differs from the pure verdict")
    require(summary["verdict"] == "PASS" and "finding" not in summary, f"{arm}: verdict {summary['verdict']}")
    require(summary["continues"] == str(CONTINUES[arm]).lower() and
            summary["restores_cache"] == str(RESTORES[arm]).lower(), f"{arm}: shape flags")
    require(summary["faulted_role"] == "Key" and int(summary["faulted_valid_bytes"]) > 0 or arm == "host-budget-short",
            f"{arm}: faulted plane")
    require(not (r / "REFUSED.txt").exists(), f"{arm}: REFUSED.txt on a passing arm")
    prefix = sha(r / "prefix-state.tsv")
    require(prefix == baseline["surfaces_sha256"]["prefix-state.tsv"], f"{arm}: suspended prefix differs from the bundle")
    if RESTORES[arm]:
        require(sha(r / "restored-prefix-state.tsv") == prefix, f"{arm}: restored prefix differs")
    else:
        require(not (r / "restored-prefix-state.tsv").exists(), f"{arm}: incomplete cache must not be captured")
        holed = [row for row in rows if row["check"] == "holed-cache-refuses-continuation"]
        require(holed and holed[0]["observed"] == "Err", f"{arm}: holed cache did not refuse a continuation")
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
    committed = "8192" if CONTINUES[arm] else baseline["identity"]["prompt_tokens"]
    generated = "128" if CONTINUES[arm] else "0"
    require(lines[-1] == f"{line} committed={committed} generated={generated}", f"{arm}: last console line {lines[-1]!r}")
    return {"arm": arm, "injection": INJECTION[arm],
            "expected": ("PASS, cache whole and bit-identical, continuation matches the frozen bundle"
                         if CONTINUES[arm] else "PASS, no publish, no token, holed cache refuses a continuation"),
            "printed": lines[-1], "checks": len(rows), "failed": [r["check"] for r in rows if r["ok"] == "false"],
            "faulted_layer": summary["faulted_layer"], "faulted_valid_bytes": summary["faulted_valid_bytes"],
            "observations": {k[len("observation."):]: v for k, v in summary.items() if k.startswith("observation.")},
            "receipt": str(folder.relative_to(HERE.parent))}, samples


def transfer_cell(folder, case, build):
    cap, lines, samples = collector(folder, f"tier-transfer-gate {case}", build["transfer_binary"], build["source"])
    require(cap["command"][-1] == case and len(cap["command"]) == 2, f"{case}: command is not <binary> {case}")
    verdicts = [line for line in lines if line.startswith("PASS ")]
    require(verdicts and all(line.startswith("PASS ") for line in lines if line.strip()), f"{case}: non-PASS output")
    if case == "conformance":
        for rule in RULE_LINES:
            require(rule in lines, f"{case}: missing line {rule!r}")
        require(lines[-1] == RULE_LINES[-1], f"{case}: last line {lines[-1]!r}")
    else:
        require(all(line.startswith("PASS native D2H-H2D roundtrip bytes=") and "byte_exact=true" in line
                    for line in verdicts), f"{case}: roundtrip lines")
    return {"case": case, "lines": len(verdicts), "printed_last": lines[-1],
            "rule_lines": [line for line in lines if line in RULE_LINES[:2]],
            "receipt": str(folder.relative_to(HERE.parent))}, samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=HERE / "pro-single-day12")
    p.add_argument("--baselines", type=Path, default=HERE.parent / "spill-b-20260919/BOX3-BASELINES.json")
    a = p.parse_args()
    bundle = json.loads(a.baselines.read_text())
    baseline = bundle["baselines"]["8192"]
    # Every native build attempt is retained; the cells must bind to one whose build, clippy and
    # tests all exited 0 on a clean tree.
    builds = []
    for build_dir in sorted(a.root.glob("build*")):
        exits = {name: (build_dir / name).read_text().strip() for name in ("exit", "clippy.exit", "tests.exit")}
        record = {"dir": build_dir.name, "exits": exits,
                  "source": (build_dir / "source.txt").read_text().strip(),
                  "binary": (build_dir / "binary.sha256").read_text().split()[0],
                  "transfer_binary": (build_dir / "transfer-binary.sha256").read_text().split()[0],
                  "dirty": (build_dir / "dirty.txt").read_text().strip(),
                  "green": all(v == "0" for v in exits.values()) and (build_dir / "dirty.txt").read_text().strip() == ""}
        builds.append(record)
    cell_binary = (a.root / "binary.sha256").read_text().split()[0]
    cell_source = (a.root / "source.commit").read_text().strip()
    transfer_root = a.root / "transfer-gate"
    transfer_binary = (transfer_root / "binary.sha256").read_text().split()[0]
    require((transfer_root / "source.commit").read_text().strip() == cell_source, "transfer-gate source commit")
    green = [b for b in builds if b["green"] and b["binary"] == cell_binary and b["source"] == cell_source
             and b["transfer_binary"] == transfer_binary]
    require(len(green) == 1, "the cells do not bind to exactly one green native build receipt")
    build = green[0]
    overall = json.loads((a.root / "validate.json").read_text())
    require(overall["cells"] == len(ARMS) + len(TRANSFER_CASES) and overall["failed_commands"] == 0 and
            overall["refused_commands"] == 0 and overall["qualification"] is False, "root --validate")
    table, transfer, temps, powers = [], [], [], []
    for arm in ARMS:
        row, samples = arm_cell(a.root / arm, arm, baseline, build)
        table.append(row)
        for s in samples:
            s = {k.strip(): v.strip() for k, v in s.items()}
            temps.append(int(s["temperature.gpu"]))
            powers.append(float(s["power.draw [W]"].split()[0]))
    for case in TRANSFER_CASES:
        row, samples = transfer_cell(transfer_root / case, case, build)
        transfer.append(row)
        for s in samples:
            s = {k.strip(): v.strip() for k, v in s.items()}
            temps.append(int(s["temperature.gpu"]))
            powers.append(float(s["power.draw [W]"].split()[0]))
    print(json.dumps({"kind": "day12-fault-arms-replay", "arms": len(table), "source": build["source"],
                      "binary_sha256": build["binary"], "transfer_binary_sha256": build["transfer_binary"],
                      "build_receipt": build["dir"],
                      "build_attempts": [{"dir": b["dir"], "exits": b["exits"], "green": b["green"]} for b in builds],
                      "frozen_bundle_source": baseline["source"],
                      "telemetry": {"samples": len(temps), "gpu_temperature_c": [min(temps), max(temps)],
                                    "power_draw_w": [min(powers), max(powers)], "power_limit_w": 600},
                      "qualification": False, "table": table, "transfer_gate": transfer}, indent=2))
    print(f"DAY12 FAULT-ARM REPLAY MATCH: {len(ARMS)} arms; {len(ARMS)} PASS, 0 refusals; N=1 each; NOT qualification")
    print("DAY12 TIER-TRANSFER-GATE REPLAY MATCH: 2 cells; both day-11 rule lines printed natively; NOT qualification")


if __name__ == "__main__":
    main()
