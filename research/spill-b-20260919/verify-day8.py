#!/usr/bin/env python3
"""Replay day-8 native receipt integrity, not another GPU or serving gate."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

LANE = Path(__file__).resolve().parent
ROOT = LANE / "rented-5090-20260919"
SOURCE = "c9569169"
VERDICT = "ACTIVE-8K copy/restore bit-identical, no reclaim — not G1 PASS"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def classify(metrics, identical):
    """Never promote positive counters or allocator bookkeeping to observed reclaim."""
    if not identical:
        raise ValueError("continuation mismatch")
    require(metrics["demote_count"] > 0 and metrics["reload_count"] == metrics["demote_count"],
            "missing engagement")
    require(metrics["device_charged_after_demote"] == 0, "source still charged")
    require(metrics["pinned_charged_after_demote"] == metrics["logical_d2h_bytes"] > 0,
            "host accounting mismatch")
    before = metrics["free_before_bytes"]
    demoted = metrics["free_after_demote_bytes"]
    restored = metrics["free_after_restore_bytes"]
    require(metrics["reclaimed_bytes"] == demoted - before, "reclaim delta mismatch")
    require(metrics["reacquired_bytes"] == demoted - restored, "restore delta mismatch")
    reclaim = demoted > before and demoted > restored
    require(metrics["reclaim_observed"] == reclaim, "reclaim observation mismatch")
    return "ACTIVE-8K G1 PASS" if reclaim else VERDICT


def replay():
    manifest = json.loads((LANE / "day8-manifest.json").read_text())
    for row in manifest:
        path = LANE / row["path"]
        require(path.resolve().is_relative_to(LANE), "escaping receipt")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"receipt hash mismatch: {row['path']}")
    cell = ROOT / "day8-active-8192"
    receipt = cell / "receipt"
    baseline = ROOT / "day6-baseline-8192/receipt"
    build = ROOT / "day8-build"
    source = (cell / "source.commit").read_text().strip()
    require(source.startswith(SOURCE) and len(source) == 40, "wrong native source")
    require(source == (build / "source.commit").read_text().strip(), "build source mismatch")
    binary = (build / "binary.sha256").read_text().split()[0]
    require((cell / "binary.sha256").read_text().split()[0] == binary, "binary mismatch")
    identity = fields(receipt / "identity.txt")
    require(identity["binary_sha256"] == binary, "identity binary mismatch")
    baseline_identity = fields(baseline / "identity.txt")
    for key in ["artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "mode", "context"]:
        require(identity[key] == baseline_identity[key], f"baseline identity mismatch: {key}")
    # Use the actual collector validator for descriptor hashes, lock and CELL mirrors.
    spec = importlib.util.spec_from_file_location("tier_battery", LANE.parents[1] / "tools/tier-battery.py")
    battery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(battery)
    battery.validate_cell(cell / "collector/CELL.jsonl")
    capture = json.loads((cell / "collector/command.capture.json").read_text())
    require(capture["exit_code"] == 0 and capture["qualification"] is False,
            "not a successful unqualified capture")
    require(capture["status"] == "executed-not-qualified" and not capture["timed_out"], "bad capture")
    require(capture["command"][1:] == [
        "--artifact", "/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf",
        "--case", "active", "--context", "8192", "--tiers", "host", "--same-program",
        "--out", "/root/wt-b/research/spill-b-20260919/rented-5090-20260919/day8-active-8192/receipt"
    ], "wrong cell command")
    require(capture["gpu_telemetry"]["interval_ms"] == 250, "wrong sampler interval")
    require(capture["gpu_power_limits"] == [{"device": "0", "power.limit": "400.00 W",
                                             "power.max_limit": "600.00 W"}], "power envelope changed")
    for name in ["gpu-before.csv", "collector/command.before.log", "collector/command.after.log"]:
        require(len((cell / name).read_text().splitlines()) == 1, f"non-idle GPU: {name}")
    require(not (cell / "processes-before.txt").read_text(), "competing GPU process")
    verdict = json.loads((cell / "verdict.json").read_text())
    require(verdict["native_source"] == source, "verdict source mismatch")
    for name, expected in verdict["compared_sha256"].items():
        require(sha(receipt / name) == sha(baseline / name) == expected, f"baseline mismatch: {name}")
    require((receipt / "prefix-state.tsv").read_bytes() ==
            (receipt / "restored-prefix-state.tsv").read_bytes(), "restored state mismatch")
    active = fields(receipt / "ACTIVE.txt")
    require(active["active_engaged"] == "true" and active["prefix_engaged"] == "false", "wrong engagement")
    require(active["committed"] == "8192", "wrong context")
    for name, key in [("prefix-state.tsv", "prefix_state_manifest_sha256"),
                      ("final-state.tsv", "final_state_manifest_sha256"),
                      ("tokens.u32le", "tokens_sha256"), ("logits.tsv", "logit_rows_sha256")]:
        require(sha(receipt / name) == active[key], f"ACTIVE hash mismatch: {name}")
    rows = list(csv.DictReader((receipt / "logits.tsv").read_text().splitlines(), delimiter="\t"))
    require([int(row["committed"]) for row in rows] == list(range(8064, 8193)), "logit coverage")
    require(sha(receipt / "final-logits.f32le") == rows[-1]["logits_f32le_sha256"], "final logits")
    require(len((receipt / "tokens.u32le").read_bytes()) == 128 * 4, "token coverage")
    require(len((receipt / "prompt.u32le").read_bytes()) == 8064 * 4, "prompt coverage")
    metrics = {k: (v == "true" if k == "reclaim_observed" else int(v))
               for k, v in fields(receipt / "active-reclaim.txt").items()}
    bundles = list(csv.DictReader((receipt / "active-bundles.tsv").read_text().splitlines(), delimiter="\t"))
    require(len(bundles) == metrics["demote_count"] == 32, "bundle engagement mismatch")
    require(sum(int(r["valid_bytes"]) for r in bundles) == metrics["logical_d2h_bytes"], "bundle byte sum")
    require(metrics["layers"] == 16 and metrics["committed"] == 8064, "wrong suspended state")
    require(classify(metrics, True) == verdict["verdict"] == VERDICT, "verdict mismatch")
    # Negative arms: bookkeeping alone must not pass, and altered claims must fail closed.
    for bad in [dict(metrics, reclaim_observed=True), dict(metrics, demote_count=0),
                dict(metrics, reload_count=0), dict(metrics, reclaimed_bytes=1)]:
        try:
            classify(bad, True)
        except ValueError:
            pass
        else:
            raise ValueError("negative classification arm accepted")
    try:
        classify(metrics, False)
    except ValueError:
        pass
    else:
        raise ValueError("mismatched continuation accepted")
    require("Finished `release`" in (build / "build.log").read_text(), "native build incomplete")
    native_tests = re.findall(r"test result: ok\. (\d+) passed; 0 failed", (build / "test.log").read_text())
    require(sum(map(int, native_tests)) == 243, "native tests incomplete")
    checks = json.loads((LANE / "day8-checks/commands.json").read_text())
    require(len(checks) == 8 and all(c["exit"] == 0 for c in checks), "Mac checks incomplete")
    for check in checks:
        raw = LANE / "day8-checks" / check["raw"]
        require(sha(raw) == check["sha256"], "check log hash mismatch")
        require(hashlib.sha256(gzip.decompress(raw.read_bytes())).hexdigest() ==
                check["uncompressed_sha256"], "check raw hash mismatch")
    print(VERDICT)
    print(f"PASS integrity: {len(manifest)} files; 32 demotes/reloads; zero reclaimed/reacquired bytes; "
          "243 native tests; 8 Mac checks; 5 negative verdict arms. No 32k/G1/serving qualification.")


if __name__ == "__main__":
    replay()
