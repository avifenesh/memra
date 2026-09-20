#!/usr/bin/env python3
"""Replay exact pressure receipts; CPU checks never stand in for missing GPU cells."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "research/spill-c-20260919"
RAW = LANE / "rented-5090-20260919/day8"
SPEC = importlib.util.spec_from_file_location("day7", LANE / "verify-day7.py")
DAY7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY7)
require = DAY7.require
CASES = [f"{gb}g-{gate}-{arm}" for gb in (4, 8) for gate in ("gen", "spec") for arm in ("off", "on")] + ["host-refusal"]
TRACE = re.compile(r"^\[expert-host-slru\] key=(\d+:\d+:\d+) bytes=(\d+) slot=(\d+) hit=(true|false) victim=(-|\d+:\d+:\d+)$", re.M)


def replay(case):
    root = RAW / case
    capture = json.loads((root / "command.capture.json").read_text())
    DAY7.hashes(root, capture)
    require(not capture["timed_out"] and capture["parse_error"] is None, "timed out or malformed: " + case)
    require(capture["qualification"] is False and capture["status"] == ("refused" if case == "host-refusal" else "executed-not-qualified"), "collector cannot qualify")
    require(capture["gpu_telemetry"]["interval_ms"] == 250, "telemetry cadence")
    require(capture["gpu_power_limits"] == [{"device": "0", "power.limit": "400.00 W", "power.max_limit": "600.00 W"}], "power regime")
    require(json.loads((root / "lock.json").read_text()) == {"rig": "rtx5090", "lock": "/tmp/memra-5090.lock", "acquired": True}, "canonical collector lock")
    argv = capture["command"]
    require("MEMRA_NGEN=32" in argv and "MEMRA_MOE_RESIDENT=0" in argv, "different request configuration")
    log = (root / "command.log").read_text()
    if case == "host-refusal":
        require(capture["exit_code"] == 2, "refusal must exit 2")
        quote = "experts-via-tier host bank budget cannot hold one expert record"
        require("REFUSED: " + quote in log and "native_exit_code=1" in log and not TRACE.findall(log), "missing pre-dispatch capacity refusal")
        expected = ["python3", "/root/spill-c-day8/pressure-refusal.py", "env", "MEMRA_MOE_RESIDENT=0", "MEMRA_MOE_SLOTS=4993", "MEMRA_NGEN=32", "/root/spill-c-day8/bin/run-gen", "/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "55", "88", "13", "--experts-via-tier", "--expert-bank-host-bytes=1"]
        require(argv == expected, "refusal command/budget changed")
        require(capture["failure_quote"] == "REFUSED: " + quote, "collector refusal quote changed")
        return {"case": case, "verdict": "REFUSED: " + quote, "exit_code": capture["exit_code"]}, log
    require(capture["exit_code"] == 0, "cell failed: " + case)
    gb = int(case[0])
    slots = (gb * 1024**3) // (860160 + 8)
    require(f"MEMRA_MOE_SLOTS={slots}" in argv, "different GPU slot budget")
    gate = "gen" if "-gen-" in case else "spec"
    expected_argv = ["env", "MEMRA_MOE_RESIDENT=0", f"MEMRA_MOE_SLOTS={slots}", "MEMRA_NGEN=32", f"/root/spill-c-day8/bin/run-{gate}", "/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "55", "88", "13"]
    if case.endswith("-on"):
        expected_argv.append("--experts-via-tier")
    require(argv == expected_argv, "runtime/artifact/prompt/arm command identity changed")
    tape = DAY7.tokens(log)
    if "-gen-" in case:
        verdict = "prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH"
        require(verdict in log, "argmax failed")
    else:
        verdict = "=== SELF-CONSISTENCY PASS ==="
        require(verdict in log, "spec failed")
        require(re.findall(r"\[generate_spec K=(\d)\]", log) == list("12345678"), "incomplete K ladder")
        rows = re.findall(r"acceptance: ([^\n]+)", log)
        require(len(rows) == 8 and all("self-consistency: PASS" in r for r in rows), "failed spec row")
    result = {"case": case, "verdict": verdict, "tokens": len(tape)}
    if case.endswith("-on"):
        require("--experts-via-tier" in argv, "bank installer absent")
        matches = TRACE.findall(log)
        require(matches and len(matches) == log.count("[expert-host-slru]"), "host trace absent/malformed")
        seen = set()
        rereads = evictions = misses = 0
        for key, _, _, hit, victim in matches:
            if hit == "false":
                rereads += int(key in seen)
                misses += 1
            seen.add(key)
            evictions += int(victim != "-")
        physical = re.search(r"physical_reads=(\d+) owner_close=Ok\(\(\)\)", log)
        require(physical and int(physical[1]) == misses, "read count does not match host misses/drain")
        gpu = re.search(r"\[expert-gpu-slru\] slots=(\d+) allocated_bytes=(\d+) evictions=(\d+)", log)
        require(gpu and int(gpu[1]) == slots and int(gpu[2]) == slots * 860168, "GPU allocation differs from budget")
        require(int(gpu[3]) > 0 and evictions > 0 and rereads > 0, "pressure did not engage")
        result.update(gpu_slots=slots, gpu_allocated_bytes=int(gpu[2]), gpu_evictions=int(gpu[3]), host_demands=len(matches), host_evictions=evictions, physical_reads=int(physical[1]), rereads=rereads)
    else:
        require("--experts-via-tier" not in argv, "control unexpectedly banked")
    return result, log


def cpu():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = LANE / "raw/day8-cpu" / ("recheck-" + stamp)
    out.mkdir(parents=True, exist_ok=True)
    commands = {
        "fmt": ["cargo", "fmt", "--all", "--", "--check"],
        "mac-check": ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets"],
        "linux-check": ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"],
        "tests": ["cargo", "test", "-p", "memra-tier", "--offline", "--no-fail-fast"],
        "clippy": ["cargo", "clippy", "-p", "memra-tier", "--offline", "--all-targets", "--", "-D", "warnings"],
        "engine-linux-clippy": ["cargo", "clippy", "-p", "memra-engine", "--offline", "--lib", "--bin", "run-gen", "--bin", "run-spec", "--target", "x86_64-unknown-linux-gnu", "--", "-D", "warnings"],
        "diff": ["git", "diff", "--check"],
        "flags": ["bash", "tools/check-flags.sh"],
        "frozen": ["python3", "research/spill-c-20260919/slru-trace.py", "--check"],
        "verifier-red": ["python3", "research/spill-c-20260919/test-day8.py"],
    }
    results = []
    for name, argv in commands.items():
        env = os.environ.copy()
        if name == "engine-linux-clippy":
            env.update(DOCS_RS="1", MEMRA_MMQ_ARCHIVE_HASH="0000000000000000")
        path = out / (name + ".log")
        with path.open("wb") as log:
            run = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        results.append({"name": name, "argv": argv, "exit": run.returncode, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, run.returncode, flush=True)
    files = list((ROOT / "crates/memra-tier/src/bank").glob("*.rs"))
    files += list((ROOT / "crates/memra-tier/tests/bank").glob("*.rs"))
    files += [ROOT / "crates/memra-engine/src" / f for f in ["moe_cache.rs", "banked_residency.rs", "banked_residency/native.rs"]]
    files += [LANE / f for f in ["verify-day7.py", "verify-day8.py", "pressure-refusal.py", "test-day8.py"]]
    tested_files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (out / "checks.json").write_text(json.dumps({"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "scope": "CPU only; engine Linux compile-only placeholder", "checks": results, "tested_files_sha256": tested_files}, indent=2) + "\n")
    print("CPU receipt:", out.relative_to(ROOT), flush=True)
    require(all(r["exit"] == 0 for r in results), "CPU checks failed; inspect raw logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.cpu:
        cpu()
        return
    names = [args.case] if args.case else CASES
    results, logs = {}, {}
    for name in names:
        results[name], logs[name] = replay(name)
    if not args.case:
        source = (RAW / "source.txt").read_text().strip()
        require(source == "44f87f181bbbcc75a14d8d9362609f60186f293d", "unexpected runtime source")
        before = (RAW / "binaries.sha256").read_text()
        require(before == (RAW / "binaries-post.sha256").read_text(), "runtime binaries changed during cells")
        binary_rows = {Path(line.split()[1]).name: line.split()[0] for line in before.splitlines()}
        require(binary_rows == {"run-gen": "e3d86df83364d3339c75214cfbb55a6fb6cdc35524b89bd61f65f5911bd4c368", "run-spec": "59e17c94c9470518d49e6efae6a7d05c1ccf202b66db090e9eea57251ffb4c80"}, "runtime binary identity changed")
        post = json.loads((RAW / "binary-postcheck.json").read_text())
        require(post["runtime_source"] == source and post["runtime_worktree_clean"] is True, "runtime postcheck source/cleanliness mismatch")
        require(set(post["completed_cells"]) == set(CASES) and post["binary_sha256"] == binary_rows, "runtime postcheck incomplete or binary mismatch")
        checked = datetime.datetime.fromisoformat(post["checked_utc"])
        for case in CASES:
            capture = json.loads((RAW / case / "command.capture.json").read_text())
            require(datetime.datetime.fromisoformat(capture["ended_utc"]) <= checked, "binary postcheck preceded a cell")
        wrapper_hash = (RAW / "refusal-wrapper.sha256").read_text().split()[0]
        require(wrapper_hash == hashlib.sha256((LANE / "pressure-refusal.py").read_bytes()).hexdigest(), "refusal wrapper identity changed")
        require((RAW / "native-clippy.exit").read_text().strip() == "0", "native clippy failed")
        for gb in (4, 8):
            for gate in ("gen", "spec"):
                off, on = (logs[f"{gb}g-{gate}-{arm}"] for arm in ("off", "on"))
                require(DAY7.tokens(off) == DAY7.tokens(on), "ON/OFF tape mismatch")
                baseline = (LANE / "rented-5090-20260919/day7" / f"final-{gate}-off/command.log").read_text()
                require(DAY7.tokens(off) == DAY7.tokens(baseline), "pressure changed token tape")
                if gate == "spec":
                    require(re.findall(r"acceptance: ([^\n]+)", off) == re.findall(r"acceptance: ([^\n]+)", on), "ON/OFF acceptance changed")
    print(json.dumps({"receipt_replay": "PASS", "scope": "rented RTX 5090 development correctness only", "production_qualified": False, "cells": list(results.values())}, indent=2))


if __name__ == "__main__":
    main()
