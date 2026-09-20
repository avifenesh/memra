#!/usr/bin/env python3
"""Replay exact pressure receipts; CPU checks never stand in for missing GPU cells."""
import argparse
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
    require(not capture["timed_out"], "timed out: " + case)
    require(capture["qualification"] is False, "collector cannot qualify")
    require(capture["gpu_telemetry"]["interval_ms"] == 250, "telemetry cadence")
    require(capture["gpu_power_limits"] == [{"device": "0", "power.limit": "400.00 W", "power.max_limit": "600.00 W"}], "power regime")
    require(json.loads((root / "lock.json").read_text()) == {"rig": "rtx5090", "lock": "/tmp/memra-5090.lock", "acquired": True}, "canonical collector lock")
    argv = capture["command"]
    require("MEMRA_NGEN=32" in argv and "MEMRA_MOE_RESIDENT=0" in argv, "different request configuration")
    log = (root / "command.log").read_text()
    if case == "host-refusal":
        require(capture["exit_code"] != 0, "refusal unexpectedly passed")
        quote = "experts-via-tier host bank budget cannot hold one expert record"
        require(quote in log and not TRACE.findall(log), "missing pre-dispatch capacity refusal")
        require("--expert-bank-host-bytes=1" in argv, "refusal budget changed")
        return {"case": case, "verdict": quote, "exit_code": capture["exit_code"]}, log
    require(capture["exit_code"] == 0, "cell failed: " + case)
    gb = int(case[0])
    slots = (gb * 1024**3) // (860160 + 8)
    require(f"MEMRA_MOE_SLOTS={slots}" in argv, "different GPU slot budget")
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
        require(matches, "host trace absent")
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
    out = LANE / "raw/day8-cpu/final"
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
    (out / "checks.json").write_text(json.dumps({"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "scope": "CPU only; engine Linux compile-only placeholder", "checks": results}, indent=2) + "\n")
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
