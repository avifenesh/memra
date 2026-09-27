#!/usr/bin/env python3
"""Four-tier qualification scaffold: plan and strict byte-receipt comparison, and bounded native subprocess capture.

A validation pass proves supplied byte evidence agrees; it does not prove every required
model/cell ran, qualify a numeric program, or replace step-pro and the standard battery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

CANONICAL_LOCKS = {"rtx5090": "/tmp/memra-5090.lock", "pro-single": "/tmp/memra-gpu.lock", "pro-pair": "/tmp/memra-gpu.lock", "pro-four": "/tmp/memra-gpu.lock", "cpu": None}
# Test seam (lead ruling 12, 2026-09-21): the battery's CPU tests must never contend a serving
# job's rig lock. MEMRA_TIER_BATTERY_LOCK_DIR re-roots the two canonical NAMES under a private
# directory (`<dir>/memra-5090.lock`, `<dir>/memra-gpu.lock`); the names and the rig->name table
# are unchanged and a receipt written under the seam records the private path, so it never
# validates against the canonical table in a process without the seam. Unset (the default and
# every production launcher), the table is the two rig locks. It is a test seam, not a third name.
LOCK_DIR_SEAM = "MEMRA_TIER_BATTERY_LOCK_DIR"
# The seam alone never moves a campaign: an inherited export must not make `--execute` or
# `--dry-run` flock `<private>/memra-gpu.lock` beside a serving job holding the canonical file.
# Those modes refuse under the seam unless this flag is also passed (the battery's tests pass
# it; nothing else does), the process announces the seam on stderr, and every lock.json and
# dry-run manifest written under it carries `"seam": "<dir>"`, which `--validate` refuses in any
# process whose own seam differs (review of memra #545 / PR #592, 2026-09-21).
PRIVATE_LOCK_FLAG = "--private-lock-dir-for-tests"


def active_seam():
    """The private lock directory in force for THIS process, or None. Read at call time."""
    return os.environ.get(LOCK_DIR_SEAM) or None


def lock_table(private_dir=None):
    if private_dir is None:
        return dict(CANONICAL_LOCKS)
    if not Path(private_dir).is_dir():
        raise ValueError(f"{LOCK_DIR_SEAM}={private_dir!r} is not an existing directory")
    return {rig: None if path is None else str(Path(private_dir) / Path(path).name)
            for rig, path in CANONICAL_LOCKS.items()}


LOCKS = lock_table(active_seam())
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


def evidence(root, record, allow_empty=False):
    require(set(record) == {"path", "sha256", "bytes"}, "invalid evidence descriptor")
    require(isinstance(record["path"], str), "invalid evidence path")
    rel = Path(record["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, "evidence path escapes bundle")
    path = (root / rel).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "missing/escaping evidence")
    require(type(record["bytes"]) is int and record["bytes"] >= (0 if allow_empty else 1), "empty evidence")
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
        validate_telemetry(samples, "gpu")
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


# Collector support. CPU dry runs never create positive GPU rows.
import contextlib
import csv
import datetime
import fcntl
import math
import signal
import shutil
import statistics
import subprocess
import threading
import time

INTERVAL_NS = 250_000_000


def percentiles(values):
    require(bool(values), "empty wait distribution")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), "invalid wait")
    ordered = sorted(values)
    return {f"p{p}": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)] for p in (50, 95, 99)}


def validate_telemetry(samples, kind):
    require(len(samples) >= 2, "telemetry needs window coverage")
    previous = None
    previous_sample = None
    for s in samples:
        require(set(s) == {"schema_version", "kind", "monotonic_ns", "interval_ms", "devices", "host", "nvme", "wait_ns"}, "telemetry fields")
        require(type(s["schema_version"]) is int and s["schema_version"] == 1 and type(s["interval_ms"]) is int and s["interval_ms"] == 250 and s["kind"] == kind, "telemetry version/class/cadence")
        require(type(s["monotonic_ns"]) is int and s["monotonic_ns"] >= 0, "invalid sample clock")
        if previous is not None:
            gap = s["monotonic_ns"] - previous
            require(0 < gap <= 2 * INTERVAL_NS, "telemetry gap/nonmonotonic samples")
        previous = s["monotonic_ns"]
        require(s["devices"] and len({d["device"] for d in s["devices"]}) == len(s["devices"]), "missing/duplicate devices")
        for d in s["devices"]:
            require(set(d) == {"device", "routes", "clock_mhz", "power_w", "temperature_c", "vram_bytes"}, "device telemetry fields")
            require(type(d["device"]) is int and d["device"] >= 0, "invalid device id")
            require(set(d["routes"]) == ROUTES, "route counters incomplete")
            for route in d["routes"].values():
                require(set(route) == {"bytes_in", "bytes_out"}, "route direction missing")
                require(all(type(v) is int and v >= 0 for v in route.values()), "invalid route bytes")
            require(all(v is None or (type(v) in (int,float) and math.isfinite(v) and v >= 0) for k,v in d.items() if k not in {"device", "routes"}), "invalid device measurement")
        require(set(s["host"]) == {"pinned_bytes", "pageable_bytes"}, "host counters incomplete")
        require(set(s["nvme"]) == {"queue_depth", "read_bytes", "write_bytes", "physical_bytes"}, "NVMe counters incomplete")
        for v in [*s["host"].values(), *s["nvme"].values()]:
            require(v is None or (type(v) is int and v >= 0), "invalid host/NVMe counter")
        require(set(s["wait_ns"]) == {"io", "h2d", "d2h", "p2p", "queue"}, "wait categories incomplete")
        for distribution in s["wait_ns"].values():
            require(set(distribution) == {"p50", "p95", "p99"}, "wait percentiles incomplete")
            values = list(distribution.values())
            require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values), "invalid wait percentile")
            require(distribution["p50"] <= distribution["p95"] <= distribution["p99"], "unordered waits")
        if previous_sample is not None:
            old_devices = {d["device"]:d for d in previous_sample["devices"]}
            require(set(old_devices) == {d["device"] for d in s["devices"]}, "device roster changed")
            for device in s["devices"]:
                old = old_devices[device["device"]]
                require(all(device["routes"][r][direction] >= old["routes"][r][direction] for r in ROUTES for direction in ("bytes_in", "bytes_out")), "route byte counter regressed")
            for field in ("read_bytes", "write_bytes", "physical_bytes"):
                old, new = previous_sample["nvme"][field], s["nvme"][field]
                require(old is None or new is None or new >= old, "NVMe byte counter regressed")
        previous_sample = s



@contextlib.contextmanager
def campaign_lock(rig, inherit=False):
    require(rig in LOCKS and LOCKS[rig] is not None, "GPU-shaped campaign requires canonical rig lock")
    # Never unlink a shared lock inode: waiters and other campaigns must see the same file.
    with open(LOCKS[rig], "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle if inherit else LOCKS[rig]
        finally:
            # Inherited open-file descriptions retain ownership until every child
            # closes its FD. Explicit LOCK_UN here would revoke a surviving child.
            if not inherit:
                fcntl.flock(handle, fcntl.LOCK_UN)


def paired_orders(n):
    require(type(n) is int and n >= 5, "at least five AB and five BA pairs required")
    return [(i, order) for i in range(n) for order in ("AB", "BA")]


def tee_run(command, raw_path, timeout=30, echo=True, pass_fds=(), shared_group=False):
    """Drain raw output before parsing. Nested workers MUST use shared_group=True.

    A shared-group timeout kills the worker itself as well as its children; its
    owning outer collector reaps the worker and publishes the failure capture.
    Commands must not daemonize/setsid: a process group is not a hostile sandbox.
    Pass the inherited lock FD through every nested launch to retain ownership.
    """
    with raw_path.open("xb") as log:
        try:
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=not shared_group, pass_fds=pass_fds)
        except OSError as error:
            log.write((f"ERROR: launch failed: {error}\n").encode())
            log.flush()
            return 127, False
        errors = []
        def pump():
            try:
                for line in iter(p.stdout.readline, b""):
                    log.write(line)
                    log.flush()
                    if echo:
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
            except Exception as error:
                errors.append(error)
        def kill_group():
            try:
                os.killpg(os.getpgrp() if shared_group else p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        timed_out = False
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_group()
            code = p.wait()
        thread.join(timeout=1)
        if thread.is_alive():
            # Descendants may retain stdout after their parent exits. Bound the drain.
            timed_out = True
            kill_group()
            thread.join(timeout=1)
        if thread.is_alive():
            raise TimeoutError("raw log drain remained open; no result may be published")
        p.stdout.close()
        if not shared_group:
            # Outermost commands are scoped jobs, never daemon launchers. Nested
            # successful visits leave the owning worker group alive for its next visit.
            # Reap descendants even when they closed stdout and no FD was passed.
            kill_group()
        if errors:
            raise errors[0]
    return code, timed_out


class SubprocessRunner:
    """Same argv/raw-path/timeout interface for fake and native binaries.

    The CSV is diagnostic hardware telemetry, NOT fabricated tier counters or a positive
    schema-v1 gate row. Missing sampler/counters block scoring, not raw failure retention.
    Caller holds the canonical lock across this call (including sampler and snapshots).
    """
    def __init__(self, smi="nvidia-smi"):
        self.smi = smi

    def run(self, command, raw_path, timeout=30, echo=True, run_id=None, hourly_cost=None, resume_from=None, storage=None, pass_fds=(), lock_proof=None):
        root = raw_path.parent
        stem = raw_path.stem
        started = time.monotonic_ns()
        started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        instance = os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")
        cell = {"kind": "CELL", "run_id": run_id or stem, "command": command,
                "started_utc": started_utc, "provider_instance_id": instance,
                "hourly_cost": hourly_cost, "qualification": False, "resume_from": resume_from,
                "storage": storage, "timing_scope": "collector-with-snapshots-and-sampler"}
        if lock_proof is not None:
            cell["lock_proof"] = lock_proof
        append_cell(root / "CELL.jsonl", {**cell, "event": "start"})
        smi = shutil.which(self.smi)
        snapshots = {}
        def snapshot(label):
            path = root / f"{stem}.{label}.log"
            code, expired = tee_run([smi, "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv"], path, timeout=20, echo=False)
            snapshots[label] = {"exit_code": code, "timed_out": expired,
                                "raw_log": descriptor(root, path)}
        telemetry = {"status": "unavailable", "interval_ms": 250,
                     "reason": "nvidia-smi not found", "tier_counters": "not collected"}
        sampler = None
        csv_path = root / f"{stem}.gpu.csv"
        sampler_log = root / f"{stem}.sampler.log"
        with contextlib.ExitStack() as stack:
            if smi:
                snapshot("before")
                csv = stack.enter_context(csv_path.open("xb"))
                err = stack.enter_context(sampler_log.open("xb"))
                argv = [smi, "--query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,power.max_limit,temperature.gpu,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current", "--format=csv", "-lms", "250"]
                try:
                    sampler = subprocess.Popen(argv, stdout=csv, stderr=err, start_new_session=True)
                    telemetry = {"status": "started", "interval_ms": 250, "command": argv,
                                 "tier_counters": "not collected"}
                except OSError as error:
                    err.write(str(error).encode()); err.flush()
                    telemetry["reason"] = str(error)
            try:
                code, expired = tee_run(command, raw_path, timeout, echo, pass_fds=pass_fds)
                if storage and "object_binding" in storage:
                    verify_storage_binding(storage)
                if smi and (code != 0 or expired):
                    snapshot("failure")
            finally:
                if sampler is not None:
                    previous = sampler.poll()
                    try:
                        os.killpg(sampler.pid, signal.SIGTERM)
                        sampler.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(sampler.pid, signal.SIGKILL); sampler.wait()
                    except ProcessLookupError:
                        pass
                    telemetry.update(status="captured-unvalidated" if previous is None else "exited-early",
                                     exit_code=sampler.returncode)
                if smi:
                    snapshot("after")
        ended = time.monotonic_ns()
        ended_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if smi:
            telemetry.update(raw_csv=descriptor(root, csv_path), stderr=descriptor(root, sampler_log))
            if csv_path.stat().st_size == 0:
                telemetry["status"] = "empty"
        # Only now parse: the merged log and sampler files are closed and durable to readers.
        text = raw_path.read_text(errors="replace")
        result_lines = [line[7:] for line in text.splitlines() if line.startswith("RESULT ")]
        result = None
        parse_error = None
        if code == 0 and not expired and result_lines:
            try:
                require(len(result_lines) == 1, "multiple RESULT records")
                result = json.loads(result_lines[0])
                require(isinstance(result, dict), "RESULT must be an object")
            except (ValueError, TypeError) as error:
                parse_error = str(error)
        failed = code != 0 or expired or parse_error is not None
        refusal = explicit_refusal(text, code, expired)
        power_limits = gpu_power_limits(csv_path) if smi else []
        quote = next((line for line in text.splitlines() if re.search(
            r"error|out of memory|CUDA_ERROR|fatal|panic", line, re.IGNORECASE)),
            "died, cause unknown — repro needed") if failed else None
        if refusal is not None:
            quote = refusal
        record = {"schema_version": 1, "kind": "subprocess-capture", "command": command,
                  "exit_code": code, "timed_out": expired, "parse_error": parse_error,
                  "status": "refused" if refusal is not None else ("failed" if failed else "executed-not-qualified"),
                  "gpu_power_limits": power_limits,
                  "started_monotonic_ns": started, "ended_monotonic_ns": ended,
                  "started_utc": started_utc, "ended_utc": ended_utc,
                  "elapsed_seconds": (ended-started)/1e9,
                  "timing_scope": cell["timing_scope"], "storage": storage,
                  "raw_log": descriptor(root, raw_path), "failure_quote": quote,
                  "result": result, "gpu_telemetry": telemetry, "compute_apps": snapshots,
                  "qualification": False}
        if lock_proof is not None:
            record["lock_proof"] = lock_proof
        with (root / f"{stem}.capture.json").open("x") as out:
            json.dump(record, out, indent=2); out.write("\n")
        append_cell(root / "CELL.jsonl", {**cell, "event": "end",
                    "ended_utc": ended_utc, "elapsed_seconds": (ended-started)/1e9,
                    "duration_ns": ended-started, "exit_code": code, "status": record["status"],
                    "gpu_power_limits": power_limits,
                    "estimated_cost": (ended-started)/3.6e12 * hourly_cost if hourly_cost is not None else None,
                    "capture": descriptor(root, root / f"{stem}.capture.json")})
        return record


def explicit_refusal(text, code, expired):
    """Only a terminal explicit diagnostic plus exit 2 is a refusal, never a guess."""
    lines = text.splitlines()
    if code == 2 and not expired and lines and re.match(r"^(?:kv-tier-gate: )?REFUSED: .+", lines[-1]):
        return lines[-1]
    return None


def gpu_power_limits(path):
    """Retain all observed device/limit pairs; N/A is unknown, never zero or a default."""
    limits = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            item = dict((key, row.get(column)) for key, column in (
                ("device", "index"), ("power.limit", "power.limit [W]"),
                ("power.max_limit", "power.max_limit [W]")))
            if item["device"] is not None and item not in limits:
                limits.append(item)
    return limits


def descriptor(root, path):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}


UNPROVEN_STORAGE = "overlay/unproven — not NVMe, not spill speed"
M1_PROVEN_STORAGE = "M1 proof: physical local NVMe (nvme-local-direct); not measured spill speed"
M1_PROOF_TOOL = Path(__file__).resolve().parent.parent / "research/spill-f-20260919/m1-nvme-proof.py"
M1_PROOF_SCHEMA = "m1-nvme-proof-v1"


def m1_proof_binding(proof_path, root, out):
    """Admit NVMe only through a passing M1 proof whose identity is this filesystem's.

    The receipt is `m1-nvme-proof.py`'s public or private JSON. It must be a PASS of class
    nvme-local-direct with no reasons, produced by the exact proof tool in this checkout, and its
    identity triple (device, mount id, filesystem id or its hash) must equal the live identity of
    the storage root. The receipt is copied into the capture so validation can hash it.
    """
    raw = Path(proof_path).read_bytes()
    proof = json.loads(raw)
    require(proof.get("schema") == M1_PROOF_SCHEMA, "storage proof is not an M1 proof receipt")
    require(proof.get("verdict") == "PASS" and proof.get("class") == "nvme-local-direct"
            and proof.get("reasons") == [], "storage proof did not PASS")
    tool = hashlib.sha256(M1_PROOF_TOOL.read_bytes()).hexdigest()
    require(proof.get("tool_sha256") == tool, "storage proof was produced by a different proof tool")
    want = proof.get("A8_identity") or {}
    live = filesystem_identity(root)
    require(type(want.get("device")) is int and want.get("device") == live["device"]
            and want.get("mount_id") == live.get("mount_id"), "storage proof identity is not this mount")
    if "filesystem_id" in want:
        require(want["filesystem_id"] == live["filesystem_id"], "storage proof filesystem id differs")
    else:
        live_hash = hashlib.sha256(str(live["filesystem_id"]).encode()).hexdigest()[:16]
        require(want.get("filesystem_id_sha256_16") == live_hash, "storage proof filesystem id differs")
    copy = out / "STORAGE-PROOF.json"
    copy.write_bytes(raw)
    return {"receipt": descriptor(out, copy), "tool_sha256": tool, "identity": live,
            "proof_utc": proof.get("utc")}


def storage_command(command):
    """Resolve only literal argv and the canonical one-command bash wrapper.

    Never interpret arbitrary shell code. A storage-bench mention in an opaque
    shell wrapper fails closed instead of bypassing the storage-root guard.
    """
    argv = list(command)
    if Path(argv[0]).name == "env":
        argv = argv[1:]
        while argv and ("=" in argv[0] or argv[0] == "--" or argv[0] == "-u"):
            if argv[0] == "-u":
                require(len(argv) >= 3, "invalid env wrapper")
                argv = argv[2:]
            else:
                argv = argv[1:]
    require(argv, "empty command")
    if Path(argv[0]).name in {"bash", "sh"}:
        mentions_storage = any("storage-bench" in arg for arg in argv[1:])
        if not mentions_storage:
            return None
        require(len(argv) == 3 and argv[1] == "-c", "opaque storage shell wrapper")
        inner = shlex.split(argv[2])
        require(inner and shlex.join(inner) == argv[2], "noncanonical storage shell wrapper")
        # shlex.join quotes all shell metacharacters: the round-trip above means
        # substitutions, pipelines, redirects and env expansion cannot execute.
        argv = inner
    if Path(argv[0]).name != "storage-bench":
        require(not any("storage-bench" in arg for arg in argv), "opaque storage command")
        return None
    require(3 <= len(argv) <= 5 and argv[1] in {"roundtrip", "restore"}
            and (len(argv) < 5 or argv[4] in {"buffered", "uncached", "direct"}),
            "storage root binding requires exact roundtrip/restore argv")
    return argv


def filesystem_identity(path):
    """stat(2) device plus statfs filesystem id; mount id where Linux supplies it."""
    path = path.resolve(strict=True)
    st = path.stat()
    identity = {"device": st.st_dev, "filesystem_id": os.statvfs(path).f_fsid}
    fd = os.open(path, os.O_RDONLY)
    try:
        info = Path(f"/proc/self/fdinfo/{fd}")
        if info.exists():
            match = re.search(r"^mnt_id:\s*(\d+)$", info.read_text(), re.MULTILINE)
            require(match is not None, "missing filesystem mount id")
            identity["mount_id"] = int(match[1])
    finally:
        os.close(fd)
    return identity


def storage_binding(root, object_path):
    root = root.resolve(strict=True)
    obj = object_path.resolve()
    require(obj.is_relative_to(root), "storage object must be at or beneath supplied root")
    parent = obj.parent.resolve(strict=True)
    root_fs = filesystem_identity(root)
    object_fs = filesystem_identity(obj if obj.exists() else parent)
    require(root_fs == object_fs, "storage root and object filesystem differ")
    return {"root": str(root), "object": str(obj), "root_filesystem": root_fs,
            "object_filesystem": object_fs, "method": "stat-device+statfs-id+linux-mount-id"}


def verify_storage_binding(storage):
    binding = storage["object_binding"]
    actual = storage_binding(Path(storage["root"]), Path(binding["object"]))
    require(actual == binding, "storage object filesystem changed during execution")


def capture_storage(path, root, allow_unproven=False, object_path=None, proof_path=None):
    """Read-only ancestry, retaining failed commands verbatim; no hidden fallback.

    NVMe is admitted only through `proof_path` (an M1 proof receipt bound to this mount). An
    `nvmeXnY` name in lsblk is recorded as a hint, never as proof: an emulated or fabric NVMe has
    the same name, and a bind mount's findmnt source carries a `[subdir]` suffix lsblk refuses.
    """
    require(path.is_dir(), "storage root must exist")
    path = path.resolve()
    binding = storage_binding(path, object_path) if object_path is not None else None
    commands = [
        ("storage-findmnt", ["findmnt", "-J", "-T", str(path)]),
        ("storage-lsblk", ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS"]),
        ("storage-source", ["findmnt", "-n", "-o", "SOURCE", "-T", str(path)]),
    ]
    captures = []
    def capture(name, command):
        raw = root / (name + ".log")
        code, expired = tee_run(command, raw, timeout=20, echo=False)
        captures.append({"command": command, "exit_code": code, "timed_out": expired,
                         "raw_log": descriptor(root, raw)})
        return raw.read_text(errors="replace") if code == 0 and not expired else ""
    for name, command in commands:
        text = capture(name, command)
    device = re.sub(r"\[.*\]$", "", text.strip())
    ancestry = capture("storage-ancestry", ["lsblk", "-s", "-r", "-n", "-o", "KNAME", device])
    name_hint = device.startswith("/dev/") and any(re.fullmatch(
        r"nvme[0-9]+n[0-9]+(?:p[0-9]+)?", line.strip()) for line in ancestry.splitlines())
    m1, refusal = None, None
    if proof_path is not None:
        try:
            m1 = m1_proof_binding(proof_path, path, root)
        except (OSError, ValueError) as error:  # JSONDecodeError is a ValueError
            refusal = str(error)
    proven = m1 is not None
    record = {"class": "nvme-local-direct" if proven else "m1-proof-refused" if refusal
              else "nvme-name-only-unproven" if name_hint else "overlay-unproven",
              "nvme_proven": proven, "nvme_name_hint": name_hint,
              "label": M1_PROVEN_STORAGE if proven else UNPROVEN_STORAGE,
              "allow_unproven_storage": allow_unproven, "root": str(path),
              "commands": captures, "qualification": False}
    if m1 is not None:
        record["m1_proof"] = m1
    if refusal is not None:
        record["m1_proof_refusal"] = refusal
    if binding is not None:
        record["object_binding"] = binding
        record["object_argument"] = str(object_path)
    (root / "STORAGE.json").write_text(json.dumps(record, indent=2) + "\n")
    require(refusal is None, f"storage proof refused ({refusal}); retained STORAGE.json; a failing proof never downgrades to unproven")
    require(proven or allow_unproven, "NVMe ancestry unproven (no M1 proof); retained STORAGE.json; --allow-unproven-storage is development only")
    return record


def utc_timestamp(value):
    require(isinstance(value, str), "UTC timestamp must be a string")
    parsed = datetime.datetime.fromisoformat(value)
    require(parsed.utcoffset() == datetime.timedelta(0), "timestamp must include UTC offset")
    return parsed


def validate_capture(record, root):
    """Archive integrity, not byte equality or scoring. Empty diagnostic logs are valid.

    Legacy v1 captures lack wall UTC/seconds; their CELL journal supplies UTC and ns.
    Keep those original bytes unchanged. Missing measurements never become zeroes.
    """
    require(record["schema_version"] == 1 and record["kind"] == "subprocess-capture", "not a capture")
    require(record["qualification"] is False, "capture cannot qualify hardware")
    require(type(record["exit_code"]) is int and type(record["timed_out"]) is bool, "invalid command outcome")
    failed = record["exit_code"] != 0 or record["timed_out"] or record["parse_error"] is not None
    require(record["status"] in ({"failed", "refused"} if failed else {"executed-not-qualified"}), "command status mismatch")
    require(isinstance(record["command"], list) and record["command"] and all(isinstance(a, str) for a in record["command"]), "invalid command")
    begin, end = record["started_monotonic_ns"], record["ended_monotonic_ns"]
    require(type(begin) is int and type(end) is int and 0 <= begin <= end, "invalid capture clock")
    if {"elapsed_seconds", "started_utc", "ended_utc"} & record.keys():
        require({"elapsed_seconds", "started_utc", "ended_utc"} <= record.keys(), "partial capture timing fields")
        require(type(record["elapsed_seconds"]) in (int, float) and record["elapsed_seconds"] == (end-begin)/1e9, "capture elapsed mismatch")
        require(utc_timestamp(record["ended_utc"]) >= utc_timestamp(record["started_utc"]), "capture UTC regressed")
    if "lock_proof" in record:
        proof = json.loads(evidence(root, record["lock_proof"]).read_text())
        require(proof.get("seam") == active_seam(),
                f"lock proof seam={proof.get('seam')!r} is not this process's {LOCK_DIR_SEAM}={active_seam()!r}: a private-lock test capture never validates as a rig-locked cell")
        require(proof["rig"] in LOCKS and proof["rig"] != "cpu" and proof["lock"] == LOCKS[proof["rig"]]
                and proof["acquired"] is True and proof["owner"] == "collector"
                and proof["mechanism"] == "inherited-flock-same-open-description"
                and type(proof["device"]) is int and type(proof["inode"]) is int,
                "invalid inherited collector lock proof")
    text = evidence(root, record["raw_log"], allow_empty=True).read_text(errors="replace")
    if record["status"] == "refused":
        require(explicit_refusal(text, record["exit_code"], record["timed_out"]) == record["failure_quote"]
                and record["failure_quote"] is not None, "refusal not explicitly recorded")
    if failed:
        require(isinstance(record["failure_quote"], str) and (record["failure_quote"] in text or
                record["failure_quote"] == "died, cause unknown — repro needed"), "failure quote not in raw log")
    else:
        require(record["failure_quote"] is None, "successful capture has failure quote")
    telemetry = record["gpu_telemetry"]
    for key in ("raw_csv", "stderr"):
        if key in telemetry:
            evidence(root, telemetry[key], allow_empty=True)
    if "gpu_power_limits" in record:
        expected = gpu_power_limits(root / telemetry["raw_csv"]["path"]) if "raw_csv" in telemetry else []
        require(record["gpu_power_limits"] == expected, "power limits do not match raw CSV")
    for snapshot in record["compute_apps"].values():
        evidence(root, snapshot["raw_log"], allow_empty=True)
    storage = record.get("storage")
    if storage is not None:
        validate_storage_record(storage, root, record["command"])
    return record


def validate_storage_record(storage, root, command):
    """Offline integrity of a capture's storage section; NVMe only with its M1 proof binding."""
    require(storage["qualification"] is False, "storage ancestry is not qualification")
    if not storage["nvme_proven"]:
        require(storage["allow_unproven_storage"] is True and storage["label"] == UNPROVEN_STORAGE,
                "unproven storage lacks explicit opt-in/label")
    else:
        m1 = storage.get("m1_proof")
        require(storage["class"] == "nvme-local-direct" and storage["label"] == M1_PROVEN_STORAGE
                and isinstance(m1, dict), "NVMe label without an M1 proof binding")
        evidence(root, m1["receipt"])
        proof = json.loads((root / m1["receipt"]["path"]).read_text())
        require(proof.get("verdict") == "PASS" and proof.get("tool_sha256") == m1["tool_sha256"],
                "archived M1 proof does not match its binding")
    if "object_binding" in storage:
        binding = storage["object_binding"]
        command = storage_command(command)
        require(command is not None and command[2] == storage["object_argument"],
                "storage binding command mismatch")
        obj, parent = Path(binding["object"]), Path(storage["root"])
        require(obj.is_absolute() and parent.is_absolute()
                and obj.is_relative_to(parent) and binding["root"] == storage["root"],
                "storage binding path mismatch")
        require(binding["root_filesystem"] == binding["object_filesystem"]
                and {"device", "filesystem_id"} <= binding["root_filesystem"].keys()
                and all(type(v) is int for v in binding["root_filesystem"].values()),
                "storage binding filesystem mismatch")
    for capture in storage["commands"]:
        evidence(root, capture["raw_log"], allow_empty=True)


def validate_cell(path):
    rows, torn = read_cell_journal(path)
    require(not torn and len(rows) == 2 and [r["event"] for r in rows] == ["start", "end"],
            "interrupted/invalid CELL journal; not a completed capture")
    start, end = rows
    for key in ("kind", "run_id", "command", "started_utc", "qualification", "resume_from"):
        require(start[key] == end[key], "CELL start/end identity mismatch")
    require(start["kind"] == "CELL" and start["qualification"] is False, "not an unqualified CELL")
    require(utc_timestamp(end["ended_utc"]) >= utc_timestamp(start["started_utc"]), "CELL UTC regressed")
    root = path.parent
    capture = json.loads(evidence(root, end["capture"]).read_text())
    validate_capture(capture, root)
    require(capture["command"] == end["command"] and capture["exit_code"] == end["exit_code"] and
            capture["status"] == end["status"], "CELL capture outcome mismatch")
    if "gpu_power_limits" in capture:
        require(end.get("gpu_power_limits") == capture["gpu_power_limits"], "CELL power limit mismatch")
    if "lock_proof" in capture:
        require(start.get("lock_proof") == end.get("lock_proof") == capture["lock_proof"],
                "CELL inherited lock proof mismatch")
    duration = capture["ended_monotonic_ns"] - capture["started_monotonic_ns"]
    require(type(end["duration_ns"]) is int and end["duration_ns"] == duration, "CELL duration mismatch")
    if "elapsed_seconds" in capture:
        require(end["elapsed_seconds"] == capture["elapsed_seconds"], "CELL elapsed mismatch")
        require(end["started_utc"] == capture["started_utc"] and end["ended_utc"] == capture["ended_utc"], "CELL capture UTC mismatch")
        require(start.get("storage") == end.get("storage") == capture.get("storage"), "CELL storage mismatch")
    lock = json.loads((root / "lock.json").read_text())
    require(lock.get("seam") == active_seam(),
            f"lock.json seam={lock.get('seam')!r} is not this process's {LOCK_DIR_SEAM}={active_seam()!r}: a private-lock test capture never validates as a rig-locked cell")
    require(lock["rig"] in LOCKS and lock["rig"] != "cpu" and lock["lock"] == LOCKS[lock["rig"]]
            and lock["acquired"] is True, "missing/noncanonical collector lock")
    return {"kind": "capture-integrity", "status": capture["status"],
            "started_utc": start["started_utc"], "ended_utc": end["ended_utc"],
            "elapsed_seconds": duration/1e9, "qualification": False,
            "legacy_timing": "elapsed_seconds" not in capture,
            "storage_label": (capture.get("storage") or {}).get("label", "not recorded; no NVMe/spill-speed claim")}


def sample_fake(ns, moved):
    return {"schema_version": 1, "kind": "cpu-fixture", "monotonic_ns": ns, "interval_ms": 250,
            "devices": [{"device": d, "routes": {r: {"bytes_in": moved if r == "host" and d == 1 else 0, "bytes_out": moved if r == "host" and d == 0 else 0} for r in sorted(ROUTES)}, "clock_mhz": None, "power_w": None, "temperature_c": None, "vram_bytes": None} for d in range(2)],
            "host": {"pinned_bytes": 4096, "pageable_bytes": 8192},
            "nvme": {"queue_depth": 1, "read_bytes": moved, "write_bytes": 0, "physical_bytes": None},
            "wait_ns": {k: percentiles([10,20,30,40,50]) for k in ("io","h2d","d2h","p2p","queue")}}


class Sampler:
    """250ms collector, injected counter reader; unknown counters stay None.

    Real providers must read instrumented route counters, not infer bytes from PCIe labels.
    Dry-run uses virtual timestamps below; it cannot manufacture clocks/temperatures.
    """
    def __init__(self, read, clock=time.monotonic_ns):
        self.read, self.clock = read, clock
        self.samples = []
        self.stop = threading.Event()
        self.error = None

    def sample_at(self, ns):
        self.samples.append(self.read(ns))

    def run(self):
        deadline = self.clock()
        try:
            while not self.stop.is_set():
                self.sample_at(self.clock())
                deadline += INTERVAL_NS
                self.stop.wait(max(0, (deadline - self.clock()) / 1e9))
        except Exception as error:
            self.error = error
            self.stop.set()


def run_dry_campaign(out, n=5, rig="pro-pair", thermal="synthetic-no-thermal-measurement", echo=True):
    require(thermal == "synthetic-no-thermal-measurement", "dry-run cannot claim a real thermal regime")
    orders = paired_orders(n)
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir(); (out / "telemetry").mkdir()
    root = Path(__file__).resolve().parents[1]
    runner = root / "crates/memra-tier/tests/battery/fake_runner.py"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    identity = {"runtime_commit": revision, "binary_sha256": digest(runner),
                **{f"{name}_sha256": hashlib.sha256(("synthetic:" + name).encode()).hexdigest() for name in ("artifact","plan","layout","prompt")},
                "numeric_class": "opaque-fixture-no-executor", "context_tokens": 3, "requests": 1, "rig": "cpu", "kind": "cpu-fixture"}
    records, runs, failures = [], [], []
    with campaign_lock(rig) as lock:
        (out / "manifest.json").write_text(json.dumps({"schema_version":1,"kind":"cpu-fixture","status":"dry-run-not-qualification","source":revision,"runner_sha256":digest(runner),"collector_sha256":digest(Path(__file__)),"lock":lock,"lock_acquired":True,"seam":active_seam(),"rig_label_for_lock_only":rig,"thermal_regime":thermal,"AB_pairs":n,"BA_pairs":n,"sampler_interval_ms":250,"clock":"virtual-monotonic","telemetry_unknowns":"No measured GPU clocks/power/temperature or physical SSD bytes","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+"\n")
        def run(arm, pair, phase, order, fail=False):
            rid = f"{phase}-{pair}-{arm}"
            run_dir = out / rid; run_dir.mkdir()
            log = out / "raw" / f"{rid}.log"
            command = [sys.executable, str(runner), "--out", str(run_dir), "--arm", arm]
            if fail: command.append("--fail")
            capture = SubprocessRunner(smi=str(out / "no-gpu-in-cpu-fixture")).run(command, log, echo=echo)
            code, timeout = capture["exit_code"], capture["timed_out"]
            # The complete raw file exists and has been closed before parsing anything.
            text = log.read_text()
            if code != 0 or timeout:
                quote = next((line for line in text.splitlines() if line.startswith("ERROR:")), "died, cause unknown — repro needed")
                failure = {"schema_version":1,"kind":"cpu-fixture","run_id":rid,"command":command,"exit_code":code,"timed_out":timeout,"failure_quote":quote,"raw_log":descriptor(out,log),"concurrent_gpu_state":"not queried: CPU fake; no GPU invocation"}
                failures.append(failure)
                append_cell(out / "failures.jsonl", failure)
                require(fail, "runner failed; raw failure retained")
                return
            result = json.loads(next(line[len("RESULT "):] for line in text.splitlines() if line.startswith("RESULT ")))
            require(result["arm"] == arm, "forced arm not honored")
            require(type(result["duration_ns"]) is int and result["duration_ns"] > 0, "invalid duration")
            require(result["migrated_bytes"] == (3 if arm == "on" else 0), "forced route did not engage")
            outputs = {k: descriptor(out,run_dir / f"{k}.bin") for k in ("state","logits","tokens")}
            row = {"schema_version":1,"cell":"boundary","pair_id":pair,"run_id":rid,"arm":arm,**identity,"lock":None,"route":"host" if arm=="on" else "local","direct_path_proven":False,"migrated_bytes":result["migrated_bytes"],**outputs,"raw_log":descriptor(out,log),"telemetry":None,"telemetry_interval_ms":None,"status":"pass","exit_code":0}
            records.append(row)
            append_cell(out / "runs.jsonl", row)
            # Exercise the sampler schema with an explicitly VIRTUAL 250ms clock.
            sampler = Sampler(lambda ns: sample_fake(ns, min(1, ns//INTERVAL_NS)*result["migrated_bytes"]))
            for i in range(3): sampler.sample_at(i*INTERVAL_NS)
            samples = sampler.samples
            validate_telemetry(samples,"cpu-fixture")
            telemetry = out / "telemetry" / f"{rid}.jsonl"
            telemetry.write_text("".join(json.dumps(s,sort_keys=True)+"\n" for s in samples))
            runs.append({"schema_version":1,"kind":"cpu-fixture","phase":phase,"order":order,"pair_id":pair,"run_id":rid,"arm":arm,"thermal_regime":thermal,"duration_ns":result["duration_ns"],"clock":"virtual-monotonic","telemetry":descriptor(out,telemetry),"raw_log":descriptor(out,log),"command":command,**outputs})
            append_cell(out / "collector.jsonl", runs[-1])
        try:
            run("on","quoted-failure","red","red",True)
            for arm in ("off","on"): run(arm,"control","correctness","AB")
            validate_rows(records,out) # correctness BEFORE any performance sample
            for i,order in orders:
                pair=f"{order}-{i}"
                for letter in order: run("off" if letter=="A" else "on",pair,"performance",order)
            validate_rows(records,out)
            # Also bind every pair to the FIRST control, not just its adjacent partner.
            for row in records:
                require(all(row[k]==records[0][k] for k in IDENTITY), "campaign program changed")
                require(all(row[k]["sha256"]==records[0][k]["sha256"] for k in ("state","logits","tokens")), "campaign control changed")
        finally:
            (out / "runs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            (out / "collector.jsonl").write_text("".join(json.dumps(r)+"\n" for r in runs))
            (out / "failures.jsonl").write_text("".join(json.dumps(r)+"\n" for r in failures))
    summary = {"schema_version":1,"kind":"cpu-fixture","status":"dry-run-not-qualification","thermal_regime":thermal,"percentiles":"descriptive synthetic samples, not tail-confidence evidence","arms":{}}
    for arm in ("off","on"):
        values=[r["duration_ns"] for r in runs if r["phase"]=="performance" and r["arm"]==arm]
        summary["arms"][arm]={"N":len(values),"AB_N":n,"BA_N":n,"median_ns":statistics.median(values),**percentiles(values)}
    (out / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    validate_campaign(out)
    return summary

def validate_campaign(root):
    """Check retained order, controls, telemetry hashes/window and published synthetic N."""
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["kind"] == "cpu-fixture" and manifest["status"] == "dry-run-not-qualification", "live qualification needs native runner bindings")
    require(manifest.get("seam") == active_seam(),
            f"manifest seam={manifest.get('seam')!r} is not this process's {LOCK_DIR_SEAM}={active_seam()!r}: a private-lock dry run never validates as a rig-locked campaign")
    require(manifest["lock"] == LOCKS[manifest["rig_label_for_lock_only"]] and manifest["lock_acquired"] is True, "campaign lock metadata")
    n = manifest["AB_pairs"]
    require(n == manifest["BA_pairs"], "unbalanced order counts")
    orders = paired_orders(n)
    rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    validate_rows(rows, root)
    runs = [json.loads(line) for line in (root / "collector.jsonl").read_text().splitlines()]
    expected = [("correctness", "control", "AB", "off"), ("correctness", "control", "AB", "on")]
    expected += [("performance", f"{order}-{i}", order, "off" if arm == "A" else "on") for i,order in orders for arm in order]
    require([(r["phase"],r["pair_id"],r["order"],r["arm"]) for r in runs] == expected, "not interleaved AB/BA after correctness")
    require(len(runs) == len(rows), "collector/byte receipt cardinality")
    by_id = {r["run_id"]:r for r in rows}
    require(len({r["run_id"] for r in runs}) == len(runs), "duplicate collector run")
    for run in runs:
        require(run["schema_version"] == 1 and run["kind"] == "cpu-fixture", "collector version/class")
        row = by_id[run["run_id"]]
        require(all(row[k] == rows[0][k] for k in IDENTITY), "campaign program changed")
        require(run["thermal_regime"] == manifest["thermal_regime"] == "synthetic-no-thermal-measurement", "thermal regime changed/invented")
        require(run["arm"] == row["arm"] and run["pair_id"] == row["pair_id"], "collector arm mapping")
        require(run["raw_log"] == row["raw_log"], "collector raw log mapping")
        for key in ("state", "logits", "tokens"):
            require(run[key] == row[key] and run[key]["sha256"] == rows[0][key]["sha256"], "collector control mismatch")
        path = evidence(root,run["telemetry"])
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        validate_telemetry(samples, "cpu-fixture")
        for device, direction in ((0, "bytes_out"), (1, "bytes_in")):
            counters = [{d["device"]:d for d in sample["devices"]}[device]["routes"]["host"][direction] for sample in (samples[0], samples[-1])]
            require(counters[1] - counters[0] == row["migrated_bytes"], "synthetic telemetry/migration byte mismatch")
        require(run["clock"] == "virtual-monotonic" and samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"] >= run["duration_ns"], "telemetry does not cover run window")
    summary = json.loads((root / "summary.json").read_text())
    require(summary["status"] == "dry-run-not-qualification" and summary["kind"] == "cpu-fixture", "synthetic summary mislabeled")
    require(summary["thermal_regime"] == manifest["thermal_regime"], "summary thermal regime")
    for arm in ("off", "on"):
        values = [r["duration_ns"] for r in runs if r["phase"] == "performance" and r["arm"] == arm]
        require(summary["arms"][arm] == {"N":2*n,"AB_N":n,"BA_N":n,"median_ns":statistics.median(values),**percentiles(values)}, "summary median/N mismatch")
    failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()]
    require(len(failures) == 1 and failures[0]["exit_code"] == 9, "unexpected/missing dry red failure")
    for failure in failures:
        text = evidence(root,failure["raw_log"]).read_text()
        require(failure["failure_quote"] in text, "failure cause not captured verbatim")
    return len(runs)


def validate_schema(value, schema, path="row"):
    """Offline subset used by the two checked-in schemas; unknown keywords fail closed."""
    supported = {"$schema", "$id", "$comment", "title", "type", "properties", "required",
                 "additionalProperties", "items", "minItems", "minimum", "minLength", "pattern", "enum", "const", "anyOf"}
    require(set(schema) <= supported, "unsupported schema keyword")
    if "anyOf" in schema:
        for arm in schema["anyOf"]:
            try:
                validate_schema(value, arm, path)
                return
            except ValueError:
                pass
        raise ValueError(path + ": no schema alternative matched")
    if "type" in schema:
        matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
                   "string": isinstance(value, str), "boolean": type(value) is bool,
                   "integer": type(value) is int, "number": type(value) in (int, float) and math.isfinite(value),
                   "null": value is None}
        require(matches.get(schema["type"], False), path + ": wrong type")
    if "const" in schema:
        require(type(value) is type(schema["const"]) and value == schema["const"], path + ": wrong constant")
    if "enum" in schema:
        require(any(type(value) is type(v) and value == v for v in schema["enum"]), path + ": wrong enum")
    if isinstance(value, dict):
        require(set(schema.get("required", [])) <= set(value), path + ": missing field")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(props), path + ": unknown field")
        for key in value.keys() & props.keys():
            validate_schema(value[key], props[key], path + "." + key)
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), path + ": empty array")
        for item in value:
            validate_schema(item, schema["items"], path + "[]")
    if "minimum" in schema:
        require(value >= schema["minimum"], path + ": below minimum")
    if "minLength" in schema:
        require(len(value) >= schema["minLength"], path + ": empty string")
    if "pattern" in schema:
        require(re.fullmatch(schema["pattern"], value) is not None, path + ": pattern mismatch")


def join_storage(rows, samples):
    """A's unmodified StorageSample inside a run-id envelope, never guessed by order."""
    ids = {r["run_id"] for r in rows}
    require(len(ids) == len(rows), "duplicate run id")
    joined = {rid: [] for rid in ids}
    required = {"version", "fixture", "backend_requested", "backend_actual", "status",
                "valid_bytes", "padded_bytes", "io_bytes", "physical_bytes", "queue_ns", "io_ns",
                "h2d_ns", "d2h_ns", "p2p_ns", "total_ns", "inflight", "pinned_bytes",
                "pageable_bytes", "fallbacks", "payload_checksum"}
    optional = {"physical_bytes", "queue_ns", "io_ns", "h2d_ns", "d2h_ns", "p2p_ns", "pageable_bytes"}
    for envelope in samples:
        require(set(envelope) == {"run_id", "sample"}, "storage envelope needs explicit run_id/sample")
        rid, sample = envelope["run_id"], envelope["sample"]
        require(isinstance(rid, str) and rid in ids, "orphan storage run id")
        require(isinstance(sample, dict) and set(sample) == required, "StorageSample fields")
        require(type(sample["version"]) is int and sample["version"] == 1, "StorageSample version")
        for field in ("fixture", "backend_requested", "backend_actual", "status"):
            require(isinstance(sample[field], str) and bool(sample[field]), "StorageSample empty identity")
        for field in required - {"fixture", "backend_requested", "backend_actual", "status", "payload_checksum"}:
            value = sample[field]
            require((field in optional and value is None) or (type(value) is int and 0 <= value < 2**64), "StorageSample invalid counter")
        checksum = sample["payload_checksum"]
        require(isinstance(checksum, list) and len(checksum) == 32 and all(type(v) is int and 0 <= v <= 255 for v in checksum), "StorageSample checksum")
        require(sample["valid_bytes"] <= sample["padded_bytes"], "StorageSample invalid padding")
        joined[rid].append(sample)
    require(all(joined.values()), "missing storage samples for a run")
    # Preserve null physical counters, fallbacks and error statuses; joining is not scoring.
    return [{"run_id": r["run_id"], "samples": joined[r["run_id"]], "qualification": False} for r in rows]


def first_hour_plan():
    """Booking/order only. Never execute performance cells in the first rental hour."""
    stages = [("bootstrap-native-compile", 20, "bootstrap"), ("A-filesystem", 5, "lane-self-lock"),
              ("D1-local-bytes", 5, "collector"), ("C-ple-tiny", 15, "lane-self-lock"),
              ("B-qwen-fitting", 15, "lane-self-lock")]
    return {"schema_version": 1, "kind": "plan", "rig": "rtx5090", "qualification": False,
            "lock": LOCKS["rtx5090"], "stop_starting_after_minutes": 60,
            "stages": [{"cell": name, "budget_minutes": minutes, "lock_owner": owner,
                        "status": "planned-not-run"} for name, minutes, owner in stages],
            "correctness_orders": [{"pair_id": order + "-correctness", "order": order,
                                    "arms": ["off" if arm == "A" else "on" for arm in order]}
                                   for order in ("AB", "BA")],
            "order_applies_to": "future bound tier ON/OFF cells only; baseline runners have no invented arm flags",
            "performance_cells": [], "performance_medians_allowed": False,
            "later_performance_minimum": {"AB_pairs": 5, "BA_pairs": 5},
            "warning": "Cold native build may consume entire hour. Missing adapters remain BLOCKED."}


def read_cell_journal(path):
    """Recover only a torn final append; never hide corruption of a complete row."""
    lines = path.read_text().splitlines(keepends=True)
    records, torn = [], False
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            require(i == len(lines)-1 and not line.endswith("\n"), "corrupt completed CELL journal row")
            torn = True
    require(records, "no complete CELL receipt to resume")
    return records, torn


def append_cell(path, row):
    with path.open("a") as out:
        out.write(json.dumps(row) + "\n"); out.flush(); os.fsync(out.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", nargs=argparse.REMAINDER, metavar="ARGV")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--first-hour", action="store_true")
    modes.add_argument("--validate", type=Path, metavar="RECEIPT", help="byte/telemetry JSONL, CELL.jsonl, capture JSON, or directory of CELL journals; integrity is not qualification")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-campaign", type=Path, metavar="BUNDLE")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--external-lock", action="store_true", help="inherit canonical lock FD; replace exactly one @COLLECTOR_LOCK_FD@ child argument (explicit opt-in only)")
    parser.add_argument(PRIVATE_LOCK_FLAG, action="store_true", help=f"required with {LOCK_DIR_SEAM} for --execute and --dry-run: a test's explicit statement that the private lock directory is intended; an inherited environment variable alone never moves a campaign off the canonical rig lock")
    parser.add_argument("--schema", choices=["auto", "runs", "telemetry", "storage-cell"], default="auto")
    parser.add_argument("--storage-root", type=Path, help="actual filesystem path for this storage cell; ancestry captured before execution")
    parser.add_argument("--allow-unproven-storage", action="store_true", help="explicit overlay/unproven development mode; never NVMe/spill-speed evidence")
    parser.add_argument("--storage-proof", type=Path, help="M1 proof receipt (m1-nvme-proof.py PASS) bound to --storage-root; the only way a root is labelled NVMe")
    parser.add_argument("--storage-samples", type=Path, help="run-id wrapped canonical StorageSample JSONL")
    parser.add_argument("--resume", action="store_true", help="read last CELL receipt; rerun in a new attempt")
    parser.add_argument("--run-id", help="stable cell identity (defaults to output directory name)")
    parser.add_argument("--hourly-cost", type=float, help="private optional operator rate")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--pairs-per-order", type=int, default=5)
    parser.add_argument("--rig", choices=["rtx5090", "pro-single", "pro-pair", "pro-four"], default="pro-pair")
    # argparse REMAINDER still treats a literal -- as its own option terminator.
    # Split before parsing so every child byte/argument (including --) survives.
    argv = sys.argv[1:]
    execute = argv.index("--execute") if "--execute" in argv else None
    args = parser.parse_args(argv if execute is None else argv[:execute + 1])
    if execute is not None:
        args.execute = argv[execute + 1:]
    seam = active_seam()
    if seam is not None:
        print(f"tier-battery: {LOCK_DIR_SEAM}={seam}: PRIVATE lock directory (test seam); the rig lock is NOT held by this process", file=sys.stderr)
    require(not args.private_lock_dir_for_tests or seam is not None,
            f"{PRIVATE_LOCK_FLAG} without {LOCK_DIR_SEAM}: the flag only accompanies the test seam")
    require(seam is None or args.private_lock_dir_for_tests or (args.execute is None and not args.dry_run),
            f"{LOCK_DIR_SEAM} is set but {PRIVATE_LOCK_FLAG} was not passed: an inherited environment variable alone never moves a campaign off the canonical rig lock; unset it, or pass the flag from a test")
    if args.execute is not None:
        require(args.execute and args.out is not None and args.timeout > 0, "--execute requires argv, new --out and positive timeout")
        if args.hourly_cost is not None:
            require(math.isfinite(args.hourly_cost) and args.hourly_cost >= 0, "invalid hourly cost")
        previous = None
        if args.resume:
            old = args.out / "CELL.jsonl"
            records, torn = read_cell_journal(old)
            require(records and records[-1]["command"] == args.execute, "resume command mismatch")
            previous = {"receipt": str(old), "sha256": digest(old), "last_event": records[-1]["event"], "torn_tail_preserved": torn}
            args.out = args.out / "attempts" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        require(not args.allow_unproven_storage or args.storage_root is not None,
                "--allow-unproven-storage requires --storage-root")
        require(args.storage_proof is None or args.storage_root is not None,
                "--storage-proof requires --storage-root")
        storage_argv = storage_command(args.execute)
        require(storage_argv is None or args.storage_root is not None,
                "storage-bench requires --storage-root; overlay needs --allow-unproven-storage")
        args.out.mkdir(parents=True, exist_ok=False)
        storage = capture_storage(args.storage_root, args.out, args.allow_unproven_storage,
                                  Path(storage_argv[2]) if storage_argv else None,
                                  args.storage_proof) if args.storage_root else None
        token = "@COLLECTOR_LOCK_FD@"
        require(not args.external_lock or args.execute.count(token) == 1,
                "--external-lock requires exactly one @COLLECTOR_LOCK_FD@ argument")
        require(not args.external_lock or not args.resume,
                "external-lock FD argv is ephemeral; use a fresh cell instead of --resume")
        with campaign_lock(args.rig, inherit=args.external_lock) as lock:
            proof = {"rig": args.rig, "lock": LOCKS[args.rig], "acquired": True}
            if seam is not None:
                proof["seam"] = seam
            pass_fds = ()
            if args.external_lock:
                pass_fds = (lock.fileno(),)
                stat = os.fstat(lock.fileno())
                proof.update(owner="collector", mechanism="inherited-flock-same-open-description",
                             device=stat.st_dev, inode=stat.st_ino)
                args.execute = [str(lock.fileno()) if arg == token else arg for arg in args.execute]
            lock_path = args.out / "lock.json"
            lock_path.write_text(json.dumps(proof) + "\n")
            record = SubprocessRunner().run(args.execute, args.out / "command.log", args.timeout,
                                            run_id=args.run_id or args.out.name, hourly_cost=args.hourly_cost, resume_from=previous, storage=storage,
                                            pass_fds=pass_fds, lock_proof=descriptor(args.out, lock_path) if args.external_lock else None)
        print(json.dumps(record, indent=2))
        if record["status"] in {"failed", "refused"}:
            sys.exit(record["exit_code"] if 0 < record["exit_code"] < 126 else 2)
        return
    require(not args.external_lock, "--external-lock requires --execute")
    if args.first_hour:
        print(json.dumps(first_hour_plan(), indent=2))
        return
    require(not args.resume, "--resume requires --execute")
    if args.validate_campaign:
        count = validate_campaign(args.validate_campaign)
        print(f"CPU CAMPAIGN MATCH: {count} runs; synthetic protocol evidence only, NOT GPU qualification")
        return
    if args.dry_run:
        require(args.out is not None, "--dry-run requires a new --out directory")
        print(json.dumps(run_dry_campaign(args.out, args.pairs_per_order, args.rig), indent=2))
        return
    if args.plan:
        print(json.dumps({"schema_version": 1, "status": "pending-gpu-adapters", "cases": CASES, "standard_gates": ["kernel-check", "run-gen argmax", "run-spec K=1..8 / manifest refusals", "step-pro"], "locks": LOCKS, "performance": {"AB_pairs": 5, "BA_pairs": 5, "telemetry_interval_ms": 250}, "warning": "Scaffold only. No GPU cell executed; pair is four tiers, not four-card qualification."}, indent=2))
        return
    if args.validate.is_dir():
        require(args.schema == "auto" and args.storage_samples is None, "directory capture validation cannot join storage or validate byte schemas")
        paths = sorted(args.validate.rglob("CELL.jsonl"))
        require(paths, "no CELL journals in receipt directory")
        results = [{"cell": str(path.parent.relative_to(args.validate)), **validate_cell(path)} for path in paths]
        print(json.dumps({"kind": "capture-integrity", "cells": len(results),
                          "failed_commands": sum(r["status"] == "failed" for r in results),
                          "refused_commands": sum(r["status"] == "refused" for r in results),
                          "qualification": False, "results": results}, indent=2))
        return
    if args.schema == "auto" and args.validate.name == "CELL.jsonl":
        require(args.storage_samples is None, "capture integrity is not a storage/byte-receipt join")
        print(json.dumps(validate_cell(args.validate), indent=2))
        return
    if args.schema == "auto" and args.validate.name.endswith(".capture.json"):
        require(args.storage_samples is None, "capture integrity is not a storage/byte-receipt join")
        record = json.loads(args.validate.read_text())
        validate_capture(record, args.validate.parent)
        print("CAPTURE INTEGRITY MATCH; command status=" + record["status"] + "; NOT qualification")
        return
    rows = [json.loads(line) for line in args.validate.read_text().splitlines() if line.strip()]
    require(rows, "empty JSONL")
    if args.schema == "storage-cell":
        # Keep A's diagnostic sample join behind this explicit schema. First run
        # current capture integrity (including UTC, storage and inherited locks),
        # then A's exact command/raw-sample/run-id binding. Never auto-promote it.
        import runpy
        validate_cell(args.validate)
        module = Path(__file__).resolve().parents[1] / "research/spill-a-20260919/storage_capture.py"
        envelopes = None if args.storage_samples is None else [
            json.loads(line) for line in args.storage_samples.read_text().splitlines() if line.strip()]
        joined = runpy.run_path(str(module))["validate_storage_cell"](args.validate, sys.modules[__name__], envelopes)
        require(args.out is not None, "storage CELL join requires new --out JSONL file")
        with args.out.open("x") as out:
            out.write(json.dumps(joined) + "\n")
        print("STORAGE-CAPTURE MATCH: 1 run; diagnostic join only, NOT hardware/serving qualification")
        return
    kind = args.schema if args.schema != "auto" else ("telemetry" if "monotonic_ns" in rows[0] else "runs")
    schema = json.loads((Path(__file__).resolve().parents[1] / "research/spill-d-20260919" / (kind + ".schema.json")).read_text())
    for row in rows:
        validate_schema(row, schema)
    if kind == "telemetry":
        require(args.storage_samples is None, "telemetry has no run_id; join storage against runs instead")
        validate_telemetry(rows, rows[0]["kind"])
        print(f"TELEMETRY MATCH: {len(rows)} samples; NOT hardware qualification")
        return
    pairs = validate_rows(rows, args.validate.parent)
    if args.storage_samples:
        samples = [json.loads(line) for line in args.storage_samples.read_text().splitlines() if line.strip()]
        joined = join_storage(rows, samples)
        require(args.out is not None, "storage join requires new --out JSONL file")
        with args.out.open("x") as out:
            for row in joined:
                out.write(json.dumps(row) + "\n")
    print(f"BYTE-RECEIPTS MATCH: {len(rows)} records / {pairs} forced pairs; not full battery qualification")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
