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
import datetime
import fcntl
import math
import os
import signal
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


@contextlib.contextmanager
def campaign_lock(rig):
    require(rig in LOCKS and LOCKS[rig] is not None, "GPU-shaped campaign requires canonical rig lock")
    # Never unlink a shared lock inode: waiters and other campaigns must see the same file.
    with open(LOCKS[rig], "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield LOCKS[rig]
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def paired_orders(n):
    require(type(n) is int and n >= 5, "at least five AB and five BA pairs required")
    return [(i, order) for i in range(n) for order in ("AB", "BA")]


def tee_run(command, raw_path, timeout=30, echo=True):
    """Drain merged stdout/stderr to a raw file BEFORE parsing, including on timeout."""
    with raw_path.open("xb") as log:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
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
                os.killpg(p.pid, signal.SIGKILL)
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
        if errors:
            raise errors[0]
    return code, timed_out


def descriptor(root, path):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}


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
        (out / "manifest.json").write_text(json.dumps({"schema_version":1,"kind":"cpu-fixture","status":"dry-run-not-qualification","source":revision,"runner_sha256":digest(runner),"collector_sha256":digest(Path(__file__)),"lock":lock,"lock_acquired":True,"rig_label_for_lock_only":rig,"thermal_regime":thermal,"AB_pairs":n,"BA_pairs":n,"sampler_interval_ms":250,"clock":"virtual-monotonic","telemetry_unknowns":"No measured GPU clocks/power/temperature or physical SSD bytes","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+"\n")
        def run(arm, pair, phase, order, fail=False):
            rid = f"{phase}-{pair}-{arm}"
            run_dir = out / rid; run_dir.mkdir()
            log = out / "raw" / f"{rid}.log"
            command = [sys.executable, str(runner), "--out", str(run_dir), "--arm", arm]
            if fail: command.append("--fail")
            code, timeout = tee_run(command, log, echo=echo)
            # The complete raw file exists and has been closed before parsing anything.
            text = log.read_text()
            if code != 0 or timeout:
                quote = next((line for line in text.splitlines() if line.startswith("ERROR:")), "died, cause unknown — repro needed")
                failure = {"schema_version":1,"kind":"cpu-fixture","run_id":rid,"command":command,"exit_code":code,"timed_out":timeout,"failure_quote":quote,"raw_log":descriptor(out,log),"concurrent_gpu_state":"not queried: CPU fake; no GPU invocation"}
                failures.append(failure)
                require(fail, "runner failed; raw failure retained")
                return
            result = json.loads(next(line[len("RESULT "):] for line in text.splitlines() if line.startswith("RESULT ")))
            require(result["arm"] == arm, "forced arm not honored")
            require(type(result["duration_ns"]) is int and result["duration_ns"] > 0, "invalid duration")
            require(result["migrated_bytes"] == (3 if arm == "on" else 0), "forced route did not engage")
            outputs = {k: descriptor(out,run_dir / f"{k}.bin") for k in ("state","logits","tokens")}
            row = {"schema_version":1,"cell":"boundary","pair_id":pair,"run_id":rid,"arm":arm,**identity,"lock":None,"route":"host" if arm=="on" else "local","direct_path_proven":False,"migrated_bytes":result["migrated_bytes"],**outputs,"raw_log":descriptor(out,log),"telemetry":None,"telemetry_interval_ms":None,"status":"pass","exit_code":0}
            records.append(row)
            # Exercise the sampler schema with an explicitly VIRTUAL 250ms clock.
            sampler = Sampler(lambda ns: sample_fake(ns, ns//INTERVAL_NS*result["migrated_bytes"]))
            for i in range(3): sampler.sample_at(i*INTERVAL_NS)
            samples = sampler.samples
            validate_telemetry(samples,"cpu-fixture")
            telemetry = out / "telemetry" / f"{rid}.jsonl"
            telemetry.write_text("".join(json.dumps(s,sort_keys=True)+"\n" for s in samples))
            runs.append({"schema_version":1,"kind":"cpu-fixture","phase":phase,"order":order,"pair_id":pair,"run_id":rid,"arm":arm,"thermal_regime":thermal,"duration_ns":result["duration_ns"],"clock":"virtual-monotonic","telemetry":descriptor(out,telemetry),"raw_log":descriptor(out,log),"command":command,**outputs})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate", type=Path, metavar="RUNS_JSONL")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-campaign", type=Path, metavar="BUNDLE")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pairs-per-order", type=int, default=5)
    parser.add_argument("--rig", choices=["rtx5090", "pro-pair", "pro-four"], default="pro-pair")
    args = parser.parse_args()
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
    rows = [json.loads(line) for line in args.validate.read_text().splitlines() if line.strip()]
    pairs = validate_rows(rows, args.validate.parent)
    print(f"BYTE-RECEIPTS MATCH: {len(rows)} records / {pairs} forced pairs; not full battery qualification")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
