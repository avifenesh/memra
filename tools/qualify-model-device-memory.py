#!/usr/bin/env python3
"""Build/run issue #544's native seam gates under coordinator-owned GPU locks.

No rental, lock creation/acquisition, checkpoint download, or serving-instance access.
The build phase runs no tests and hides GPUs. Each run is one exact native test.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "same-device": ("engine", "model_memory::native_tests::glm_same_ordinal_owners_are_not_lost", 1),
    "pair": ("engine", "model_memory::native_tests::glm_peer_admission_materialization_trim_and_refill", 2),
    "worker": ("server", "worker::tests::native_glm_peer_admission_trim_preserves_lease", 2),
}
CAPABILITIES = {"89": "8.9", "90a": "9.0", "100a": "10.0", "120a": "12.0"}


def capture(argv, **kwargs):
    return subprocess.check_output(argv, cwd=ROOT, text=True, timeout=30, **kwargs).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_identity(expected):
    head = capture(["git", "rev-parse", "HEAD"])
    if head != expected or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError(f"expected source {expected}, found {head}")
    if capture(["git", "status", "--porcelain", "--untracked-files=normal"]):
        raise ValueError("qualification requires a clean committed checkout; put receipts under target/")
    return head


def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_") and k != "DOCS_RS"}
    return env


def native_build_env(arch, target, nvcc):
    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MEMRA_CUDA_ARCH"] = arch
    # A clean environment does not invalidate DOCS_RS build-script outputs cached by Cargo.
    # Each new receipt owns a new target tree; inherited/default artifacts cannot be reused.
    env["CARGO_TARGET_DIR"] = str(target)
    if nvcc:
        env["MEMRA_NVCC"] = str(nvcc.resolve())
    return env


def run_logged(argv, path, env, timeout):
    # Bank stdout AND stderr before parsing any verdict, including failures/timeouts.
    with Path(path).open("w") as log:
        try:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            log.write(f"\nQUALIFICATION TIMEOUT after {timeout}s\n")
            raise
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {path}")


def build(args):
    head = source_identity(args.expected_sha)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out / "cargo-target"
    env = native_build_env(args.arch, target, args.nvcc)
    # Explicit arch prevents build.rs GPU auto-detection; --no-run prevents CUDA tests.
    receipt = {"source_sha": head, "arch": args.arch, "cuda_visible_devices": "",
               "cargo_target_dir": str(target), "rustc": capture(["rustc", "--version"]), "binaries": {}, "commands": []}
    write_json(out / "build.pending.json", receipt)
    for key, crate in [("engine", "memra-engine"), ("server", "memra-server")]:
        argv = ["cargo", "test", "--release", "--locked", "--no-run", "--lib", "-p", crate,
                "--jobs", str(args.jobs), "--message-format=json"]
        receipt["commands"].append(argv)
        run_logged(argv, out / f"build-{key}.log", env, args.timeout)
        artifacts = []
        for line in (out / f"build-{key}.log").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (row.get("reason") == "compiler-artifact" and row.get("executable")
                    and row.get("profile", {}).get("test")
                    and row.get("target", {}).get("name") == crate.replace("-", "_")):
                artifacts.append(Path(row["executable"]).resolve())
        if len(artifacts) != 1:
            raise ValueError(f"expected exactly one {crate} test executable, found {artifacts}")
        receipt["binaries"][key] = {"path": str(artifacts[0]), "sha256": sha256(artifacts[0])}
    source_identity(head)
    write_json(out / "build.json", receipt)
    (out / "build.pending.json").unlink()
    print(f"BUILT {out / 'build.json'}; no GPU execution performed")


def ancestors():
    result = set()
    pid = os.getpid()
    while pid > 0 and pid not in result:
        result.add(pid)
        status = Path(f"/proc/{pid}/status").read_text()
        pid = int(re.search(r"^PPid:\s+(\d+)$", status, re.MULTILINE)[1])
    return result


def exclusive_lock_held(lock_stat, caller_pids, locks_text):
    # Linux /proc/locks: id FLOCK ADVISORY WRITE pid major:minor:inode start end.
    # Ignore blocked waiter rows ("->") and read/POSIX locks. Never acquire a lock here.
    for line in locks_text.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        major, minor, inode = fields[5].split(":")
        if (int(fields[4]) in caller_pids and int(major, 16) == os.major(lock_stat.st_dev)
                and int(minor, 16) == os.minor(lock_stat.st_dev)
                and int(inode) == lock_stat.st_ino):
            return True
    return False


def lock_mapping(gpus, specs, caller_pids, locks_text):
    if len(set(gpus)) != len(gpus) or any(not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", g) for g in gpus):
        raise ValueError("use distinct full GPU UUIDs, not ordinals or MIG instances")
    mapping = {}
    for spec in specs:
        uuid, separator, name = spec.partition("=")
        if not separator or uuid in mapping or not Path(name).is_absolute():
            raise ValueError("--lock-file must be a unique GPU-UUID=/absolute/canonical/lock mapping")
        mapping[uuid] = Path(name)
    if set(mapping) != set(gpus):
        raise ValueError("lock mappings must exactly match this stage's physical GPU set")
    if len({path.resolve() for path in mapping.values()}) != len(gpus):
        raise ValueError("each physical GPU must have its own canonical lock file")
    stats = {uuid: path.stat() for uuid, path in mapping.items()}
    if len({(stat.st_dev, stat.st_ino) for stat in stats.values()}) != len(gpus):
        raise ValueError("per-card locks must have distinct inodes")
    for uuid, path in mapping.items():
        if not exclusive_lock_held(stats[uuid], caller_pids, locks_text):
            raise ValueError(f"no exclusive flock held by this process ancestry for {uuid}: {path}")
    return {uuid: str(mapping[uuid]) for uuid in gpus}


def lease_locks(gpus, lease, caller_pids):
    if lease.get("requested_uuids") != gpus or lease.get("lock_order") != sorted(gpus):
        raise ValueError("lease GPU order/set does not match requested UUIDs and stable lock order")
    wrapper = lease.get("wrapper_pid")
    child = lease.get("child_pid")
    if (not isinstance(wrapper, int) or not isinstance(child, int)
            or wrapper not in caller_pids or child not in caller_pids or wrapper == child):
        raise ValueError("lease wrapper/child are not the live caller process ancestry")
    files = {uuid: f"/tmp/memra-gpu-locks/{uuid}.lock" for uuid in gpus}
    if lease.get("lock_files") != files:
        raise ValueError("lease does not name the canonical per-UUID lock files")
    return [f"{uuid}={files[uuid]}" for uuid in gpus], {wrapper}


def test_verdict(text):
    if not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", text):
        raise ValueError("exact native test did not report one executed pass")
    if re.search(r"\bSKIP(?:PED)?\b", text):
        raise ValueError("native qualification output contains a skip")


def execute(args):
    head = source_identity(args.expected_sha)
    key, name, count = STAGES[args.stage]
    if sys.platform != "linux":
        raise ValueError("native execution requires Linux and the coordinator's per-card flock wrapper")
    if len(args.gpu_uuid) != count:
        raise ValueError(f"stage {args.stage} uses exactly {count} physical GPU(s)")
    visible = ",".join(args.gpu_uuid)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise ValueError("wrapper CUDA_VISIBLE_DEVICES must exactly equal the ordered --gpu-uuid list")
    lease_path = os.environ.get("MEMRA_GPU_LEASE_FILE")
    if not lease_path:
        raise ValueError("MEMRA_GPU_LEASE_FILE is absent; invoke through memra-gpu-run")
    lease = json.loads(Path(lease_path).read_text())
    specs, holders = lease_locks(args.gpu_uuid, lease, ancestors())
    if args.lock_file and set(args.lock_file) != set(specs):
        raise ValueError("explicit lock mappings disagree with the wrapper lease")
    locks = lock_mapping(args.gpu_uuid, specs, holders, Path("/proc/locks").read_text())
    out = args.out.resolve()
    built = json.loads((out / "build.json").read_text())
    if built["source_sha"] != head:
        raise ValueError("source does not match build receipt")
    binary = built["binaries"][key]
    if sha256(binary["path"]) != binary["sha256"]:
        raise ValueError("test binary hash changed after build")
    target = out / args.stage
    target.mkdir(exist_ok=False)
    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    receipt = {"source_sha": head, "stage": args.stage, "test": name,
               "binary": binary, "arch": built["arch"], "gpu_uuids": args.gpu_uuid,
               "lock_files": locks, "lease": {key: lease[key] for key in
                   ["wrapper_pid", "child_pid", "requested_uuids", "lock_order", "lock_files"]},
               "status": "running", "started_unix": time.time()}
    write_json(target / "receipt.json", receipt)
    telemetry = None
    try:
        gpu_csv = capture(["nvidia-smi", "-i", visible,
            "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version,power.limit",
            "--format=csv,noheader,nounits"])
        (target / "devices.csv").write_text(gpu_csv + "\n")
        # Inventory only: other locked cards may have independent jobs. Do not require
        # an idle host, and do not execute work on cards outside this stage's lease.
        try:
            topology = capture(["nvidia-smi", "topo", "-m"])
            (target / "topology.txt").write_text(topology + "\n")
            activity = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader,nounits"])
            (target / "host-processes-before.csv").write_text(activity + "\n")
        except (OSError, subprocess.SubprocessError) as error:
            receipt["host_inventory_error"] = str(error)
        rows = list(csv.reader(io.StringIO(gpu_csv), skipinitialspace=True))
        if (len(rows) != count or {r[1] for r in rows} != set(args.gpu_uuid)
                or any(r[4] != CAPABILITIES[built["arch"]] for r in rows)):
            raise ValueError("physical GPU identities/capabilities do not match this build")
        processes = capture(["nvidia-smi", "-i", visible,
            "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        (target / "processes-before.csv").write_text(processes + "\n")
        if processes:
            raise ValueError("selected GPU set already has a compute process; do not disturb it")
        listing = capture([binary["path"], "--list", "--ignored", "--exact", name], env=env)
        (target / "test-list.txt").write_text(listing + "\n")
        if f"{name}: test" not in listing or "1 test, 0 benchmarks" not in listing:
            raise ValueError("named ignored native test is absent; refusing a vacuous run")
        with (target / "telemetry.csv").open("w") as samples:
            telemetry = subprocess.Popen(["nvidia-smi", "-i", visible,
                "--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,nounits", "--loop-ms=250"], stdout=samples, stderr=subprocess.STDOUT)
            argv = [binary["path"], "--exact", name, "--ignored", "--test-threads=1", "--nocapture"]
            receipt["command"] = argv
            run_logged(argv, target / "test.log", env, args.timeout)
            test_verdict((target / "test.log").read_text())
            if telemetry.poll() is not None:
                raise ValueError("250 ms telemetry process stopped before the test completed")
        source_identity(head)
        if sha256(binary["path"]) != binary["sha256"]:
            raise ValueError("test binary changed during execution")
        final_lease = json.loads(Path(lease_path).read_text())
        specs, holders = lease_locks(args.gpu_uuid, final_lease, ancestors())
        lock_mapping(args.gpu_uuid, specs, holders, Path("/proc/locks").read_text())
        receipt["status"] = "passed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        if telemetry:
            try:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait(timeout=10)
            except (OSError, subprocess.SubprocessError) as error:
                receipt["status"] = "failed"
                receipt["telemetry_cleanup_error"] = str(error)
        try:
            after = capture(["nvidia-smi", "-i", visible,
                "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
            (target / "processes-after.csv").write_text(after + "\n")
        except (OSError, subprocess.SubprocessError) as error:
            receipt["status"] = "failed"
            receipt["post_process_query_error"] = str(error)
        receipt["finished_unix"] = time.time()
        receipt["files"] = {p.name: sha256(p) for p in target.iterdir() if p.name != "receipt.json"}
        write_json(target / "receipt.json", receipt)
    if receipt["status"] != "passed":
        raise RuntimeError(f"qualification cleanup/inventory failed; see {target / 'receipt.json'}")
    print(f"PASS {args.stage}: {target / 'receipt.json'} (synthetic native seam only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    build_parser = commands.add_parser("build", help="GPU-free native compilation; no test execution")
    run_parser = commands.add_parser("run", help="one native test, only inside provided per-card locks")
    for sub in [build_parser, run_parser]:
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--expected-sha", required=True)
        sub.add_argument("--timeout", type=int, default=1800)
    build_parser.add_argument("--arch", choices=CAPABILITIES, required=True)
    build_parser.add_argument("--nvcc", type=Path)
    build_parser.add_argument("--jobs", type=int, default=8)
    run_parser.add_argument("--stage", choices=STAGES, required=True)
    run_parser.add_argument("--gpu-uuid", action="append", required=True)
    run_parser.add_argument("--lock-file", action="append", default=[],
                            help="optional cross-check of coordinator lease paths")
    args = parser.parse_args()
    if args.timeout <= 0 or (args.mode == "build" and args.jobs <= 0):
        parser.error("timeout and jobs must be positive")
    try:
        (build if args.mode == "build" else execute)(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"REFUSED/FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
