#!/usr/bin/env python3
"""Build, capture, seal and bank native release evidence; never rent or acquire GPUs.

Capture must run through the coordinator's memra-gpu-run physical-card wrapper.
Seal runs after that wrapper exits, so its final cleanup status is part of the proof.
"""
from __future__ import annotations
import argparse
import atexit
import csv
import io
import json
import os
import platform
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tempfile

# Never execute stale project bytecode from an ignored __pycache__ directory.
_PRIVATE_PY_CACHE = tempfile.TemporaryDirectory(prefix="memra-native-proof-python-")
atexit.register(_PRIVATE_PY_CACHE.cleanup)
sys.pycache_prefix = _PRIVATE_PY_CACHE.name
sys.dont_write_bytecode = True

import release_qualification as q
import release_inputs
import release_input_view


def write(path, value):
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def reference(root, path):
    return {"path": path, "sha256": q.digest((root / path).read_bytes())}


def clean_source(repo):
    q.require(not q.git(repo, "status", "--porcelain", "--untracked-files=normal").strip(),
              "native qualification requires clean committed source")
    release_inputs.verify_checkout(repo)
    return q.source_snapshot(repo)


def environment():
    # Every ambient Memra switch is removed. This runner qualifies the ordinary release
    # battery, not an inherited benchmark/alternate draft program. No secret values banked.
    q.require(not os.environ.get("LD_PRELOAD"), "LD_PRELOAD is unsupported for native qualification")
    return {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_") and k not in
            ("DOCS_RS", "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")}


def build_environment(out, nvcc, rustc):
    # Admit basic process/network settings only. CUDA/GCC/Clang option injection,
    # compiler wrappers and include/library search overrides must not reach Cargo.
    admitted = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in environment().items() if key in admitted}
    env.update(CUDA_VISIBLE_DEVICES="", MEMRA_CUDA_ARCH="120a", MEMRA_NVCC=str(nvcc),
               CUDA_HOME=str(nvcc.parent.parent), CUDA_PATH=str(nvcc.parent.parent),
               CARGO_HOME=str(out / "cargo-home"), CARGO_TARGET_DIR=str(out / "target"),
               RUSTUP_TOOLCHAIN="1.97.1", RUSTC=str(rustc),
               PYTHONPYCACHEPREFIX=str(out / "python-cache"), PYTHONDONTWRITEBYTECODE="1")
    return env


def prepare_cargo_home(out):
    home = out / "cargo-home"
    home.mkdir()
    original = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).resolve()
    # Reuse only downloadable archive/index caches. Never inherit config,
    # credentials, compiler wrappers or mutable extracted registry source trees.
    for name in ("registry/cache", "registry/index"):
        source = original / name
        if source.is_dir():
            shutil.copytree(source, home / name)
    return home


def numeric_environment(env):
    return {k: q.digest(v.encode()) for k, v in sorted(env.items()) if k.startswith(
        ("MEMRA_", "CUDA_", "NVIDIA_", "CUBLAS_", "NCCL_", "OMP_", "MKL_", "OPENBLAS_")) or k in ("LD_LIBRARY_PATH", "LD_PRELOAD")}


def binary_id(path):
    with path.open("rb") as f:
        header = f.read(20)
    q.require(header[:4] == b"\x7fELF" and header[4:6] == b"\x02\x01" and
              int.from_bytes(header[18:20], "little") == 62, f"not a native x86-64 ELF: {path}")
    return {**q.file_identity(path), "format": "ELF-x86_64"}


def platform_identity():
    release = platform.freedesktop_os_release()
    return {"profile": release["ID"] + "-" + release["VERSION_ID"],
            "machine": platform.machine(), "glibc": platform.libc_ver()[1]}


def prepare_build_source(repo, expected, out):
    provenance = out / "provenance"
    # Keep the full immutable checkout outside the compiler-visible input view.
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--shared", "--no-checkout",
                    str(repo), str(provenance)], check=True)
    subprocess.run(["git", "-C", str(provenance), "-c", "core.hooksPath=/dev/null",
                    "checkout", "--detach", expected["commit"]], check=True)
    q.require(clean_source(provenance) == expected, "owned provenance differs from requested Git input")
    return release_input_view.materialize(provenance, expected, out / "source")


def build(args):
    q.require(sys.platform == "linux", "controlled native builds require Linux input isolation")
    bwrap = args.bwrap or shutil.which("bwrap")
    q.require(bwrap, "controlled native builds require bubblewrap; no unisolated fallback")
    bwrap = Path(bwrap).resolve()
    source = clean_source(args.repo)
    q.require(source["commit"] == args.expected_head, "build checkout differs from requested source")
    q.require(not args.out.is_relative_to(args.repo), "build output/source staging must be outside the input checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    build_source = prepare_build_source(args.repo, source, args.out)
    prepare_cargo_home(args.out)
    rustc = Path(subprocess.check_output(["rustup", "which", "--toolchain", "1.97.1", "rustc"], text=True).strip())
    cargo = Path(subprocess.check_output(["rustup", "which", "--toolchain", "1.97.1", "cargo"], text=True).strip())
    env = build_environment(args.out, args.nvcc.resolve(), rustc)
    q.require(cargo.parent.parent == rustc.parent.parent, "cargo/rustc toolchain roots differ")
    # Fetching dependencies never executes workspace build scripts. Compilation is
    # offline inside the isolated view, so it cannot fetch excluded provenance.
    with (args.out / "fetch.log").open("w") as log:
        fetch = subprocess.run([str(cargo), "fetch", "--locked", "--target", "x86_64-unknown-linux-gnu"],
                               cwd=build_source, env=env, stdout=log, stderr=subprocess.STDOUT)
    q.require(fetch.returncode == 0, "dependency fetch failed; fetch.log retained")
    hidden = [args.repo, args.out / "provenance"]
    for option in ("--git-dir", "--git-common-dir"):
        path = Path(q.git(args.repo, "rev-parse", option).decode().strip())
        hidden.append(path if path.is_absolute() else args.repo / path)
    prefix, compiler_env = release_input_view.sandbox(build_source, args.out, rustc.parent.parent,
        args.nvcc.resolve().parent.parent, env, bwrap, hidden)
    command = prefix + ["/toolchain/bin/cargo", "build", "--offline", "--release", "--locked", "--jobs", str(args.jobs),
               "-p", "memra-engine", "--bin", "kernel-check", "--bin", "run-gen", "--bin", "run-spec",
               "--bin", "argmax-margin-probe", "-p", "memra-server", "--bin", "memra-server",
               "-p", "memra-tokenizer", "--bin", "tok-parity"]
    write(args.out / "source.json", source)
    result = {"schema": "memra-native-build-v3", "source": reference(args.out, "source.json"),
              "source_before": source["inputs_sha256"], "command": command,
              "docs_rs": "DOCS_RS" in env, "cuda_arch": env["MEMRA_CUDA_ARCH"],
              "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "platform": platform_identity(),
              "recipe": {"policy": "controlled-cargo-v3", "cargo_home": "fresh-config-free",
                         "checkout": release_input_view.POLICY, "build_source": "fingerprinted-input-view",
                         "cargo_config": "tracked-jobs-only",
                         "sandbox": {"policy": release_input_view.SANDBOX_POLICY,
                                     "executable": q.file_identity(bwrap),
                                     "version": subprocess.check_output([str(bwrap), "--version"], env=env, text=True).strip()},
                         "compilers": {"cargo": q.file_identity(cargo), "rustc": q.file_identity(rustc),
                                       "nvcc": q.file_identity(args.nvcc.resolve())}},
              "input_view_before": release_input_view.verify(build_source, source),
              "compiler_environment": {k: q.digest(v.encode()) for k, v in sorted(compiler_env.items())},
              "fetch_log": reference(args.out, "fetch.log"),
              "numeric_environment": numeric_environment(compiler_env),
              "rustc": subprocess.check_output([str(rustc), "-Vv"], env=env, text=True),
              "nvcc": subprocess.check_output([env["MEMRA_NVCC"], "--version"], env=env, text=True)}
    write(args.out / "build.pending.json", result)
    with (args.out / "build.log").open("w") as log:
        result["exit_code"] = subprocess.run(command, cwd=build_source, env=env, stdout=log,
                                             stderr=subprocess.STDOUT).returncode
    result["log"] = reference(args.out, "build.log")
    write(args.out / "build.pending.json", result)
    q.require(result["exit_code"] == 0, "native build failed; build.pending.json and log retained")
    result["input_view_after"] = release_input_view.verify(build_source, source)
    after = clean_source(args.out / "provenance")
    q.require(after == source, "owned provenance changed during native build")
    result["source_after"] = after["inputs_sha256"]
    result["binaries"] = {name: binary_id(args.out / "target/release" / name) for name in q.BINARIES}
    write(args.out / "build.json", result)
    print("BUILT (not GPU-qualified):", args.out / "build.json")


def live_lease(*, generic=True, observations=None):
    q.require(sys.platform == "linux", "native capture requires Linux physical-card leases")
    observation_start = time.monotonic_ns()
    path = Path(os.environ.get("MEMRA_GPU_LEASE_FILE", ""))
    q.require(path.is_file(), "invoke capture through the coordinator's memra-gpu-run wrapper")
    lease = q.json_bytes(path.read_bytes())
    ids = lease["requested_uuids"]
    q.require(type(ids) is list and ids and len(set(ids)) == len(ids)
              and all(isinstance(x, str) and q.GPU_UUID.fullmatch(x) for x in ids)
              and lease["lock_order"] == sorted(ids) and (not generic or len(ids) == 1),
              "capture physical-card set differs from its selected profile")
    q.require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(ids), "visible GPU differs from lease")
    ancestors, pid = set(), os.getpid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        status = Path(f"/proc/{pid}/status").read_text()
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    q.require(lease["wrapper_pid"] in ancestors and lease["child_pid"] in ancestors
              and lease["wrapper_pid"] != lease["child_pid"], "lease owner is not the live process ancestry")
    locks = Path("/proc/locks").read_text().splitlines()
    selected_locks = []
    for uuid in sorted(ids):
        lock = Path(f"/tmp/memra-gpu-locks/{uuid}.lock")
        q.require(lease["lock_files"].get(uuid) == str(lock), "noncanonical physical-card lock")
        st = lock.stat()
        found = False
        for line in locks:
            fields = line.split()
            if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                continue
            major, minor, inode = fields[5].split(":")
            found |= (int(fields[4]) == lease["wrapper_pid"] and int(major, 16) == os.major(st.st_dev)
                      and int(minor, 16) == os.minor(st.st_dev) and int(inode) == st.st_ino)
        q.require(found, "physical-card exclusive FLOCK not held by wrapper")
        if observations is not None:
            matches = [line for line in locks if len(line.split()) >= 8
                and line.split()[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
                and int(line.split()[4]) == lease["wrapper_pid"]
                and tuple([int(x, 16) for x in line.split()[5].split(":")[:2]] + [int(line.split()[5].split(":")[2])])
                    == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)]
            q.require(len(matches) == 1, "ambiguous physical lock observation")
            selected_locks.append({"uuid": uuid, "path": str(lock), "device_major": os.major(st.st_dev),
                "device_minor": os.minor(st.st_dev), "inode": st.st_ino, "raw": matches[0]})
    if observations is not None:
        from dataclasses import asdict
        from serving_process import process_identity
        chain, pid = [], os.getpid()
        while pid:
            value = process_identity(pid)
            q.require(value is not None and value.pid not in {p["pid"] for p in chain}, "lease ancestry disappeared")
            chain.append(asdict(value)); pid = value.ppid
        unix_ns = time.time_ns()
        observations.append({"started_ns": observation_start, "finished_ns": time.monotonic_ns(),
            "unix_ns": unix_ns, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "controller": chain[0], "ancestors": chain, "locks": selected_locks})
    return lease


def observe_hardware(ids, *, generic=True):
    raw = subprocess.check_output(["nvidia-smi", "-i", ",".join(ids),
        "--query-gpu=index,uuid,name,compute_cap,driver_version,pci.bus_id,memory.total", "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    devices = [dict(zip(("index", "uuid", "name", "compute_cap", "driver_version", "pci_bus_id", "memory_mib"), row)) for row in rows]
    by_uuid = {d["uuid"]: d for d in devices}
    q.require(set(by_uuid) == set(ids), "cannot observe every leased GPU")
    # release-battery.sh currently queries physical NVML index 0 for headroom.
    # Refuse another CUDA-visible card; do not reinterpret CUDA ordinals as NVML indices.
    headroom_uuid = None
    if generic:
        headroom_uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
                                                "--format=csv,noheader,nounits"], text=True).strip()
        q.require(ids == [headroom_uuid] and by_uuid[headroom_uuid]["index"] == "0",
                  "battery headroom queries NVML GPU0; leased CUDA UUID differs")
        q.require(all("RTX PRO 6000 Blackwell" in d["name"] and d["compute_cap"] == "12.0"
                      for d in devices), "capture requires the designated PRO 6000 Blackwell hardware class")
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"])
    return {"devices": [by_uuid[x] for x in ids], "topology_sha256": q.digest(topology),
            "headroom_query": ({"nvml_index": 0, "uuid": headroom_uuid} if generic else None),
            "runtime_platform": platform_identity()}, topology


def model_inventory(repo, oracle_dir):
    repo, oracle_dir = repo.resolve(), oracle_dir.resolve()
    roster = q.read_roster((repo / "tools/release-roster.tsv").read_bytes())
    q.require(oracle_dir.is_dir(), "kernel oracle directory is missing")
    files = {Path(row["path"]) for row in roster} | set(oracle_dir.glob("*.gguf"))
    q.require(any(oracle_dir.glob("*.gguf")), "kernel oracle inventory is empty")
    result = {}
    for path in sorted(files):
        actual = path if path.is_absolute() else repo / path
        release_inputs.single_file_gguf(actual)
        result[str(path)] = q.file_identity(actual)
    return result


def capture(args):
    lease = live_lease()
    source = clean_source(args.repo)
    q.require(source["commit"] == args.expected_head, "capture checkout differs from requested source")
    built = q.json_bytes((args.build / "build.json").read_bytes())
    q.require(built.get("schema") == "memra-native-build-v3"
              and built.get("recipe", {}).get("policy") == "controlled-cargo-v3",
              "capture requires an isolated v3 build; older provenance is unqualified")
    q.require(q.json_bytes((args.build / "source.json").read_bytes()) == source, "build is for different source")
    q.validate_build(built, source, built["source"], q.Evidence(args.build))
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("build.json", "source.json", "build.log", "fetch.log"):
        shutil.copy2(args.build / name, args.out / name)
    staged = args.repo / "target/release"
    staged.mkdir(parents=True, exist_ok=True)
    for name in q.BINARIES:
        original = args.build / "target/release" / name
        q.require(binary_id(original) == built["binaries"][name], f"build binary changed: {name}")
        shutil.copy2(original, staged / name)
    binaries = lambda: {name: binary_id(staged / name) for name in q.BINARIES}
    env = environment()
    env["MEMRA_KC_MODELS_DIR"] = str(args.oracles.resolve())
    env["PYTHONPYCACHEPREFIX"] = str(args.out / "python-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    ids = lease["requested_uuids"]
    models_before = model_inventory(args.repo, args.oracles)
    activity = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader,nounits"], text=True)
    occupied = [row for row in csv.reader(io.StringIO(activity), skipinitialspace=True)
                if row and row[0] in ids]
    q.require(not occupied, "leased GPU already has a compute process; do not disturb it")
    hardware, topology = observe_hardware(ids)
    (args.out / "topology.txt").write_bytes(topology)
    (args.out / "roster.tsv").write_bytes((args.repo / "tools/release-roster.tsv").read_bytes())
    manifests = {}
    for path in q.MANIFESTS:
        name = Path(path).name
        shutil.copy2(args.repo / path, args.out / name)
        manifests[path] = reference(args.out, name)
    run = {"schema": "memra-native-release-run-v1", "source_before": source["inputs_sha256"],
           "binaries_before": binaries(), "models_before": models_before,
           "numeric_environment": numeric_environment(env), "hardware": hardware,
           "oracle_directory": str(args.oracles.resolve()),
           "lease_owner": {k: lease[k] for k in ("wrapper_pid", "child_pid", "requested_uuids")},
           "roster": reference(args.out, "roster.tsv"), "manifests": manifests,
           "topology": reference(args.out, "topology.txt"),
           "command": ["bash", "tools/release-battery.sh", "--generic-only", "--evidence-dir", "cells"],
           "started_unix": time.time()}
    write(args.out / "run.pending.json", run)
    command = ["bash", "tools/release-battery.sh", "--generic-only", "--evidence-dir", str(args.out / "cells")]
    with (args.out / "battery.log").open("w") as log, (args.out / "telemetry.csv").open("w") as samples:
        telemetry = subprocess.Popen(["nvidia-smi", "-i", ids[0],
            "--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,nounits", "--loop-ms=250"], stdout=samples, stderr=subprocess.STDOUT)
        try:
            result = subprocess.run(command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            q.require(telemetry.poll() is None, "native telemetry stopped before the battery finished")
        finally:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill(); telemetry.wait(timeout=10)
    run["exit_code"] = result.returncode
    run["telemetry"] = reference(args.out, "telemetry.csv")
    run["source_after"] = clean_source(args.repo)["inputs_sha256"]
    run["binaries_after"] = binaries()
    run["models_after"] = model_inventory(args.repo, args.oracles)
    run["hardware_after"], _ = observe_hardware(ids)
    run["numeric_environment_after"] = numeric_environment(env)
    run["battery_log"] = reference(args.out, "battery.log")
    run["cells"] = []
    for row in (args.out / "cells/runs.tsv").read_text().splitlines():
        kind, model, code, log = row.split("\t")
        run["cells"].append({"kind": kind, "model": model, "exit_code": int(code),
                             "log": reference(args.out, "cells/" + q.safe_path(log))})
    live_lease()
    run["finished_unix"] = time.time()
    write(args.out / "run.json", run)
    q.require(result.returncode == 0, "battery failed; all captured evidence retained")
    print("GENERIC ONLY, NOT RELEASE QUALIFIED: wait for wrapper cleanup and required serving stage, then seal", args.out)


def seal(args):
    q.require(not (args.out / "record.json").exists(), "record already exists; do not overwrite evidence")
    q.require(not (args.out / "generic-record.json").exists(),
              "v2 seal already attempted here; preserve it and use a fresh copy of the capture")
    lease_bytes = args.lease.read_bytes()
    lease_path = args.out / "lease.json"
    if lease_path.exists():
        q.require(lease_path.read_bytes() == lease_bytes, "different lease would overwrite retained evidence")
    else:
        with lease_path.open("xb") as output:
            output.write(lease_bytes)
    evidence = q.Evidence(args.out)
    record = {"schema": "memra-release-qualification-v1", "status": "qualified"}
    for key in ("source", "build", "run", "lease"):
        record[key] = reference(args.out, key + ".json")
    source, run = evidence.obj(record["source"]), evidence.obj(record["run"])
    verdicts = q.cell_verdicts(run, evidence, args.repo, "HEAD")
    record["verdicts"] = verdicts
    record["identity_sha256"] = q.object_digest({"source": source["inputs_sha256"],
        "build": record["build"]["sha256"], "models": run["models_before"],
        "numeric": run["numeric_environment"], "hardware": run["hardware"], "verdicts": verdicts})
    record["payloads"] = {str(path.relative_to(args.out)): q.sha256_file(path)
                          for path in sorted(args.out.rglob("*")) if path.is_file()
                          and not path.name.endswith(".pending.json") and path.name != "record.json"}
    q.validate_historical_record(record, evidence, args.repo, "HEAD", binaries=args.repo / "target/release",
                      models=model_inventory(args.repo, args.oracles))
    write(args.out / "generic-record.json", record)
    from serving_run import import_stage, validate_stage_directory
    stage = validate_stage_directory(args.serving, args.repo, "HEAD")
    q.require(stage["source"] == source and stage["build"] == evidence.obj(record["build"]),
              "serving stage belongs to another source/build")
    serving_ref = import_stage(args.serving, args.out)
    verdicts = {"generic": record["verdicts"], "serving": stage["verdicts"]}
    full = {"schema": "memra-release-qualification-v2", "status": "qualified",
            "source": record["source"], "build": record["build"], "generic": reference(args.out, "generic-record.json"),
            "serving": serving_ref, "verdicts": verdicts}
    full["payloads"] = {str(p.relative_to(args.out)): q.sha256_file(p) for p in sorted(args.out.rglob("*"))
                        if p.is_file() and not p.name.endswith(".pending.json") and p.name != "record.json"}
    full["identity_sha256"] = q.object_digest({"source": source["inputs_sha256"], "build": full["build"]["sha256"],
        "generic": full["generic"]["sha256"], "serving": full["serving"]["sha256"], "verdicts": verdicts})
    q.validate_record(full, q.Evidence(args.out), args.repo, "HEAD", binaries=args.repo / "target/release")
    write(args.out / "record.json", full)
    print("SEALED v2 generic + required serving evidence:", args.out / "record.json")


def bank(args):
    record = q.json_bytes((args.out / "record.json").read_bytes())
    q.validate_record(record, q.Evidence(args.out), args.repo, "HEAD")
    q.require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.name), "invalid publication name")
    destination = args.repo / q.PUBLICATION / args.name
    q.require(destination.resolve().is_relative_to(args.repo.resolve()), "publication directory escapes repository")
    q.require(not (args.repo / q.POINTER).is_symlink(), "qualification pointer cannot be a symlink")
    q.require(not destination.exists(), "publication already exists")
    # Bank only evidence. Executables stay in the preserved build archive, identified by hash.
    destination.mkdir(parents=True, exist_ok=False)
    evidence = q.Evidence(args.out)
    # Copy only the verified manifest closure. An unrelated/symlinked file added
    # beside the receipt must never be swept into a public evidence publication.
    for name, expected in record["payloads"].items():
        data = evidence.bound({"path": name, "sha256": expected})
        target = destination / q.safe_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / "record.json").write_bytes((args.out / "record.json").read_bytes())
    relative = str(destination.relative_to(args.repo) / "record.json")
    entries = []
    if args.append:
        # Append only after all earlier profiles have also been validated for this source.
        q.verify_published(args.repo, "HEAD")
        pointer = q.json_bytes((args.repo / q.POINTER).read_bytes())
        entries = pointer.get("records", [pointer])
    entries.append({"record": relative, "sha256": q.digest((destination / "record.json").read_bytes())})
    write(args.repo / q.POINTER, {"records": entries})
    print("BANKED; commit the research-only publication:", relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "capture", "seal", "bank"))
    parser.add_argument("--repo", type=Path, default=q.ROOT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--bwrap", type=Path, help="Linux bubblewrap executable; default: bwrap on PATH")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--oracles", type=Path)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--serving", type=Path, help="sealed required serving stage directory (mandatory for full seal)")
    parser.add_argument("--name")
    parser.add_argument("--append", action="store_true", help="bank an additional qualified build profile")
    args = parser.parse_args()
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if args.oracles is not None:
        args.oracles = args.oracles.resolve()
    try:
        q.require(args.jobs > 0, "jobs must be positive")
        for mode, required in {"build": ("nvcc", "expected_head"), "capture": ("build", "oracles", "expected_head"),
                               "seal": ("lease", "oracles", "serving"), "bank": ("name",)}.items():
            if args.mode == mode:
                q.require(all(getattr(args, key) is not None for key in required), f"{mode} requires {required}")
        if args.expected_head is not None:
            q.require(q.COMMIT.fullmatch(args.expected_head), "expected-head must be an immutable 40-character commit")
        globals()[args.mode](args)
        return 0
    except (q.GateError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"UNQUALIFIED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
