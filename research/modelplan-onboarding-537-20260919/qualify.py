#!/usr/bin/env python3
"""Issue 537 qualification. Build/fetch first; GPU phases require memra-gpu-run's lease."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]
LOCK = LANE / "artifacts.lock.json"
COUNTS = {"focused": 1, "dense": 1, "ornith": 1, "serve-ornith": 1,
          "step-gguf": 2, "step-fp8": 3}
BINS = ["kernel-check", "run-gen", "run-spec", "argmax-margin-probe"]


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def run(out, name, argv, env, patterns=()):
    log = out / (name + ".log")
    print(f"RUN {name}: {argv}", flush=True)
    start = time.time()
    timeout = None
    with log.open("xb") as output:
        try:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=14400)
            code = result.returncode
        except subprocess.TimeoutExpired as error:
            timeout = str(error)
            code = None
    text = log.read_text(errors="replace")
    matched = all(re.search(pattern, text, re.M) for pattern in patterns)
    save(out / (name + ".json"), {"argv": [str(a) for a in argv],
         "exit_code": code, "timeout": timeout, "required_output_matched": matched,
         "started_unix": start, "finished_unix": time.time(), "log_sha256": sha(log)})
    if timeout or code or not matched:
        raise RuntimeError(f"{name} failed; complete stdout/stderr: {log}")
    return text


def ancestors(pid):
    seen = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        # comm is parenthesized and can contain spaces/parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(fields[1])
    return seen


def validate_lease(lease, visible, chain, lock_rows, stats, count):
    """Pure validation seam; caller supplies the actual kernel lock snapshot and stat tuples."""
    requested = lease["requested_uuids"]
    if (len(requested) != count or len(set(requested)) != count or requested != visible
            or any(not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid)
                   for uuid in requested)):
        raise RuntimeError("GPU visibility must equal the exact requested physical UUID set")
    if lease["wrapper_pid"] not in chain or lease["child_pid"] not in chain:
        raise RuntimeError("lease wrapper/child is not an ancestor of this runner")
    if lease["lock_order"] != sorted(requested) or set(lease["lock_files"]) != set(requested):
        raise RuntimeError("lease must describe the complete set in stable lock order")
    owned = set()
    for line in lock_rows.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[1:4] == ["FLOCK", "ADVISORY", "WRITE"]:
            if int(fields[4]) == lease["wrapper_pid"]:
                major, minor, inode = fields[5].split(":")
                owned.add((int(major, 16), int(minor, 16), int(inode)))
    expected = set()
    for uuid in requested:
        path = f"/tmp/memra-gpu-locks/{uuid}.lock"
        if lease["lock_files"][uuid] != path:
            raise RuntimeError("noncanonical GPU lock path")
        expected.add(stats[path])
    if len(expected) != count:
        raise RuntimeError("distinct GPUs must have distinct lock inodes")
    actual_gpu_locks = {identity for identity in stats.values() if identity in owned}
    if actual_gpu_locks != expected:
        raise RuntimeError("wrapper does not hold exactly the requested exclusive GPU FLOCKs")


def gpu_lease(count):
    lease = json.loads(Path(os.environ["MEMRA_GPU_LEASE_FILE"]).read_text())
    stats = {}
    for path in Path("/tmp/memra-gpu-locks").glob("GPU-*.lock"):
        st = path.stat()
        stats[str(path)] = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    validate_lease(lease, visible, ancestors(os.getpid()),
                   Path("/proc/locks").read_text(), stats, count)
    return lease


def artifacts():
    return {a["role"]: a for a in json.loads(LOCK.read_text())["artifacts"]}


def verify_artifact(role, models):
    artifact = artifacts()[role]
    directory = models / role
    # Directory loaders switch to a repack when manifest.json exists. Extra tensor/index
    # files can also alter source selection: a verified subset is not an artifact lock.
    expected = {f["path"] for f in artifact["files"]}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if (path.is_file() or path.is_symlink())
              and path.relative_to(directory).parts[:2] != (".cache", "huggingface")}
    if actual != expected or (directory / "manifest.json").exists():
        raise RuntimeError(f"artifact inventory differs for {role}: extra={sorted(actual-expected)}, "
                           f"missing={sorted(expected-actual)}")
    for file in artifact["files"]:
        relative = Path(file["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("invalid artifact path")
        path = models / role / relative
        if path.stat().st_size != file["size"] or sha(path) != file["sha256"]:
            raise RuntimeError(f"artifact bytes differ from lock: {path}")
    return artifact


def server_smoke(out, binary, model, env, port):
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()
    env.update(MEMRA_COMPAT="openai", MEMRA_MODELS=f"q537={model}",
               MEMRA_ADDR=f"127.0.0.1:{port}", MEMRA_CTX="2048")
    log = out / "server.log"
    with log.open("xb") as output:
        # Stay in the wrapper-supervised process group so cancellation cannot leave
        # a detached GPU server alive after the coordinator releases the lease.
        child = subprocess.Popen([binary], cwd=ROOT, env=env, stdout=output,
                                 stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 300
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"server exited {child.returncode}; see {log}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"server startup timed out; see {log}")
                    time.sleep(0.2)
            listeners = {line.split()[9] for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
                         if line.split()[1].endswith(f":{port:04X}") and line.split()[3] == "0A"}
            owned = set()
            for fd in Path(f"/proc/{child.pid}/fd").iterdir():
                try:
                    owned.add(os.readlink(fd))
                except FileNotFoundError:
                    pass  # A transient worker fd closed while enumerating; not a listener.
            if not any(f"socket:[{inode}]" in owned for inode in listeners):
                raise RuntimeError("health response did not come from this server's listener")
            body = json.dumps({"model": "q537", "messages": [{"role": "user",
                               "content": "Count from one to five."}], "max_tokens": 32,
                               "temperature": 0}).encode()
            request = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            save(out / "server-response.json", result)
            if len(result.get("choices", [])) != 1 or result.get("usage", {}).get("completion_tokens", 0) <= 0:
                raise RuntimeError("server response has no completed generation")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            save(out / "server-process.json", {"pid": child.pid, "exit_code": child.returncode,
                                                "log_sha256": sha(log), "port": port})


def main():
    def terminate(_signum, _frame):
        raise SystemExit(143)  # unwind owned server cleanup on ordinary SIGTERM
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["fetch", "build", *COUNTS])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--arch", choices=["120a", "100a", "90a"], default="120a")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--nvcc", type=Path, help="explicit compiler for build (or MEMRA_NVCC)")
    parser.add_argument("--port", type=int, default=15379)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.models:
        args.models = args.models.resolve()
    # Runtime knobs must be explicit in each receipt, not inherited from another campaign.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["CARGO_TARGET_DIR"] = str(ROOT / "target")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise RuntimeError("qualification requires a clean, committed source tree")
    if args.phase == "fetch":
        if not args.models:
            parser.error("fetch requires --models")
        for role, artifact in artifacts().items():
            run(args.out, "fetch-" + role, ["hf", "download", artifact["repo"],
                *[f["path"] for f in artifact["files"]], "--revision", artifact["revision"],
                "--local-dir", str(args.models / role)], env)
            verify_artifact(role, args.models)
        save(args.out / "artifacts-verified.json", {"artifact_lock_sha256": sha(LOCK),
             "models_root": str(args.models), "verified_unix": time.time()})
        return
    if args.phase == "build":
        selected_nvcc = args.nvcc if args.nvcc is not None else os.environ.get("MEMRA_NVCC")
        if not selected_nvcc:
            parser.error("build requires --nvcc or MEMRA_NVCC; ambient compiler selection is not qualification")
        nvcc = Path(selected_nvcc).resolve(strict=True)
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            parser.error("selected nvcc must be an executable file")
        # Runtime controls remain filtered, but the selected build tool must survive.
        env["MEMRA_NVCC"] = str(nvcc)
        env["MEMRA_CUDA_ARCH"] = args.arch
        env["CUDA_VISIBLE_DEVICES"] = ""
        compiler = {"path": str(nvcc), "sha256": sha(nvcc),
                    "version": run(args.out, "nvcc-version", [str(nvcc), "--version"], env).strip()}
        command = ["cargo", "build", "--release", "-j", str(args.jobs), "-p", "memra-engine"]
        for name in BINS:
            command.extend(["--bin", name])
        run(args.out, "build-engine", command, env)
        run(args.out, "build-server", ["cargo", "build", "--release", "-j", str(args.jobs),
            "-p", "memra-server", "--bin", "memra-server"], env)
        text = run(args.out, "build-focused", ["cargo", "test", "--release", "-j", str(args.jobs),
                   "-p", "memra-engine", "--test", "step_rope_load_gpu", "--no-run",
                   "--message-format=json"], env)
        executable = None
        for line in text.splitlines():
            if line.startswith("{"):
                item = json.loads(line)
                if item.get("target", {}).get("name") == "step_rope_load_gpu" and item.get("executable"):
                    executable = item["executable"]
        if not executable:
            raise RuntimeError("cargo did not report the focused test executable")
        binaries = {name: str(ROOT / "target/release" / name) for name in BINS}
        binaries["memra-server"] = str(ROOT / "target/release/memra-server")
        binaries["focused"] = executable
        if sha(nvcc) != compiler["sha256"]:
            raise RuntimeError("selected compiler changed during build")
        save(args.out / "build.json", {"commit": commit, "cuda_arch": args.arch,
             "compiler": compiler,
             "binaries": {name: {"path": path, "sha256": sha(path)} for name, path in binaries.items()}})
        return
    if not args.build_record:
        parser.error("GPU phases require --build-record")
    build = json.loads(args.build_record.read_text())
    if build["commit"] != commit:
        raise RuntimeError("build record is stale for this source commit")
    binaries = build["binaries"]
    for binary in binaries.values():
        if sha(binary["path"]) != binary["sha256"]:
            raise RuntimeError("binary changed after build")
    lease = gpu_lease(COUNTS[args.phase])
    save(args.out / "lease.json", lease)
    hardware = run(args.out, "hardware", ["nvidia-smi", "-i", ",".join(lease["requested_uuids"]),
        "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version,power.limit,clocks.current.sm",
        "--format=csv,noheader"], env)
    hardware_uuids = [row[1].strip() for row in csv.reader(hardware.splitlines()) if row]
    if len(hardware_uuids) != len(lease["requested_uuids"]) or set(hardware_uuids) != set(lease["requested_uuids"]):
        raise RuntimeError("hardware query does not resolve to the leased canonical UUIDs")
    run(args.out, "topology", ["nvidia-smi", "topo", "-m"], env)
    run(args.out, "compute-apps-before", ["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], env)
    target = build["cuda_arch"] == "120a" and all(
        "RTX PRO 6000 Blackwell" in line for line in hardware.splitlines() if line)
    save(args.out / "identity.json", {"commit": commit, "build": build,
         "artifact_lock_sha256": sha(LOCK), "target_pro6000": target,
         "qualification_scope": "native target" if target else "secondary-backend diagnostic only"})
    if args.phase == "focused":
        run(args.out, "step-rope-load", [binaries["focused"]["path"], "--ignored", "--test-threads=1",
            "--nocapture"], env, [
                r"STEP_ROPE_LOAD_PASS .*missing_factors_refused=true identity_fixture_max_delta=",
                r"1 passed; 0 failed"])
    else:
        if not args.models:
            parser.error("model phases require --models")
        role = "ornith" if args.phase == "serve-ornith" else args.phase.replace("-", "_")
        artifact = verify_artifact(role, args.models)
        save(args.out / "artifact.json", artifact)
        path = args.models / role
        if args.phase == "step-fp8":
            env.update(MEMRA_PP_STAGES="3", MEMRA_PP_DEVICES="0,1,2", MEMRA_LOAD_MTP="1")
        elif args.phase == "step-gguf":
            env.update(MEMRA_PP_STAGES="2", MEMRA_PP_DEVICES="0,1",
                       MEMRA_MTP_DRAFT=str(path / "Step3.7-flash-mtp-Q8_0.gguf"))
            path /= "IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"
        else:
            path /= artifact["files"][0]["path"]
        env.update(MEMRA_NGEN="64", MEMRA_PRINT_TEXT="1", MEMRA_PARALLEL="off")
        # The published corpus prompt exercises the window boundary on Step.
        prompt = ROOT / "research/e2e/prompts/board-2048.txt"
        env["MEMRA_PROMPT_FILE"] = str(prompt)
        save(args.out / "runtime-env.json", {k: v for k, v in env.items()
             if k.startswith("MEMRA_") or k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE")})
        gpu_lease(COUNTS[args.phase])
        if args.phase == "serve-ornith":
            server_smoke(args.out, binaries["memra-server"]["path"], path, env, args.port)
            gpu_lease(1)
            run(args.out, "compute-apps-after", ["nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], env)
            save(args.out / "PASS.json", {"phase": args.phase, "commit": commit,
                 "target_pro6000": target, "native_model_qualified": False})
            return
        if args.phase == "dense":
            run(args.out, "kernel-check", [binaries["kernel-check"]["path"], str(path)], env,
                [r"ALL GREEN"])
        run(args.out, "run-gen", [binaries["run-gen"]["path"], str(path)], env,
            [r"argmax=.*decode argmax=.*\bMATCH\b"])
        if args.phase != "dense":
            patterns = [rf"\[generate_spec K={k}\]" for k in range(1, 9)]
            patterns += [r"self-consistency: PASS"]
            text = run(args.out, "run-spec", [binaries["run-spec"]["path"], str(path)], env, patterns)
            if len(re.findall(r"self-consistency: PASS", text)) != 8 or "self-consistency: FAIL" in text:
                raise RuntimeError("the complete K=1..8 speculative gate did not pass")
    gpu_lease(COUNTS[args.phase])
    run(args.out, "compute-apps-after", ["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], env)
    save(args.out / "PASS.json", {"phase": args.phase, "commit": commit, "target_pro6000": target,
         "native_model_qualified": False, "note": "phase receipt only; whole required bundle remains necessary"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"QUALIFICATION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
