#!/usr/bin/env python3
"""Repeatable scratch-tree preparation + legacy prefix battery; not active-KV proof.
No perf verdicts: all GPU results are single-run correctness probes. Missing native
active gate refuses after baseline collection rather than relabeling legacy tests.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = "/tmp/memra-5090.lock"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub-fail", action="store_true", help="dry-run error-path self-test")
    ap.add_argument("--exclusive-non-serving", action="store_true")
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--artifact-sha256")
    ap.add_argument("--tokens-8k", type=Path)
    ap.add_argument("--tokens-32k", type=Path)
    ap.add_argument("--fit-plan", type=Path, help="artifact/source-bound required-free-byte envelope for both contexts")
    ap.add_argument("--active-gate", type=Path, help="future native gate; absent means BLOCKED")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.stub_fail and not args.dry_run:
        ap.error("--stub-fail requires --dry-run")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # A non-deployment hardware label, not a rented instance id or provider hostname.
    host = "cpu-stub" if args.dry_run else "rtx5090"
    out = (args.out or HERE / "raw" / f"{host}-{stamp}-{os.getpid()}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    count = 0
    fit_plan = None
    manifest = {"mode": "DRY_RUN" if args.dry_run else "SINGLE_RUN_CORRECTNESS", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "patch_sha256": sha(HERE / "HOSTPREFIX-PATCH.diff"), "lock": LOCK, "contexts": [8192, 32768], "excluded_pro_contexts": [131072, 262144], "gpu_claim": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(label, command, cwd=ROOT, gpu=False, internal_lock=False):
        nonlocal count
        count += 1
        actual = ["flock", "-w", "300", LOCK, *command] if gpu and not internal_lock else list(command)
        log = out / f"{count:02d}-{label}.log"
        started = time.monotonic()
        telemetry = None
        telemetry_file = None
        if gpu and not args.dry_run:
            telemetry_file = (out / f"{count:02d}-{label}-250ms.csv").open("wb")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu", "--format=csv", "--loop-ms=250"], stdout=telemetry_file, stderr=subprocess.STDOUT)
        with log.open("wb") as raw:
            proc = subprocess.Popen(["bash", str(HERE / "rig-stub-b.sh"), "fail" if args.stub_fail else label] if args.dry_run else actual, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # tee bytes to raw first, then the console; parse only the closed raw file.
            for chunk in iter(lambda: proc.stdout.read(8192), b""):
                raw.write(chunk)
                raw.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            rc = proc.wait()
        if telemetry:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            telemetry_file.close()
        row = {"cell": label, "command": actual, "cwd": str(cwd), "status": "DRY_RUN" if args.dry_run and rc == 0 else "PASS_COMMAND" if rc == 0 else "FAIL", "exit": rc, "elapsed_seconds": time.monotonic() - started, "raw": log.name, "raw_sha256": sha(log), "gpu_executed": gpu and not args.dry_run}
        with (out / "runs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if rc:
            if not args.dry_run and shutil.which("nvidia-smi"):
                with (out / f"{count:02d}-failure-processes.log").open("wb") as f:
                    subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"], stdout=f, stderr=subprocess.STDOUT, check=False)
            raise RuntimeError(f"{label} exit {rc}; failure cause is in {log.name}")
        return log

    def idle():
        if args.dry_run:
            return
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if apps.strip():
            raise RuntimeError("GPU process present; refusing legacy gate with box-global pkill")
        if subprocess.run(["pgrep", "-x", "memra-server"], stdout=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError("memra-server exists; not an exclusive non-serving box")

    if not args.dry_run:
        if platform.system() != "Linux" or not args.exclusive_non_serving:
            raise RuntimeError("requires Linux and explicit --exclusive-non-serving")
        for tool in ["nvcc", "cargo", "git", "flock", "nvidia-smi", "pgrep"]:
            if not shutil.which(tool):
                raise RuntimeError(f"required tool absent: {tool}")
        for p in [args.artifact, args.tokens_8k, args.tokens_32k]:
            if not p or not p.is_file():
                raise RuntimeError("artifact and immutable 8k/32k prompt token files are mandatory")
        if not args.artifact_sha256 or sha(args.artifact) != args.artifact_sha256:
            raise RuntimeError("artifact SHA256 mismatch or absent")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True)
        if len(gpu.strip().splitlines()) != 1 or "RTX 5090" not in gpu:
            raise RuntimeError("runner admits only one RTX 5090, not a PRO/serving/multi-card box")
        (out / "hardware.txt").write_text(gpu)
        for n, path in [(8192, args.tokens_8k), (32768, args.tokens_32k)]:
            tokens = json.loads(path.read_text())
            if len(tokens) != n - 128 or not all(type(t) is int and 0 <= t < 2**32 for t in tokens):
                raise RuntimeError(f"{n}: need exactly context-128 u32 prompt tokens")
        if not args.fit_plan or not args.fit_plan.is_file():
            raise RuntimeError("fitting-cell placement envelope required: --fit-plan")
        fit_plan = json.loads(args.fit_plan.read_text())
        if fit_plan.get("artifact_sha256") != args.artifact_sha256 or fit_plan.get("source_commit") != manifest["source_commit"]:
            raise RuntimeError("fit-plan artifact/source identity mismatch")
        for n in [8192, 32768]:
            need = fit_plan.get("required_free_bytes", {}).get(str(n))
            if type(need) is not int or need <= 0:
                raise RuntimeError("fit-plan must include positive full run-gen peak envelope for 8k AND 32k")
        manifest.update({"fit_plan_sha256": sha(args.fit_plan), "artifact_sha256": args.artifact_sha256, "tokens_8k_sha256": sha(args.tokens_8k), "tokens_32k_sha256": sha(args.tokens_32k)})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
            raise RuntimeError("tracked source must be committed before rig run")
    scratch = None
    added = False
    branch = f"lane/spill-b-rig-{stamp}-{os.getpid()}"
    work = ROOT if args.dry_run else Path(tempfile.mkdtemp(prefix="memra-spill-b-")) / "tree"
    try:
        if not args.dry_run:
            scratch = work.parent
            run("scratch", ["git", "worktree", "add", "-b", branch, str(work), manifest["source_commit"]])
            added = True
        run("patch-check", ["git", "apply", "--check", str(HERE / "HOSTPREFIX-PATCH.diff")], work)
        artifact = str(args.artifact.resolve()) if args.artifact else "<PINNED_QWEN38_GGUF>"
        for arm in ["before", "after"]:
            if arm == "after":
                run("apply-patch", ["git", "apply", str(HERE / "HOSTPREFIX-PATCH.diff")], work)
            target = work / "target" / f"spill-b-{arm}"
            run(f"{arm}-build", ["cargo", "build", "--release", "-p", "memra-server", "-p", "memra-engine", "--bin", "memra-server", "--bin", "run-gen", "--target-dir", str(target)], work)
            run(f"{arm}-server-tests", ["cargo", "test", "-p", "memra-server", "--lib", "--target-dir", str(target)], work)
            # Qwen fitting contexts FIRST. These prove baseline execution, NOT active tiering.
            for context, token_file in [(8192, args.tokens_8k), (32768, args.tokens_32k)]:
                idle()
                if not args.dry_run:
                    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    if fit_plan["required_free_bytes"][str(context)] > free_mib * 1024 * 1024:
                        raise RuntimeError(f"{context}: does not fit available VRAM; no cell executed, no format fallback")
                tokens = ["<PINNED_U32_TOKENS>"] if args.dry_run else [str(t) for t in json.loads(token_file.read_text())]
                run(f"{arm}-qwen-{context}-baseline", ["env", f"MEMRA_MAX_CTX={context}", "MEMRA_NGEN=128", str(target / "release/run-gen"), artifact, *tokens], work, gpu=True)
            binary = str(target / "release/memra-server")
            if not args.dry_run:
                (out / f"{arm}-binary.sha256").write_text(sha(binary) + "\n")
            # Legacy scripts own flock internally. NEVER wrap these in another flock.
            for label, script, extra in [("identity", "kv-host-spill-identity-gate.sh", []), ("teeth", "kv-host-spill-identity-gate.sh", ["MEMRA_HOSTGATE_TEETH=1"]), ("failures", "kv-host-spill-failure-gate.sh", [])]:
                idle()
                run(f"{arm}-{label}", ["env", f"MEMRA_GPU_LOCK={LOCK}", *extra, "bash", f"tools/{script}", artifact, binary, str(out / f"{arm}-{label}")], work, gpu=True, internal_lock=True)
        # No shipped binary consumes the generic active materializer yet. A supplied future
        # gate must write exact-state, logits and token receipts plus nonzero engagement.
        if not args.active_gate and not args.dry_run:
            raise RuntimeError("BLOCKED: native active-KV GPU gate/binding absent; legacy prefix checks do not qualify active tiers")
        for context in [8192, 32768]:
            for case in ["active", "prefix-restore"]:
                idle()
                gate = str(args.active_gate.resolve()) if args.active_gate else "<FUTURE_NATIVE_ACTIVE_GATE>"
                run(f"{case}-{context}-PENDING", [gate, "--artifact", artifact, "--case", case, "--context", str(context), "--tiers", "host,nvme", "--same-program", "--out", str(out / f"{case}-{context}")], work, gpu=True)
        if not args.dry_run:
            raise RuntimeError("BLOCKED: native gate receipt acceptance and enabled bootstrap not implemented; no GPU tier qualification is claimed")
        print(f"DRY_RUN complete: {count} stub commands; no GPU/build/patch applied. {out}")
        return 0
    finally:
        if added:
            # Only the task-owned scratch tree/branch; no caller work or other lane touched.
            subprocess.run(["git", "worktree", "remove", "--force", str(work)], cwd=ROOT, check=True)
            subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, check=True)
            shutil.rmtree(scratch)
        elif scratch:
            scratch.rmdir()  # only an empty task-created directory; never another worktree


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(1)
