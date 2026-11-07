#!/usr/bin/env python3
"""Capture the exact nvcc invocations crates/memra-engine/build.rs makes for each build variant,
without compiling a kernel: build.rs is compiled standalone with rustc and run once per variant
with MEMRA_NVCC pointing at a logging stand-in that writes empty outputs.

Output: $MEMRA_IMPACT_WORK/stage0/buildrs/out-<variant>/nvcc.jsonl  (one {"cwd","args"} per call)
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
REPO = os.environ.get("MEMRA_IMPACT_REPO", str(Path(__file__).resolve().parents[2]))
ROOT = os.path.join(WORK, "stage0", "buildrs")

# (variant tag, MEMRA_CUDA_ARCH, extra env). CUTLASS variants are build-time doors.
VARIANTS = [("120a", "120a", {}), ("120a-MEMRA_CUTLASS", "120a", {"MEMRA_CUTLASS": "1"}),
            ("100a", "100a", {}), ("90a", "90a", {}), ("89", "89", {})]

FAKE = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("nvcc: NVIDIA (R) Cuda compiler driver\\nCuda compilation tools, release 13.1, V13.1.80"); sys.exit(0)
with open(os.environ["FAKE_NVCC_LOG"], "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "args": args}) + "\\n")
if "-o" in args:
    open(args[args.index("-o") + 1], "wb").close()
"""


def main():
    os.makedirs(ROOT, exist_ok=True)
    fake = os.path.join(ROOT, "fake-nvcc")
    with open(fake, "w") as f:
        f.write(FAKE)
    os.chmod(fake, 0o755)
    exe = os.path.join(ROOT, "buildrs")
    crate = os.path.join(REPO, "crates/memra-engine")
    r = subprocess.run(["rustc", "--edition", "2024", "-O", "-o", exe, os.path.join(crate, "build.rs")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("rustc build.rs failed:\n" + r.stderr[-3000:])
    version = "0.0.0"
    for line in open(os.path.join(REPO, "Cargo.toml")):
        if line.startswith("version"):
            version = line.split('"')[1]
            break
    failed = []
    for tag, arch, extra in VARIANTS:
        out = os.path.join(ROOT, "out-" + tag)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "OUT_DIR": out, "CARGO_MANIFEST_DIR": crate,
               "TARGET": "x86_64-unknown-linux-gnu", "HOST": "x86_64-unknown-linux-gnu", "PROFILE": "release",
               "CARGO_PKG_VERSION": version, "MEMRA_CUDA_ARCH": arch, "MEMRA_NVCC": fake,
               "FAKE_NVCC_LOG": os.path.join(out, "nvcc.jsonl"), **extra}
        r = subprocess.run([exe], cwd=crate, env=env, capture_output=True, text=True)
        n = sum(1 for _ in open(env["FAKE_NVCC_LOG"])) if os.path.exists(env["FAKE_NVCC_LOG"]) else 0
        print(f"{tag}: exit={r.returncode} nvcc_calls={n}")
        if r.returncode != 0:
            open(os.path.join(out, "stderr.txt"), "w").write(r.stderr)
        if r.returncode != 0 or n == 0:
            failed.append(f"{tag} (exit {r.returncode}, {n} nvcc calls; {out}/stderr.txt)")
    # a variant whose replay failed has missing or partial nvcc logs: its kernels would read as dead
    if failed:
        sys.exit("build.rs replay failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
