#!/usr/bin/env python3
"""Measure controlled native compilation versus verified local restore without a GPU.

This runs the existing native build producer. It never runs a model, a GPU gate,
or a release capture. The output is preparation evidence only.
"""
import argparse
import atexit
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

_PY_CACHE = tempfile.TemporaryDirectory(prefix="memra-build-measure-python-")
atexit.register(_PY_CACHE.cleanup)
sys.pycache_prefix = _PY_CACHE.name
sys.dont_write_bytecode = True

import local_build_cache as cache


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def measure(repo, out, nvcc, bwrap, jobs):
    repo, out = repo.resolve(), out.resolve()
    if out.is_relative_to(repo) or out.exists():
        raise ValueError("measurement output must be new and outside the source checkout")
    out.mkdir(parents=True)
    started = time.monotonic()
    head = cache.q.commit(repo, "HEAD")
    record = {"schema": "memra-build-reuse-timing-v1", "candidate": head,
              "scope": "controlled compilation and local artifact restore only",
              "gpu_execution": False, "model_qualification": False,
              "compile_repetitions": 1, "restore_repetitions": 3, "jobs": jobs,
              "phases": [], "source_changes": False}
    write(out / "timing.json", record)

    def phase(name, function):
        begin = time.monotonic()
        try:
            result = function()
        except BaseException:
            record["phases"].append({"name": name, "seconds": time.monotonic() - begin, "passed": False})
            write(out / "timing.json", record)
            raise
        record["phases"].append({"name": name, "seconds": time.monotonic() - begin, "passed": True})
        write(out / "timing.json", record)
        return result

    build = out / "native-build"
    command = ["python3", str(repo / "tools/qualify-release.py"), "build", "--repo", str(repo),
               "--expected-head", head, "--out", str(build), "--nvcc", str(nvcc),
               "--bwrap", str(bwrap), "--jobs", str(jobs)]
    with (out / "producer.log").open("w") as log:
        phase("native_pipeline_including_source_staging_fetch_compile_verify",
              lambda: subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True))
    expected = phase("describe_and_verify_exact_source_and_build", lambda: cache.describe(repo, build))
    write(out / "expected.json", expected)
    phase("store_verify_and_publish", lambda: cache.store(repo, build, out / "cache", expected))
    record["binaries"] = expected["build"]["binaries"]
    record["build_context"] = {k: expected["build"][k] for k in ("rustc", "nvcc", "platform", "cuda_arch")}
    record["input_count"] = len(expected["source"]["files"])
    record["payload_bytes"] = sum(v["bytes"] for v in expected["payloads"].values())
    version = phase("built_binary_version", lambda: subprocess.check_output(
        [str(build / "target/release/memra-server"), "--version"]))
    (out / "built-version.txt").write_bytes(version)
    for index in range(3):
        restored = out / f"restored-{index}"
        phase(f"restore_{index}_including_source_and_payload_verification",
              lambda: cache.load(repo, out / "cache", expected, restored))
        restored_version = phase(f"restored_{index}_binary_version", lambda: subprocess.check_output(
            [str(restored / "target/release/memra-server"), "--version"]))
        if restored_version != version:
            raise ValueError("restored binary version output differs")
        (out / f"restored-{index}-version.txt").write_bytes(restored_version)
        # These copies belong only to this measurement; retain the immutable capsule.
        shutil.rmtree(restored)

    changed = repo / "crates/memra-engine/src/lib.rs"
    original = changed.read_bytes()
    try:
        changed.write_bytes(original + b"\n// build-cache invalidation control\n")
        def refusal():
            try:
                cache.load(repo, out / "cache", expected, out / "must-not-restore")
            except cache.q.GateError as error:
                record["changed_source_refusal"] = str(error)
            else:
                raise ValueError("changed source incorrectly reused a build")
        phase("changed_source_refusal", refusal)
    finally:
        changed.write_bytes(original)
    if (out / "must-not-restore").exists():
        raise ValueError("refused restore published an output")
    phase("restored_source_admission", lambda: cache.admit_checkout(repo, expected))
    record["source_changes"] = bool(cache.q.git(repo, "status", "--porcelain").strip())
    if record["source_changes"]:
        raise ValueError("measurement left changed source")
    log = (build / "build.log").read_text(errors="replace")
    record["cargo_finished_lines"] = re.findall(r"^.*Finished .*?$", log, re.M)
    record["total_measurement_s"] = time.monotonic() - started
    record["passed"] = True
    write(out / "timing.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--bwrap", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    measure(args.repo, args.out, args.nvcc, args.bwrap, args.jobs)


if __name__ == "__main__":
    main()
