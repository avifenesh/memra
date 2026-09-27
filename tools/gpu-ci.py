#!/usr/bin/env python3
"""Transport a controlled build to a physical-card qualification job.

This adapter delegates every native verdict to the release qualification tools.
CPU orchestration tests and a successful build never constitute GPU qualification.
"""

import argparse
import atexit
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request


# Qualification imports must neither consume nor create ignored project bytecode.
# The native producer applies the same isolation when it takes over capture/seal.
_PY_CACHE = tempfile.TemporaryDirectory(prefix="memra-gpu-ci-python-")
atexit.register(_PY_CACHE.cleanup)
sys.pycache_prefix = _PY_CACHE.name
sys.dont_write_bytecode = True


BINARIES = ("kernel-check", "run-gen", "run-spec", "argmax-margin-probe", "memra-server", "tok-parity")
METADATA = ("source.json", "build.json", "build.log", "fetch.log", "oracle-manifest.json")
CAPSULE_FILES = set(METADATA) | {"target/release/" + name for name in BINARIES}
MAX_CAPSULE_BYTES = 16 * 1024**3
MAX_ORACLE_BYTES = 200 * 1024**3
REQUIRED_TOOLS = (
    "qualify-release.py", "release_qualification.py", "release_inputs.py", "release_input_view.py",
)


class Refused(RuntimeError):
    pass


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate(repo, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("candidate must be a full immutable Git SHA")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise Refused("checkout does not match the requested candidate")
    for name in REQUIRED_TOOLS:
        if not (repo / "tools" / name).is_file():
            raise Refused(f"content-bound qualification prerequisite missing: tools/{name}")


def descriptor(value):
    if not isinstance(value, dict) or set(value) != {"name", "url", "sha256", "size"}:
        raise Refused("artifact descriptors require name, URL, digest and size")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", value["name"]):
        raise Refused("artifact name must be one plain filename")
    url = urllib.parse.urlsplit(value["url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
        raise Refused("artifact URL must be HTTPS without embedded credentials")
    if not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise Refused("artifact needs a SHA-256 digest")
    if type(value["size"]) is not int or not 0 < value["size"] <= MAX_ORACLE_BYTES:
        raise Refused("artifact size is outside the admitted range")
    return value


def manifest(data):
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) != {"schema", "lease_wrapper", "oracles"}:
        raise Refused("invalid oracle manifest fields")
    if value["schema"] != "memra-gpu-ci-inputs-v1":
        raise Refused("unsupported oracle manifest schema")
    wrapper = descriptor(value["lease_wrapper"])
    if wrapper["name"] != "memra-gpu-run" or wrapper["size"] > 1024**2:
        raise Refused("unexpected physical-card lease wrapper")
    if not isinstance(value["oracles"], list) or not value["oracles"]:
        raise Refused("oracle inventory is empty")
    names = set()
    total = 0
    for item in value["oracles"]:
        descriptor(item)
        if not item["name"].endswith(".gguf") or item["name"] in names:
            raise Refused("oracle names must be unique GGUF filenames")
        names.add(item["name"])
        total += item["size"]
    if total > MAX_ORACLE_BYTES:
        raise Refused("oracle inventory exceeds the qualification disk budget")
    return value


def download(item, directory):
    descriptor(item)
    target = directory / item["name"]
    if target.exists() or target.is_symlink():
        raise Refused("artifact destination already exists")
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(item["url"], headers={"User-Agent": "memra-gpu-ci"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            if urllib.parse.urlsplit(response.url).scheme != "https":
                raise Refused("artifact redirected away from HTTPS")
            total = 0
            while block := response.read(1024**2):
                total += len(block)
                if total > item["size"]:
                    raise Refused("artifact exceeded its declared size")
                output.write(block)
        if total != item["size"] or sha256(temporary) != item["sha256"]:
            raise Refused("artifact bytes differ from the pinned descriptor")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def fetch_manifest(url, digest, out):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise Refused("oracle manifest needs a pinned SHA-256 digest")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise Refused("oracle manifest needs an HTTPS URL without embedded credentials")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "memra-gpu-ci"}), timeout=30) as response:
        if urllib.parse.urlsplit(response.url).scheme != "https":
            raise Refused("manifest redirected away from HTTPS")
        data = response.read(1024**2 + 1)
    if len(data) > 1024**2 or hashlib.sha256(data).hexdigest() != digest:
        raise Refused("oracle manifest digest or size mismatch")
    manifest(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def qualification(repo):
    sys.path.insert(0, str(repo / "tools"))
    spec = importlib.util.spec_from_file_location("release_qualification", repo / "tools/release_qualification.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_build(repo, build, commit):
    candidate(repo, commit)
    q = qualification(repo)
    source = json.loads((build / "source.json").read_text())
    record = json.loads((build / "build.json").read_text())
    if source["commit"] != commit:
        raise Refused("build capsule belongs to another candidate")
    q.validate_build(record, source, record["source"], q.Evidence(build))
    for name in BINARIES:
        path = build / "target/release" / name
        observed = {**q.file_identity(path), "format": "ELF-x86_64"}
        if observed != record["binaries"][name]:
            raise Refused(f"controlled build executable changed: {name}")
    manifest((build / "oracle-manifest.json").read_bytes())


def pack_build(repo, build, commit, out):
    verify_build(repo, build, commit)
    with tarfile.open(out, "w") as archive:
        for name in sorted(CAPSULE_FILES):
            path = build / name
            if path.is_symlink() or not path.is_file():
                raise Refused(f"capsule member is not a regular file: {name}")
            archive.add(path, arcname=name, recursive=False)


def unpack_build(archive_path, out):
    if archive_path.stat().st_size > MAX_CAPSULE_BYTES:
        raise Refused("build capsule exceeds its size ceiling")
    out.mkdir(parents=True, exist_ok=False)
    try:
        seen = set()
        total = 0
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if not member.isfile() or member.name not in CAPSULE_FILES or member.name in seen:
                    raise Refused("build capsule contains an unexpected member")
                total += member.size
                if total > MAX_CAPSULE_BYTES:
                    raise Refused("build capsule expanded beyond its size ceiling")
                target = out / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                target.chmod(0o755 if member.name.startswith("target/release/") else 0o644)
                seen.add(member.name)
        if seen != CAPSULE_FILES:
            raise Refused("build capsule is missing required files")
    except BaseException:
        shutil.rmtree(out)
        raise


def allocation_check():
    cuda = ctypes.CDLL("libcuda.so.1")
    def check(code):
        if code:
            raise Refused(f"CUDA allocation preflight failed: driver error {code}")
    check(cuda.cuInit(0))
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0))
    context = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device))
    try:
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(4096)))
        check(cuda.cuMemFree_v2(pointer))
    finally:
        check(cuda.cuCtxDestroy_v2(context))


def sealed_commit(evidence):
    """Extract the commit bound by a sealed source descriptor, never a workflow echo."""
    record = json.loads((evidence / "record.json").read_text())
    source_ref = record.get("source", {})
    if record.get("status") != "qualified" or source_ref.get("path") != "source.json":
        raise Refused("sealed record has no canonical source descriptor")
    source_path = evidence / "source.json"
    if source_path.is_symlink() or sha256(source_path) != source_ref.get("sha256"):
        raise Refused("sealed source descriptor does not match its bytes")
    commit = json.loads(source_path.read_text()).get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("sealed source has no immutable commit")
    return commit


def capture(repo, build, commit, out, inputs):
    verify_build(repo, build, commit)
    inventory = manifest((build / "oracle-manifest.json").read_bytes())
    required = sum(item["size"] for item in inventory["oracles"])
    if shutil.disk_usage(inputs.parent).free < required + 10 * 1024**3:
        raise Refused("insufficient disk for the pinned oracle inventory and working reserve")
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if len(devices) != 1:
        raise Refused("generic qualification requires one physical card")
    index, gpu, name = (part.strip() for part in devices[0].split(",", 2))
    if index != "0" or not gpu.startswith("GPU-") or "RTX PRO 6000" not in name:
        raise Refused("generic qualification requires physical GPU0 on RTX PRO 6000")
    allocation_check()
    inputs.mkdir(parents=True, exist_ok=False)
    wrapper = download(inventory["lease_wrapper"], inputs)
    wrapper.chmod(0o755)
    oracles = inputs / "oracles"
    oracles.mkdir()
    for item in inventory["oracles"]:
        download(item, oracles)
    lease = out.parent / (out.name + "-lease")
    if out.exists() or lease.exists():
        raise Refused("capture and lease destinations must be fresh")
    subprocess.run([
        str(wrapper), "--gpus", gpu, "--receipt", str(lease), "--",
        sys.executable, str(repo / "tools/qualify-release.py"), "capture",
        "--repo", str(repo), "--expected-head", commit, "--build", str(build),
        "--out", str(out), "--oracles", str(oracles),
    ], cwd=repo, check=True)
    # Seal only after the physical-card wrapper has finalized its cleanup record.
    subprocess.run([
        sys.executable, str(repo / "tools/qualify-release.py"), "seal",
        "--repo", str(repo), "--out", str(out), "--lease", str(lease / "lease.json"),
        "--oracles", str(oracles),
    ], cwd=repo, check=True)
    q = qualification(repo)
    record = json.loads((out / "record.json").read_text())
    q.validate_record(record, q.Evidence(out), repo, commit)
    reported_commit = sealed_commit(out)
    if reported_commit != commit:
        raise Refused("sealed evidence identifies a different candidate")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"qualified-candidate={reported_commit}\n")
    print(f"SEALED GPU qualification: {commit}; exact binaries and native evidence retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "pack-build", "unpack-build", "capture"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--manifest-url")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    candidate(args.repo, args.candidate)
    if args.command == "preflight":
        if not args.manifest_url or not args.manifest_sha256:
            raise Refused("a pinned coordinator oracle/wrapper manifest is required before building or renting")
        fetch_manifest(args.manifest_url, args.manifest_sha256, args.out)
    elif args.command == "pack-build":
        if args.build is None:
            raise Refused("--build is required")
        pack_build(args.repo, args.build, args.candidate, args.out)
    elif args.command == "unpack-build":
        if args.archive is None:
            raise Refused("--archive is required")
        unpack_build(args.archive, args.out)
        verify_build(args.repo, args.out, args.candidate)
    else:
        if args.build is None or args.inputs is None:
            raise Refused("--build and --inputs are required")
        capture(args.repo, args.build, args.candidate, args.out, args.inputs)


if __name__ == "__main__":
    try:
        main()
    except (Refused, subprocess.CalledProcessError) as error:
        raise SystemExit(f"REFUSED: {error}") from None
