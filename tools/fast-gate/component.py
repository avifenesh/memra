#!/usr/bin/env python3
"""Run or reuse isolated CPU component tests. Never native/model/release qualification.

The OS input view is the independence proof for this narrow contract: only declared
source bytes, a fingerprinted Python/Git runtime, and fresh temporary state are visible.
There is no unisolated fallback, external environment, network, GPU or host home.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICY = "memra-isolated-cpu-component-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_id(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"input is not a regular file: {path}")
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return {"sha256": value, "bytes": info.st_size, "mode": stat.S_IMODE(info.st_mode)}


def source_inputs(repo, contract):
    names = sorted(set(contract["inputs"] + ["tools/fast-gate/" + n for n in
                   ("component.py", "component_worker.py", "dependencies.json")]))
    result = {}
    for name in names:
        path = repo / name
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError(f"source symlinks are not admitted: {name}")
        result[name] = file_id(path)
    return result


def runtime_inputs():
    # The full visible stdlib is hashed, including extension modules and bytecode.
    # No user/site packages are mounted. Symlinks in this tree are refused.
    if sys.platform != "linux" or Path(sys.executable).resolve().parent != Path("/usr/bin"):
        raise ValueError("component isolation currently requires Linux system Python in /usr/bin")
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    binaries = {"/usr/bin/python3": Path(sys.executable).resolve()}
    for name in ("git", "bwrap", "sh", "bash", "env", "date", "mktemp", "sort", "rm", "sed", "grep", "tr", "dirname"):
        found = shutil.which(name)
        if not found:
            raise ValueError(f"required runtime unavailable: {name}")
        binaries["/usr/bin/" + name] = Path(found).resolve()
    git_exec = Path(subprocess.check_output([str(binaries["/usr/bin/git"]), "--exec-path"], text=True).strip())
    binaries["/usr/lib/git-core/git-upload-pack"] = (git_exec / "git-upload-pack").resolve()
    files = {}
    libraries = {}
    extensions = []
    for root, dirs, entries in os.walk(stdlib):
        for name in dirs:
            if (Path(root) / name).is_symlink():
                raise ValueError("stdlib directory symlink is outside the admitted runtime")
        for name in entries:
            path = Path(root) / name
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not target.is_relative_to(stdlib):
                    libraries[str(target)] = target
                files[str(path)] = {"symlink": os.readlink(path), "target": file_id(path.resolve())}
            else:
                files[str(path)] = file_id(path)
            if path.suffix == ".so":
                extensions.append(path)
    for binary in [*binaries.values(), *extensions]:
        output = subprocess.run(["ldd", str(binary)], text=True, capture_output=True, check=True).stdout
        for name in re.findall(r"(?:=>\s+)?(/[^\s()]+)", output):
            libraries[name] = Path(name).resolve()
    for destination, path in {**binaries, **libraries}.items():
        files[destination] = file_id(path)
    return {"files": files, "stdlib": str(stdlib), "platform": platform.uname()._asdict()}, binaries, libraries


ENV = {"PATH": "/usr/bin", "HOME": "/tmp/home", "TMPDIR": "/tmp", "LANG": "C", "LC_ALL": "C",
       "PYTHONHASHSEED": "0", "GIT_EXEC_PATH": "/usr/lib/git-core", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
       "GIT_TEMPLATE_DIR": "/tmp/templates", "GIT_CONFIG_COUNT": "2",
       "GIT_CONFIG_KEY_0": "maintenance.auto", "GIT_CONFIG_VALUE_0": "false",
       "GIT_CONFIG_KEY_1": "gc.auto", "GIT_CONFIG_VALUE_1": "0"}


def sandbox(stage, runtime, binaries, libraries):
    command = [str(binaries["/usr/bin/bwrap"]), "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv", "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--dir", "/tmp/home", "--dir", "/tmp/templates"]
    command += ["--ro-bind", runtime["stdlib"], runtime["stdlib"]]
    for destination, original in sorted({**binaries, **libraries}.items()):
        command += ["--ro-bind", str(original), destination]
    command += ["--ro-bind", str(stage / "source"), "/source", "--bind", str(stage / "out"), "/out"]
    for name, value in sorted(ENV.items()):
        command += ["--setenv", name, value]
    command += ["--chdir", "/source", "--", "/usr/bin/python3", "-I", "-S", "-B",
                "-X", "pycache_prefix=/tmp/pycache", "/source/tools/fast-gate/component_worker.py"]
    return command


def valid_result(result, contract):
    roster = result.get("tests", [])
    return (result.get("pass") is True and type(result.get("run")) is int
            and result["run"] >= contract["min_tests"] and result["run"] == len(roster) == len(set(roster))
            and all(type(result.get(k)) is int and result[k] == 0 for k in
                    ("skipped", "failures", "errors", "expected_failures", "unexpected_successes")))


def verify(entry, identity, contract):
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError("receipt directory missing or a symlink")
    file_id(entry / "receipt.json")
    record = json.loads((entry / "receipt.json").read_text())
    if record["identity"] != identity or record["key"] != digest(identity) or record["exit_code"] != 0:
        raise ValueError("receipt identity/verdict mismatch")
    for name in ("run.log", "result.json"):
        if file_id(entry / name) != record["evidence"][name]:
            raise ValueError("receipt evidence changed: " + name)
    result = json.loads((entry / "result.json").read_text())
    if not valid_result(result, contract):
        raise ValueError("receipt is failed, skipped or vacuous")
    return record


def lookup(repo, cache, name):
    """Read-only plan decision. Unknown/corrupt evidence refuses, never grants reuse."""
    repo, cache = repo.resolve(), cache.resolve()
    contract = json.loads((repo / "tools/fast-gate/dependencies.json").read_text())["components"][name]
    identity = {"policy": POLICY, "component": name, "contract": contract,
                "source": source_inputs(repo, contract), "runtime": runtime_inputs()[0],
                "environment": ENV, "scope": contract["scope"]}
    key = digest(identity)
    entry = cache / key
    if (cache / (key + ".invalid")).exists():
        raise ValueError("a fresh shadow contradicted this receipt; repair the dependency contract")
    if not entry.exists():
        return {"decision": "run", "key": key, "receipt_reason": "no matching complete-input receipt"}
    verify(entry, identity, contract)
    return {"decision": "reuse", "key": key, "receipt": str(entry / "receipt.json"),
            "receipt_reason": "matching isolated inputs and verified non-vacuous evidence"}


def run(repo, cache, name, fresh=False):
    start = time.monotonic()
    repo, cache = repo.resolve(), cache.resolve()
    if cache.is_relative_to(repo):
        raise ValueError("component cache must be outside source checkout")
    config = json.loads((repo / "tools/fast-gate/dependencies.json").read_text())
    contract = config["components"][name]
    source = source_inputs(repo, contract)
    runtime, binaries, libraries = runtime_inputs()
    identity = {"policy": POLICY, "component": name, "contract": contract, "source": source,
                "runtime": runtime, "environment": ENV, "scope": contract["scope"]}
    key = digest(identity)
    entry = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / (key + ".invalid")).exists():
        raise ValueError("a fresh shadow contradicted this receipt; repair the dependency contract")
    fingerprint_s = time.monotonic() - start
    # Nonblocking exclusion: independent writers never share a mutable test view.
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if entry.exists() and not fresh:
            record = verify(entry, identity, contract)
            return {"decision": "reuse", "key": key, "receipt": str(entry / "receipt.json"),
                    "scope": contract["scope"], "fingerprint_s": fingerprint_s,
                    "execution_s": 0, "prior_execution_s": record["execution_s"],
                    "total_s": time.monotonic() - start}
        with tempfile.TemporaryDirectory(prefix=".component-", dir=cache) as temporary:
            stage = Path(temporary)
            (stage / "out").mkdir()
            for path, expected in source.items():
                target = stage / "source" / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / path, target)
                target.chmod(expected["mode"])
                if file_id(target) != expected:
                    raise ValueError("source changed during input-view creation")
            (stage / "out/contract.json").write_text(json.dumps(contract))
            execution_start = time.monotonic()
            with (stage / "out/run.log").open("wb") as log:
                process = subprocess.run(sandbox(stage, runtime, binaries, libraries), stdout=log,
                                         stderr=subprocess.STDOUT, timeout=180)
            execution_s = time.monotonic() - execution_start
            result_path = stage / "out/result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if process.returncode or not valid_result(result, contract):
                # Print the actual diagnostic; do not publish failed evidence as reusable.
                sys.stderr.write((stage / "out/run.log").read_text(errors="replace"))
                failed = cache / (key + ".failed-" + uuid.uuid4().hex)
                (stage / "out").rename(failed)
                if entry.exists():
                    (cache / (key + ".invalid")).write_text("Fresh execution contradicted a cached pass: " + str(failed) + "\n")
                raise ValueError("component failed, skipped, missing or vacuous; raw evidence: " + str(failed))
            if source_inputs(repo, contract) != source or runtime_inputs()[0] != runtime:
                raise ValueError("inputs changed during component execution; no receipt stored")
            record = {"schema": POLICY, "key": key, "identity": identity, "exit_code": 0,
                      "execution_s": execution_s, "fingerprint_s": fingerprint_s,
                      "evidence": {p: file_id(stage / "out" / p) for p in ("run.log", "result.json")}}
            (stage / "out/receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            verify(stage / "out", identity, contract)
            if not entry.exists():
                (stage / "out").rename(entry)
            else:
                verify(entry, identity, contract)
                # A shadow is new evidence. Preserve its own raw output without
                # replacing the immutable record used by the cache lookup.
                entry = cache / (key + ".shadow-" + uuid.uuid4().hex)
                (stage / "out").rename(entry)
            return {"decision": "run", "key": key, "receipt": str(entry / "receipt.json"),
                    "scope": contract["scope"], "fingerprint_s": fingerprint_s,
                    "execution_s": execution_s, "total_s": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true", help="shadow by executing despite an existing receipt")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.repo, args.cache, args.component, args.fresh), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"component: REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
