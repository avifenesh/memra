#!/usr/bin/env python3
"""Content-bound release qualification. CPU validation is not GPU execution."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from check_hardware_gate import GateError, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = "research/release-qualification/"
POINTER = PUBLICATION + "current.json"
BINARIES = ("kernel-check", "run-gen", "run-spec", "argmax-margin-probe", "memra-server", "tok-parity")
MANIFESTS = ("tools/kernel-check-27b.cells", "tools/kernel-check-step35.cells")
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def object_digest(value):
    return digest(canonical(value))


def json_bytes(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key {key}")
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=unique)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def commit(repo, ref):
    value = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    require(COMMIT.fullmatch(value), "unresolved source commit")
    return value


def safe_path(name):
    require(isinstance(name, str) and name and "\\" not in name, "invalid evidence path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ("..", ".") for p in path.parts),
            "evidence path escapes its directory")
    require(str(path) == name, "evidence path must be canonical")
    return name


def tree_files(repo, ref):
    files = {}
    for item in git(repo, "ls-tree", "-rz", "--full-tree", ref).split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = raw_path.decode()
        safe_path(path)
        require(kind == "blob" and mode in ("100644", "100755", "120000"),
                f"unsupported source entry {path}: {mode} {kind}")
        files[path] = {"mode": mode, "blob": oid}
    return files


def publication_metadata(path):
    return (path == "research/INDEX.md" or path == PUBLICATION.rstrip("/")
            or path.startswith(PUBLICATION))


def source_symlink_targets(repo, head, files):
    """Resolve file and directory links using only the immutable Git tree."""
    links = {path: git(repo, "show", f"{head}:{path}").decode()
             for path, item in files.items() if item["mode"] == "120000"}
    directories = {""}
    for path in files:
        directories.update(str(parent) for parent in PurePosixPath(path).parents if str(parent) != ".")
    resolved = {}
    for path in links:
        input_link = not publication_metadata(path)
        pending, parts, followed = deque(path.split("/")), [], 0
        while pending:
            component = pending.popleft()
            if component in ("", "."):
                continue
            if component == "..":
                require(parts, f"source symlink escapes tracked closure: {path}")
                parts.pop()
                continue
            prefix = "/".join([*parts, component])
            require(not input_link or not publication_metadata(prefix),
                    f"source input symlink crosses publication metadata: {path} via {prefix}")
            if prefix in links:
                target = links[prefix]
                followed += 1
                require(followed <= 40, f"source symlink cycle or excessive chain: {path}")
                require(target and not target.startswith("/"),
                        f"source symlink escapes tracked closure: {path}")
                pending.extendleft(reversed(target.split("/")))
            else:
                require(prefix in files or prefix in directories,
                        f"source symlink target is not tracked: {path}")
                require(not pending or prefix in directories,
                        f"source symlink traverses a non-directory: {path}")
                parts.append(component)
        destination = "/".join(parts)
        if input_link and destination in directories:
            # A directory alias can expose excluded children without naming them
            # in its own target. Refuse ancestors even before metadata is added.
            require(destination and not any(p.startswith(destination + "/") for p in
                    (PUBLICATION.rstrip("/"), "research/INDEX.md")),
                    f"source input symlink exposes publication metadata subtree: {path}")
        resolved[path] = destination
    return resolved


def source_snapshot(repo, ref="HEAD"):
    head = commit(repo, ref)
    files = tree_files(repo, head)
    source_symlink_targets(repo, head, files)
    # All tracked inputs are conservative dependencies, including research data consumed by
    # include_str!/build scripts. Only this reserved evidence namespace is metadata-only.
    inputs = {p: v for p, v in files.items() if not publication_metadata(p)}
    require(inputs, "empty source inventory")
    index = (git(repo, "show", f"{head}:research/INDEX.md").decode()
             if "research/INDEX.md" in files else "")
    return {"commit": head, "tree": git(repo, "rev-parse", f"{head}^{{tree}}").decode().strip(),
            "files": inputs, "inputs_sha256": object_digest(inputs), "index_prefix": index}


def verify_source(source, repo, head):
    require(COMMIT.fullmatch(source["commit"]) and COMMIT.fullmatch(source["tree"]),
            "invalid tested source identity")
    require(isinstance(source["files"], dict) and source["files"] and object_digest(source["files"]) == source["inputs_sha256"],
            "source inventory digest mismatch")
    actual = source_snapshot(repo, head)
    changed = sorted(p for p, value in source["files"].items() if actual["files"].get(p) != value)
    added = sorted(set(actual["files"]) - set(source["files"]))
    # An added research sidecar did not exist in the tested build. Preserve every
    # existing input byte/mode, refuse additions to runtime/tool/build namespaces,
    # and explicitly report the research additions for publication review.
    changed += [p for p in added if not p.startswith("research/")]
    require(not changed, "UNQUALIFIED: source inputs changed: " + ", ".join(changed[:20]))
    require(actual["index_prefix"].startswith(source["index_prefix"]),
            "UNQUALIFIED: research index changed beyond an append-only publication")
    # The pinned Git object is independent source provenance, not an optional label.
    # A shallow/squashed checkout must fetch it; file timestamps cannot replace it.
    present = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", source["commit"] + "^{commit}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    require(present, "tested source object missing; fetch the pinned commit or use --fetch-source")
    require(source_snapshot(repo, source["commit"]) == source, "tested source snapshot was altered")
    return {"tested_commit": source["commit"], "candidate_commit": actual["commit"],
            "inputs_sha256": source["inputs_sha256"],
            "publication_equivalent": actual["commit"] != source["commit"],
            "publication_additions": added}


def file_identity(path):
    path = Path(path)
    require(path.is_file(), f"input missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_identity(value, label):
    require(isinstance(value, dict) and type(value.get("bytes")) is int and value["bytes"] > 0
            and isinstance(value.get("sha256"), str) and SHA.fullmatch(value["sha256"]),
            f"invalid {label} content identity")


def read_roster(data):
    rows = []
    for line in data.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        require(len(fields) >= 3 and fields[0] in ("own", "vendor") and all(fields[:3]),
                "malformed release roster")
        rows.append({"class": fields[0], "id": fields[1], "path": fields[2]})
    require(rows and any(r["class"] == "own" for r in rows), "roster must include an own model")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate roster model")
    return rows


class Evidence:
    """Read a banked record from immutable Git blobs, or an unbanked capture directory."""
    def __init__(self, root, repo=None, head=None):
        self.root, self.repo, self.head = root, repo, head
        self.files = tree_files(repo, head) if repo is not None else None
        self.payloads = None

    def read(self, name):
        name = safe_path(name)
        if self.repo is not None:
            path = f"{self.root}/{name}"
            mode = self.files.get(path, {}).get("mode")
            require(mode in ("100644", "100755"), f"evidence missing or symlinked: {path}")
            return git(self.repo, "show", f"{self.head}:{path}")
        root = Path(self.root).resolve()
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "evidence symlink escape")
        return path.read_bytes()

    def bound(self, reference):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"} and SHA.fullmatch(reference["sha256"]),
                "invalid evidence reference")
        if self.payloads is not None:
            require(self.payloads.get(reference["path"]) == reference["sha256"], "unmanifested evidence reference")
        data = self.read(reference["path"])
        require(digest(data) == reference["sha256"], f"evidence changed: {reference['path']}")
        return data

    def obj(self, reference):
        value = json_bytes(self.bound(reference))
        require(isinstance(value, dict), "evidence metadata must be a JSON object")
        return value


def coverage_module():
    spec = importlib.util.spec_from_file_location("release_coverage", ROOT / "tools/release-coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_physical_lease(lease, run):
    """Validate a closed physical-card lease without selecting a model's rig.

    This is receipt validation, not proof that a lock is currently held. Capture
    must verify live wrapper ancestry and FLOCK ownership; callers must also bind
    the source, binary, numerical environment and model-specific hardware scope.
    Requested device order defines CUDA ordinals; locks use sorted UUID order.
    """
    import math

    ids = lease["requested_uuids"]
    require(type(ids) is list and ids
            and all(isinstance(x, str) and GPU_UUID.fullmatch(x) for x in ids)
            and len(set(ids)) == len(ids),
            "invalid physical GPU set")
    require(all(type(lease[k]) is int and lease[k] > 1 for k in ("wrapper_pid", "child_pid"))
            and lease["wrapper_pid"] != lease["child_pid"], "invalid lease process identities")
    require(lease["lock_order"] == sorted(ids) and lease["lock_files"] ==
            {x: f"/tmp/memra-gpu-locks/{x}.lock" for x in ids}, "lease physical locks mismatch")
    require(lease["state"] == "finished"
            and all(type(lease[k]) is int and lease[k] == 0 for k in ("exit_code", "child_exit_code"))
            and lease["timed_out"] is False and lease["interrupted_signal"] is None
            and lease["lingering_compute"] == [], "native lease did not finish cleanly")
    require(all(type(run["lease_owner"].get(k)) is int for k in ("wrapper_pid", "child_pid"))
            and run["lease_owner"] == {k: lease[k] for k in ("wrapper_pid", "child_pid", "requested_uuids")},
            "run and completed lease identities differ")
    times = (lease["started_unix"], run["started_unix"], run["finished_unix"], lease["finished_unix"])
    require(all(type(t) in (int, float) and t >= 0
                and (type(t) is int or math.isfinite(t)) for t in times),
            "invalid lease/run timestamps")
    require(lease["started_unix"] <= run["started_unix"] < run["finished_unix"] <= lease["finished_unix"],
            "run is outside the completed lease interval")
    devices = run["hardware"]["devices"]
    require([d["uuid"] for d in devices] == ids, "hardware differs from leased GPU set")
    require([d["uuid"] for d in lease["devices"]] == ids and
            [d["name"] for d in lease["devices"]] == [d["name"] for d in devices],
            "hardware observation and lease differ")
    indices = [d["index"] for d in lease["devices"]]
    require(all(type(index) is int and index >= 0 for index in indices)
            and len(set(indices)) == len(indices)
            and [d["index"] for d in devices] == [str(index) for index in indices],
            "physical GPU indices differ or are duplicated")
    require(run["numeric_environment"].get("CUDA_VISIBLE_DEVICES") == digest(",".join(ids).encode()),
            "CUDA visibility does not bind the leased physical UUIDs in order")


def validate_lease(lease, run):
    """The generic battery's existing single-card GPU0 / PRO6000 profile."""
    validate_physical_lease(lease, run)
    ids = lease["requested_uuids"]
    devices = run["hardware"]["devices"]
    require(len(ids) == 1, "generic release battery currently qualifies one physical card; use a separate Step gate for topology claims")
    require(run["hardware"]["headroom_query"] == {"nvml_index": 0, "uuid": ids[0]}
            and devices[0]["index"] == "0" and lease["devices"][0]["index"] == 0,
            "CUDA UUID and battery NVML GPU0 headroom selection differ")
    require(all("RTX PRO 6000 Blackwell" in d["name"] and d["compute_cap"] == "12.0"
                and d["driver_version"] for d in devices), "wrong release rig/architecture")


def validate_build(build, source, source_reference, evidence):
    import release_input_view as view
    require(build["schema"] == "memra-native-build-v3" and build["exit_code"] == 0
            and build["source"] == source_reference and evidence.obj(source_reference) == source
            and build["cuda_arch"] == "120a"
            and build["docs_rs"] is False and build["cuda_visible_devices"] == ""
            and build["rustc"] and build["nvcc"] and build["command"], "invalid native build provenance")
    recipe = build["recipe"]
    require(recipe["policy"] == "controlled-cargo-v3" and recipe["cargo_home"] == "fresh-config-free"
            and recipe["checkout"] == view.POLICY
            and recipe["build_source"] == "fingerprinted-input-view"
            and recipe["cargo_config"] == "tracked-jobs-only", "uncontrolled native build recipe")
    require(recipe["sandbox"]["policy"] == view.SANDBOX_POLICY
            and recipe["sandbox"]["version"].startswith("bubblewrap "), "compiler filesystem isolation missing")
    validate_identity(recipe["sandbox"]["executable"], "compiler sandbox")
    require(build["input_view_before"] == build["input_view_after"] == view.identity(source),
            "compiler input view changed or does not match the source")
    require(isinstance(build["compiler_environment"], dict)
            and all(isinstance(v, str) and SHA.fullmatch(v) for v in build["compiler_environment"].values()),
            "invalid compiler environment identities")
    for name, value in {"CUDA_VISIBLE_DEVICES": "", "MEMRA_CUDA_ARCH": "120a", "CARGO_HOME": "/cargo",
                        "CARGO_TARGET_DIR": "/target", "RUSTC": "/toolchain/bin/rustc"}.items():
        require(build["compiler_environment"].get(name) == digest(value.encode()),
                f"uncontrolled compiler environment: {name}")
    require(set(recipe["compilers"]) == {"cargo", "rustc", "nvcc"}, "missing compiler identities")
    for name, identity in recipe["compilers"].items():
        validate_identity(identity, name)
    require(build["source_before"] == build["source_after"] == source["inputs_sha256"],
            "build source changed during compilation")
    require(build["platform"]["machine"] == "x86_64" and build["platform"]["profile"]
            and build["platform"]["glibc"], "native build platform missing")
    evidence.bound(build["log"])
    evidence.bound(build["fetch_log"])
    require(set(build["binaries"]) == set(BINARIES), "native binary inventory incomplete")
    for name, value in build["binaries"].items():
        validate_identity(value, name)
        require(value.get("format") == "ELF-x86_64", f"non-native binary: {name}")


def validate_historical_record(record, evidence, repo, head, binaries=None, models=None, hardware=None):
    require(isinstance(record, dict), "record must be a JSON object")
    require(record.get("schema") == "memra-release-qualification-v1" and record.get("status") == "qualified",
            "UNQUALIFIED: absent, failed or unsupported qualification record")
    require(isinstance(record.get("payloads"), dict) and record["payloads"], "missing evidence manifest")
    evidence.payloads = record["payloads"]
    for path, expected in record["payloads"].items():
        require(isinstance(expected, str) and SHA.fullmatch(expected), "invalid payload digest")
        require(digest(evidence.read(path)) == expected, f"evidence changed: {path}")
    source = evidence.obj(record["source"])
    build = evidence.obj(record["build"])
    run = evidence.obj(record["run"])
    lease = evidence.obj(record["lease"])
    proof = verify_source(source, repo, head)
    validate_build(build, source, record["source"], evidence)
    require(run["schema"] == "memra-native-release-run-v1" and run["exit_code"] == 0,
            "battery did not exit successfully")
    require(run["source_before"] == run["source_after"] == source["inputs_sha256"],
            "run source changed")
    require(run["binaries_before"] == run["binaries_after"] == build["binaries"],
            "stale or replaced runtime binary")
    require(run["models_before"] == run["models_after"] and run["models_before"],
            "model bytes changed during qualification")
    for value in run["models_before"].values():
        validate_identity(value, "model")
    require(run["hardware"] == run["hardware_after"], "rig/topology changed during qualification")
    require(run["numeric_environment"] == run["numeric_environment_after"], "numerical environment changed")
    require(all(SHA.fullmatch(v) for v in run["numeric_environment"].values()), "invalid numerical environment digest")
    validate_lease(lease, run)
    topology = evidence.bound(run["topology"])
    require(topology.strip(), "missing topology observation")
    require(run["hardware"]["topology_sha256"] == digest(topology), "topology hash mismatch")
    require(run["command"] in (["bash", "tools/release-battery.sh", "--evidence-dir", "cells"],
                               ["bash", "tools/release-battery.sh", "--generic-only", "--evidence-dir", "cells"]),
            "noncanonical battery command")
    evidence.bound(run["battery_log"])
    samples = evidence.bound(run["telemetry"]).decode().splitlines()
    require(len(samples) >= 3 and "uuid" in samples[0], "missing native telemetry samples")
    import csv
    rows = list(csv.reader(samples[1:], skipinitialspace=True))
    require(all(len(row) >= 7 and row[1] in lease["requested_uuids"] for row in rows),
            "telemetry does not identify the leased physical GPU")
    verdicts = cell_verdicts(run, evidence, repo, head)
    require(record["verdicts"] == verdicts, "record verdict/cell/skip summary differs from raw evidence")
    identity = {"source": source["inputs_sha256"], "build": record["build"]["sha256"],
                "models": run["models_before"], "numeric": run["numeric_environment"],
                "hardware": run["hardware"], "verdicts": verdicts}
    require(record["identity_sha256"] == object_digest(identity), "qualification identity changed")
    if binaries is not None:
        for name in BINARIES:
            actual = file_identity(Path(binaries) / name)
            require(all(actual[k] == build["binaries"][name][k] for k in actual), f"stale binary: {name}")
    if models is not None:
        require(models == run["models_before"], "stale model inventory")
    if hardware is not None:
        require(hardware == run["hardware"], "stale rig/topology")
    return {**proof, "identity_sha256": record["identity_sha256"], "verdicts": verdicts,
            "binaries": build["binaries"], "build_profile": build["platform"]["profile"],
            "qualification": "historical-generic-only"}


def validate_record(record, evidence, repo, head, binaries=None, models=None, hardware=None):
    """Full release v2. A valid historical generic v1 is never a release waiver."""
    import copy
    from serving_run import validate_serving_run, tracked, MANIFEST
    require(isinstance(record, dict) and record.get("schema") == "memra-release-qualification-v2"
            and record.get("status") == "qualified", "UNQUALIFIED: full release requires sealed v2 serving evidence; v1 is generic-only")
    require(set(record) == {"schema", "status", "source", "build", "generic", "serving", "payloads", "verdicts", "identity_sha256"},
            "unknown or missing v2 release fields")
    require(isinstance(record["payloads"], dict) and record["payloads"], "missing v2 evidence manifest")
    evidence.payloads = record["payloads"]
    for path, expected in record["payloads"].items():
        require(isinstance(expected, str) and SHA.fullmatch(expected) and digest(evidence.read(path)) == expected,
                "v2 evidence changed: " + path)
    generic = evidence.obj(record["generic"])
    require(generic.get("source") == record["source"] and generic.get("build") == record["build"], "generic and serving build/source differ")
    proof = validate_historical_record(generic, copy.copy(evidence), repo, head, binaries, models, hardware)
    source, build = evidence.obj(record["source"]), evidence.obj(record["build"])
    serving = validate_serving_run(evidence.obj(record["serving"]), evidence, tracked(repo, head, MANIFEST),
        {"repo": repo, "head": head, "source": source, "source_reference": record["source"],
         "build": build, "build_reference": record["build"]})
    generic_models = evidence.obj(generic["run"])["models_before"]
    require(all(path not in generic_models or generic_models[path] == identity for path, identity in serving["models"].items()),
            "generic and serving model artifacts differ")
    verdicts = {"generic": proof["verdicts"], "serving": serving}
    require(record["verdicts"] == verdicts, "v2 verdicts differ from raw required evidence")
    identity = {"source": source["inputs_sha256"], "build": record["build"]["sha256"],
                "generic": record["generic"]["sha256"], "serving": record["serving"]["sha256"], "verdicts": verdicts}
    require(record["identity_sha256"] == object_digest(identity), "v2 release identity changed")
    return {**proof, "identity_sha256": record["identity_sha256"], "verdicts": verdicts,
            "qualification": "native-required-release-evidence-validated"}


def verify_historical_published(repo, head):
    """Read old raw records explicitly; never called by push, tag or release gates."""
    results = [validate_historical_record(record, evidence, repo, resolved)
               for record, evidence, resolved in published_records(repo, head)]
    require(results, "missing historical generic evidence")
    return results[0]


def cell_verdicts(run, evidence, repo, head):
    roster_data = git(repo, "show", f"{head}:tools/release-roster.tsv")
    require(evidence.bound(run["roster"]) == roster_data, "stale or substituted roster")
    roster = read_roster(roster_data)
    require(all(row["path"] in run["models_before"] for row in roster), "missing roster model identity")
    oracle_dir = run["oracle_directory"]
    require(isinstance(oracle_dir, str) and Path(oracle_dir).is_absolute()
            and any(Path(p).parent == Path(oracle_dir) for p in run["models_before"]),
            "missing kernel oracle identities")
    require(run["numeric_environment"].get("MEMRA_KC_MODELS_DIR") == digest(oracle_dir.encode()),
            "kernel oracle directory differs from executed environment")
    required = set()
    for path in MANIFESTS:
        data = git(repo, "show", f"{head}:{path}")
        require(evidence.bound(run["manifests"][path]) == data, "required-cell manifest changed")
        required.update(line.split("#", 1)[0].strip() for line in data.decode().splitlines()
                        if line.split("#", 1)[0].strip())
    require(required, "required-cell census is empty")
    coverage = coverage_module()
    expected = {("kernel", "kernel")} | {(kind, row["id"]) for row in roster for kind in ("argmax", "spec")}
    seen, verdicts = set(), []
    for cell in run["cells"]:
        key = (cell["kind"], cell["model"])
        require(key in expected and key not in seen and cell["exit_code"] == 0,
                "missing, duplicated, unexpected or failed release cell")
        seen.add(key)
        lines = [line.strip() for line in evidence.bound(cell["log"]).decode().splitlines() if line.strip()]
        if cell["kind"] == "kernel":
            verdict = coverage.kernel_coverage(lines, required)
        elif cell["kind"] == "spec":
            verdict = coverage.spec_coverage(lines)
        else:
            require(sum(line.startswith("PASS:") for line in lines) == 1
                    and any(re.search(r"SUMMARY flips=\d+ bad=0\b", line) for line in lines)
                    and not any("SKIP" in line or line.startswith("FAIL") for line in lines),
                    "argmax cell did not pass its calibrated gate")
            verdict = "calibrated argmax PASS"
        verdicts.append({"kind": key[0], "model": key[1], "verdict": verdict})
    require(seen == expected, "required release cells are missing")
    return verdicts


def published_records(repo, head):
    head = commit(repo, head)
    require(tree_files(repo, head).get(POINTER, {}).get("mode") == "100644",
            "UNQUALIFIED: no committed native qualification pointer")
    pointer = json_bytes(git(repo, "show", f"{head}:{POINTER}"))
    entries = pointer.get("records", [pointer])
    require(isinstance(entries, list) and entries, "qualification index is empty")
    records, seen = [], set()
    for entry in entries:
        name = safe_path(entry["record"])
        require(name.startswith(PUBLICATION) and name.endswith("/record.json") and name not in seen,
                "invalid or duplicate qualification pointer")
        seen.add(name)
        evidence = Evidence(str(PurePosixPath(name).parent), repo, head)
        data = evidence.read("record.json")
        require(entry["sha256"] == digest(data), "qualification pointer digest mismatch")
        records.append((json_bytes(data), evidence, head))
    return records


def verify_published(repo, head, binaries=None, profile=None, fetch_source=False):
    results = []
    for record, evidence, resolved in published_records(repo, head):
        if fetch_source:
            source = evidence.obj(record["source"])
            tested = source["commit"]
            require(isinstance(tested, str) and COMMIT.fullmatch(tested), "invalid tested commit")
            present = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", tested + "^{commit}"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not present:
                subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", "origin", tested], check=True)
        results.append(validate_record(record, evidence, repo, resolved))
    # Every indexed record must be current. Separate OS builds can each have native
    # evidence without pretending their different ELF bytes are one qualified binary.
    candidates = [r for r in results if profile is None or r["build_profile"] == profile]
    require(candidates, f"UNQUALIFIED: no native build for profile {profile}")
    for result in candidates:
        if binaries is None or all(file_identity(Path(binaries) / name) ==
                {k: result["binaries"][name][k] for k in ("bytes", "sha256")} for name in BINARIES):
            return result
    raise GateError("UNQUALIFIED: rebuilt binaries match no qualified build")


def check_push(repo, refs, mode):
    require(mode in ("qualified", "development"), "unknown qualification mode")
    for line in refs:
        fields = line.split()
        require(len(fields) == 4, "invalid push ref input")
        local, oid, remote, _ = fields
        if set(oid) == {"0"}:
            continue
        protected = remote.startswith("refs/tags/") or remote in ("refs/heads/main", "refs/heads/master")
        if mode == "development":
            require(not protected, "UNQUALIFIED development mode cannot push main or tags")
            print(f"UNQUALIFIED DEVELOPMENT: {remote} at {oid}; no GPU qualification claimed")
        else:
            result = verify_published(repo, oid)
            print(f"QUALIFIED SOURCE: {remote} at {result['candidate_commit']} tested={result['tested_commit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "push"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--binaries", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--fetch-source", action="store_true", help="fetch a missing pinned tested Git commit from origin")
    parser.add_argument("--refs-file", type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "push":
            require(args.refs_file is not None, "push requires --refs-file")
            check_push(args.repo, args.refs_file.read_text().splitlines(),
                       "development" if args.development else "qualified")
        else:
            require(not args.development, "release verification has no development waiver")
            result = verify_published(args.repo, args.head, args.binaries, args.profile, args.fetch_source)
            print(json.dumps(result, sort_keys=True))
        return 0
    except (GateError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"UNQUALIFIED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
