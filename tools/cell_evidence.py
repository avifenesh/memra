#!/usr/bin/env python3
"""Per-cell device evidence: produce it, bank it, verify it, and gate a push on it.

A change needs evidence only for the cells tools/change_impact.py says it reaches, and only
on build targets whose `push_gate` is true (tools/change-impact.toml). One record is one
run on one physical device of one build target, for one exact source-input digest:

    research/release-qualification/cells/<name>/record.json   + its raw logs

That namespace is metadata (excluded from the source-input digest), so committing the
record does not change what it qualifies. Evidence kinds:

    battery   tools/release-battery.sh on the device roster (kernel-check, run-gen argmax,
              run-spec K=1..8), raw cell logs re-judged here by tools/release-coverage.py
    serving   tools/serve-smoke.sh, judged by its own summary line
    perf      run-gen decode tok/s, base vs candidate interleaved A B B A A B per roster
              model; REGRESSED when every candidate run is slower than every base run
              (complete separation: one-sided exact p = 1/20 for 3 runs per arm)

Commands:

    python3 tools/cell_evidence.py need --base origin/main --head HEAD
    python3 tools/cell_evidence.py run --device rtx5090 --commit HEAD --kinds battery,perf \\
        --base origin/main --out ~/.cache/memra-cells/run-1
    python3 tools/cell_evidence.py bank --out ~/.cache/memra-cells/run-1
    python3 tools/cell_evidence.py check --base origin/main --head HEAD

Records are auditable native-run evidence, not signed attestations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import change_impact as ci  # noqa: E402

SCHEMA = "memra-cell-evidence-v1"
CELLS = "research/release-qualification/cells/"
KINDS = ("battery", "serving", "perf")
BINARIES = ("kernel-check", "run-gen", "run-spec", "argmax-margin-probe", "memra-server", "tok-parity")
MANIFESTS = ("tools/kernel-check-27b.cells", "tools/kernel-check-step35.cells")
BUILD_COMMAND = ["cargo", "build", "--release", "--locked",
                 "-p", "memra-engine", "--bin", "kernel-check", "--bin", "run-gen", "--bin", "run-spec",
                 "--bin", "argmax-margin-probe", "-p", "memra-server", "--bin", "memra-server",
                 "-p", "memra-tokenizer", "--bin", "tok-parity"]
# Rig build policy: one build at a time, low priority, fat LTO with one codegen unit.
BUILD_ENV = {"CARGO_BUILD_JOBS": "2", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
             "CARGO_PROFILE_RELEASE_LTO": "fat"}
BUILD_LOCK = Path.home() / ".cache/memra-rig-build.lock"
PERF_ORDER = ("A", "B", "B", "A", "A", "B")
PERF_NGEN = "128"
TPS = re.compile(r"generated (\d+) tokens in ([0-9.]+)s = ([0-9.]+) tok/s")
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class EvidenceError(ValueError):
    pass


class DeviceBusy(EvidenceError):
    """The device lock was not obtained; nothing ran, so nothing passed or failed."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if check and result.returncode != 0:
        raise EvidenceError("git " + " ".join(args[:3]) + ": " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def resolve(repo, ref):
    value = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    require(COMMIT.fullmatch(value), f"cannot resolve {ref}")
    return value


def inputs_digest(repo, ref):
    import release_qualification as q
    return q.source_snapshot(repo, ref)["inputs_sha256"]


def coverage_module():
    spec = importlib.util.spec_from_file_location("release_coverage", HERE / "release-coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roster_ids(data):
    rows = []
    for line in data.splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split("\t")
            require(len(fields) >= 3 and fields[0] in ("own", "vendor"), "malformed roster")
            rows.append((fields[0], fields[1]))
    require(any(kind == "own" for kind, _ in rows), "roster has no own model")
    return rows


# ------------------------------------------------------------------- judging raw logs

def judge_battery(runs_tsv, read_log, manifests, roster):
    """Raw battery evidence -> verdict rows, or EvidenceError. Mirrors cell_verdicts()."""
    coverage = coverage_module()
    required = set()
    for data in manifests:
        required.update(l.split("#", 1)[0].strip() for l in data.splitlines() if l.split("#", 1)[0].strip())
    require(required, "required-cell census is empty")
    expected = {("kernel", "kernel")} | {(kind, mid) for _, mid in roster for kind in ("argmax", "spec")}
    seen, verdicts = set(), []
    for line in runs_tsv.splitlines():
        if not line.strip():
            continue
        kind, model, code, name = line.split("\t")
        key = (kind, model)
        require(key in expected and key not in seen, f"unexpected or duplicated battery cell {key}")
        require(code == "0", f"battery cell {kind}/{model} exited {code}")
        seen.add(key)
        lines = [l.strip() for l in read_log(name).splitlines() if l.strip()]
        if kind == "kernel":
            verdict = coverage.kernel_coverage(lines, required)
        elif kind == "spec":
            verdict = coverage.spec_coverage(lines)
        else:
            require(sum(l.startswith("PASS:") for l in lines) == 1
                    and any(re.search(r"SUMMARY flips=\d+ bad=0\b", l) for l in lines)
                    and not any("SKIP" in l or l.startswith("FAIL") for l in lines),
                    f"argmax cell {model} did not pass its calibrated gate")
            verdict = "calibrated argmax PASS"
        verdicts.append({"kind": kind, "model": model, "verdict": verdict})
    missing = sorted(expected - seen)
    require(not missing, "battery cells missing: " + ", ".join(f"{k}/{m}" for k, m in missing))
    return verdicts


def judge_serving(log):
    lines = [l.strip() for l in log.splitlines() if l.strip()]
    require(lines, "empty serve-smoke log")
    require(not any(l.startswith("serve-smoke: SKIP") for l in lines), "serve-smoke skipped itself")
    require(lines[-1] == "serve-smoke: 0 failed", "serve-smoke failed: " + lines[-1])
    return "serve-smoke PASS"


def decode_tps(log):
    found = TPS.findall(log)
    require(found, "perf run printed no decode rate")
    return float(found[-1][2])


def judge_perf(runs):
    """runs: [{"arm","model","tps"}] -> per-model verdicts. Complete separation = regression."""
    out = {}
    for model in sorted({r["model"] for r in runs}):
        a = [r["tps"] for r in runs if r["model"] == model and r["arm"] == "A"]
        b = [r["tps"] for r in runs if r["model"] == model and r["arm"] == "B"]
        require(len(a) == len(b) == PERF_ORDER.count("A"), f"perf arms incomplete for {model}")
        delta = (statistics.median(b) / statistics.median(a) - 1.0) * 100.0
        regressed = max(b) < min(a)
        out[model] = {"base_median": statistics.median(a), "candidate_median": statistics.median(b),
                      "delta_pct": round(delta, 2), "regressed": regressed}
    return out


# ------------------------------------------------------------------- record verification

class TreeEvidence:
    def __init__(self, repo, ref, root):
        self.repo, self.ref, self.root = repo, ref, root.rstrip("/")

    def read(self, name):
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name, "unsafe evidence path")
        return git(self.repo, "show", f"{self.ref}:{self.root}/{name}")


class DirEvidence:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, name):
        path = (self.root / name)
        require(path.resolve().is_relative_to(self.root) and not path.is_symlink(), "unsafe evidence path")
        return path.read_bytes()


def verify_record(record, evidence, repo, candidate, rules, arch, candidate_inputs):
    """Validate one record against the candidate. Returns (target, set(kinds)) it proves."""
    require(record.get("schema") == SCHEMA and record.get("status") == "pass", "record is not a passing cell record")
    payloads = record.get("payloads")
    require(isinstance(payloads, dict) and payloads, "record has no payload manifest")
    for name, expected in payloads.items():
        require(isinstance(expected, str) and SHA.fullmatch(expected), "invalid payload digest")
        require(digest(evidence.read(name)) == expected, f"evidence changed: {name}")

    def payload(name):
        require(name in payloads, f"unmanifested evidence reference: {name}")
        return evidence.read(name).decode(errors="replace")

    require(record["inputs_sha256"] == candidate_inputs, "record is for different source inputs")
    tested = record["tested_commit"]
    require(isinstance(tested, str) and COMMIT.fullmatch(tested), "invalid tested commit")
    if tested != candidate:
        require(git(repo, "cat-file", "-t", tested, check=False).strip() == b"commit",
                f"tested commit {tested[:12]} is not present; fetch it")
        require(inputs_digest(repo, tested) == record["inputs_sha256"], "tested commit inputs differ from the record")
    device = record["device"]
    row = rules.get("device", {}).get(device.get("class"))
    require(row is not None, f"unknown device class {device.get('class')}")
    require(row["name"] in device["name"] and device["compute_cap"] == row["compute_cap"],
            "observed device does not match its class")
    target = record["target"]
    require(arch.target_for_cap(device["compute_cap"]) == target and record["build"]["target"] == target,
            "device, build target and record target disagree")
    proved = set()
    kinds = record.get("kinds", {})
    if "battery" in kinds:
        battery = kinds["battery"]
        roster = roster_ids(payload(battery["roster"]))
        tree_roster = roster_ids(git(repo, "show", f"{candidate}:tools/release-roster.tsv").decode())
        require(sorted(roster) == sorted(tree_roster), "battery roster models differ from tools/release-roster.tsv")
        manifests = [git(repo, "show", f"{candidate}:{m}").decode() for m in MANIFESTS]
        verdicts = judge_battery(payload(battery["runs"]), lambda n: payload("battery/" + n), manifests, roster)
        require(verdicts == battery["verdicts"], "battery verdicts differ from raw evidence")
        proved.add("battery")
    if "serving" in kinds:
        require(judge_serving(payload(kinds["serving"]["log"])) == kinds["serving"]["verdict"],
                "serving verdict differs from raw evidence")
        proved.add("serving")
    if "perf" in kinds:
        perf = kinds["perf"]
        if perf.get("baseline") == "none":
            base = perf.get("base_commit")
            require(isinstance(base, str) and COMMIT.fullmatch(base)
                    and target not in ci.arch_table_at(repo, base).accepted,
                    "perf without a baseline is admitted only when the base cannot build this target")
            for run in perf["runs"]:
                decode_tps(payload(run["log"]))
        else:
            base = perf["base_commit"]
            require(isinstance(base, str) and COMMIT.fullmatch(base), "perf record has no base commit")
            require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", base, candidate],
                                   capture_output=True).returncode == 0, "perf base is not an ancestor of the candidate")
            runs = [{"arm": r["arm"], "model": r["model"], "tps": decode_tps(payload(r["log"]))} for r in perf["runs"]]
            verdicts = judge_perf(runs)
            require(verdicts == perf["verdicts"], "perf verdicts differ from raw evidence")
            regressed = sorted(m for m, v in verdicts.items() if v["regressed"])
            require(not regressed or perf.get("accepted_regression"),
                    "perf REGRESSED on " + ", ".join(regressed) + " (every candidate run slower than every base run)")
        proved.add("perf")
    return target, proved


def records_in_tree(repo, ref):
    listing = git(repo, "ls-tree", "-r", "--name-only", ref, "--", CELLS).decode().splitlines()
    return sorted(p for p in listing if p.endswith("/record.json") and p.count("/") == CELLS.count("/") + 1)


def full_records_in_tree(repo, ref):
    """Full content-bound release records (research/release-qualification/<name>/record.json)."""
    import release_qualification as q
    listing = git(repo, "ls-tree", "-r", "--name-only", ref, "--", q.PUBLICATION).decode().splitlines()
    depth = q.PUBLICATION.count("/") + 1
    return sorted(p for p in listing if p.endswith("/record.json") and p.count("/") == depth
                  and not p.startswith(CELLS))


def proven_full_records(repo, candidate, arch, log=print):
    """A sealed full release record is the device battery for its card's target."""
    import release_qualification as q
    proven = {}
    for path in full_records_in_tree(repo, candidate):
        root = path.rsplit("/", 1)[0]
        try:
            record = json.loads(git(repo, "show", f"{candidate}:{path}"))
            result = q.validate_record(record, q.Evidence(root, repo, candidate), repo, candidate)
            run = q.Evidence(root, repo, candidate)
            caps = {arch.target_for_cap(d["compute_cap"]) for d in
                    json.loads(run.read(record["run"]["path"]))["hardware"]["devices"]}
            require(len(caps) == 1 and None not in caps, "full record spans several targets")
        except (q.GateError, EvidenceError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
            log(f"  full record {root}: not admitted: {error}")
            continue
        target = caps.pop()
        proven.setdefault(target, set()).add("battery")
        log(f"  full record {root}: battery on {', '.join(result['card_models'])}")
    return proven


def proven_cells(repo, candidate, rules, arch, log=print):
    candidate_inputs = inputs_digest(repo, candidate)
    proven = proven_full_records(repo, candidate, arch, log)
    for path in records_in_tree(repo, candidate):
        root = path.rsplit("/", 1)[0]
        try:
            record = json.loads(git(repo, "show", f"{candidate}:{path}"))
            if record.get("inputs_sha256") != candidate_inputs:
                continue
            target, kinds = verify_record(record, TreeEvidence(repo, candidate, root), repo, candidate,
                                          rules, arch, candidate_inputs)
        except (EvidenceError, KeyError, TypeError, ValueError) as error:
            log(f"  record {root}: not admitted: {error}")
            continue
        proven.setdefault(target, set()).update(kinds)
        log(f"  record {root}: {ci.target_name(rules, target)} {','.join(sorted(kinds))} "
            f"on {record['device']['class']}")
    return proven


# ------------------------------------------------------------------------------ gate

def devices_for(rules, arch, target):
    return sorted(name for name, row in rules.get("device", {}).items()
                  if arch.target_for_cap(row["compute_cap"]) == target)


def gate(repo, base, oid, log=print):
    """The push decision for one ref. Raises EvidenceError naming every unproven cell."""
    try:
        impact, rules, arch = ci.impact_between(repo, base, oid)
    except ci.ImpactError as error:
        raise EvidenceError(f"change impact cannot be classified: {error}") from error
    log(ci.render(impact))
    for reason in impact.get("declared_reasons", []):
        log(f"  Impact-Reason {reason}")
    needs = ci.requirements(impact)
    gated = {t: n for t, n in needs.items() if n["push_gate"]}
    for target, need in sorted(needs.items()):
        if not need["push_gate"]:
            log(f"  DEFERRED {need['name']}: {','.join(sorted(need['kinds']))} evidence comes from that "
                "target's daily regression and the pre-tag battery (push_gate = false)")
    if not gated:
        log(f"NO PUSH-GATED CELL: {oid[:12]} needs no device evidence at push")
        return impact
    proven = proven_cells(repo, oid, rules, arch, log)
    missing = []
    for target, need in sorted(gated.items()):
        lack = sorted(need["kinds"] - proven.get(target, set()))
        if lack:
            cells = ",".join(sorted(set(need["cells"])))
            missing.append(f"{need['name']} [{cells}] needs {'+'.join(lack)} evidence from one of: "
                           f"{', '.join(devices_for(rules, arch, target)) or 'no registered device'}")
    if missing:
        hint = (f"produce it: python3 tools/cell_evidence.py run --device <class> --commit {oid[:12]} "
                f"--base {base[:12] if set(base) != {'0'} else 'origin/main'} --kinds <kinds> --out <dir>, "
                "then `cell_evidence.py bank --out <dir>` and commit the record")
        raise EvidenceError("missing evidence for affected cells:\n    " + "\n    ".join(missing) + "\n  " + hint)
    log(f"CELL-QUALIFIED: {oid[:12]} " + "; ".join(f"{n['name']} {','.join(sorted(n['kinds']))}"
                                                   for _, n in sorted(gated.items())))
    return impact


# --------------------------------------------------------------------------- producing

def observe_device(row):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    raw = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=name,uuid,compute_cap,driver_version",
                                   "--format=csv,noheader"], text=True).strip().splitlines()[0]
    name, uuid, cap, driver = [f.strip() for f in raw.split(",")]
    require(row["name"] in name and cap == row["compute_cap"], f"this GPU ({name}, {cap}) is not a {row['name']}")
    return {"name": name, "uuid_sha256": digest(uuid.encode()), "compute_cap": cap, "driver": driver}


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")
           and k not in ("CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "LD_PRELOAD")}
    env.update(extra or {})
    return env


def checkout(repo, commit, path):
    if path.exists():
        git(repo, "worktree", "remove", "--force", str(path), check=False)
        if path.exists():
            shutil.rmtree(path)
    git(repo, "worktree", "add", "--detach", str(path), commit)
    return path


def build(repo, commit, target, work, log):
    """Build the six battery binaries for one commit and target; cached by commit."""
    cache = work / "bins" / f"{commit}-{target}"
    if all((cache / b).is_file() for b in BINARIES):
        return cache
    src = checkout(repo, commit, work / f"src-{commit[:12]}")
    env = clean_env({**BUILD_ENV, "MEMRA_CUDA_ARCH": target,
                     "CARGO_TARGET_DIR": str(work / f"target-{target}")})
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    command = ["flock", str(BUILD_LOCK), "nice", "-n", "15", *BUILD_COMMAND]
    with open(log, "ab") as out:
        out.write(f"$ {' '.join(command)}  (commit {commit}, target {target})\n".encode())
        out.flush()
        result = subprocess.run(command, cwd=src, env=env, stdout=out, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"build failed for {commit[:12]} (see {log})")
    cache.mkdir(parents=True, exist_ok=True)
    for name in BINARIES:
        shutil.copy2(work / f"target-{target}" / "release" / name, cache / name)
    git(repo, "worktree", "remove", "--force", str(src), check=False)
    return cache


def toolchain():
    """Compiler versions of the build environment, for the record (not a qualification input)."""
    out = {}
    for name, command in (("rustc", ["rustc", "-V"]), ("nvcc", ["nvcc", "--version"])):
        try:
            text = subprocess.run(command, capture_output=True, text=True, env=clean_env()).stdout.strip()
            out[name] = text.splitlines()[-1] if text else "unavailable"
        except OSError:
            out[name] = "unavailable"
    return out


def stage(repo, commit, bins, path):
    src = checkout(repo, commit, path)
    release = src / "target/release"
    release.mkdir(parents=True)
    for name in BINARIES:
        shutil.copy2(bins / name, release / name)
    return src


def run_battery(src, roster, out, env):
    with open(out / "battery.log", "wb") as log:
        result = subprocess.run(["bash", "tools/release-battery.sh", "--roster", str(roster),
                                 "--evidence-dir", str(out / "battery")],
                                cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def run_serving(src, row, out, env):
    args = [a for a in (row.get("serve_model"), row.get("serve_draft")) if a]
    with open(out / "serving.log", "wb") as log:
        return subprocess.run(["bash", "tools/serve-smoke.sh", *args], cwd=src, env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def run_perf(sources, roster_rows, out, env):
    """sources: {"A": src or None, "B": src}. Interleaved per model in PERF_ORDER."""
    (out / "perf").mkdir()
    prompt = out / "perf" / "prompt.txt"
    prompt.write_text("The quick brown fox jumps over the lazy dog. " * 40)
    runs, seq = [], 0
    arms = PERF_ORDER if sources.get("A") is not None else ("B", "B", "B")
    for _, model_id, path in roster_rows:
        if not Path(path).is_file():
            continue
        for arm in arms:
            seq += 1
            name = f"perf/{seq:02d}-{arm}-{re.sub(r'[^a-z0-9.-]+', '_', model_id.lower())}.log"
            with open(out / name, "wb") as log:
                subprocess.run([str(sources[arm] / "target/release/run-gen"), path], cwd=sources[arm],
                               env={**env, "MEMRA_NGEN": PERF_NGEN, "MEMRA_PROMPT_FILE": str(prompt)},
                               stdout=log, stderr=subprocess.STDOUT)
            runs.append({"arm": arm, "model": model_id, "log": name})
    return runs


def roster_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split("\t")
            rows.append((fields[0], fields[1], fields[2]))
    return rows


def seal(out, meta):
    """Judge raw evidence, manifest every file and write record.json."""
    evidence = DirEvidence(out)
    kinds, failures = {}, []
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "record.json")
    if "battery" in meta["kinds"]:
        entry = {"roster": "roster.tsv", "runs": "battery/runs.tsv", "exit_code": meta["battery_exit"]}
        try:
            manifests = [(Path(meta["src"]) / m).read_text() for m in MANIFESTS]
            roster = roster_ids((out / "roster.tsv").read_text())
            entry["verdicts"] = judge_battery((out / "battery/runs.tsv").read_text(),
                                              lambda n: evidence.read("battery/" + n).decode(errors="replace"),
                                              manifests, roster)
        except (EvidenceError, OSError, ValueError) as error:
            failures.append(f"battery: {error}")
        if meta["battery_exit"] != 0 and not any(f.startswith("battery") for f in failures):
            failures.append(f"battery: release-battery.sh exited {meta['battery_exit']} (see battery.log)")
        kinds["battery"] = entry
    if "serving" in meta["kinds"]:
        entry = {"log": "serving.log", "exit_code": meta["serving_exit"]}
        try:
            entry["verdict"] = judge_serving((out / "serving.log").read_text(errors="replace"))
        except (EvidenceError, OSError) as error:
            failures.append(f"serving: {error}")
        if meta["serving_exit"] != 0 and not any(f.startswith("serving") for f in failures):
            failures.append(f"serving: serve-smoke exited {meta['serving_exit']}")
        kinds["serving"] = entry
    if "perf" in meta["kinds"]:
        entry = {"base_commit": meta.get("base"), "runs": meta["perf_runs"],
                 "baseline": "interleaved" if meta.get("perf_baseline") else "none",
                 "accepted_regression": meta.get("accept_regression")}
        try:
            if entry["baseline"] == "interleaved":
                runs = [{**r, "tps": decode_tps((out / r["log"]).read_text(errors="replace"))} for r in meta["perf_runs"]]
                entry["verdicts"] = judge_perf(runs)
                regressed = sorted(m for m, v in entry["verdicts"].items() if v["regressed"])
                if regressed and not entry["accepted_regression"]:
                    failures.append("perf: REGRESSED on " + ", ".join(regressed))
            else:
                entry["absolute"] = {r["log"]: decode_tps((out / r["log"]).read_text(errors="replace"))
                                     for r in meta["perf_runs"]}
            require(meta["perf_runs"], "no roster model present for the perf arm")
        except (EvidenceError, OSError) as error:
            failures.append(f"perf: {error}")
        kinds["perf"] = entry
    record = {"schema": SCHEMA, "status": "fail" if failures else "pass", "failures": failures,
              "tested_commit": meta["commit"], "inputs_sha256": meta["inputs"], "target": meta["target"],
              "device": {"class": meta["device_class"], **meta["device"]},
              "build": {"target": meta["target"], "env": BUILD_ENV, "binaries": meta["binaries"],
                        "toolchain": meta.get("toolchain", {})},
              "kinds": kinds, "started": meta["started"], "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
              "payloads": {name: digest((out / name).read_bytes()) for name in files}}
    (out / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def produce(repo, commit, device_class, kinds, out, *, base=None, work=None, accept_regression=None,
            lock=True, log=print):
    """Build, run the requested kinds on this device under its lock, seal. Returns the record."""
    commit = resolve(repo, commit)
    arch = ci.arch_table_at(repo, commit)
    rules = ci.rules_at(repo, commit, arch)
    row = rules["device"].get(device_class)
    require(row is not None, f"unknown device class {device_class}")
    target = arch.target_for_cap(row["compute_cap"])
    require(set(kinds) <= set(KINDS) and kinds, f"kinds must be a subset of {KINDS}")
    work = Path(work or Path.home() / ".cache/memra-cells" / device_class)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = {"commit": commit, "inputs": inputs_digest(repo, commit), "target": target, "device_class": device_class,
            "kinds": list(kinds), "started": dt.datetime.now(dt.timezone.utc).isoformat(),
            "accept_regression": accept_regression}
    log(f"cell evidence: {device_class} ({ci.target_name(rules, target)}) {','.join(kinds)} at {commit[:12]}")
    bins = build(repo, commit, target, work, out / "build.log")
    meta["binaries"] = {n: {"bytes": (bins / n).stat().st_size, "sha256": digest((bins / n).read_bytes())}
                        for n in BINARIES}
    meta["toolchain"] = toolchain()
    base_bins = None
    if "perf" in kinds and base:
        meta["base"] = resolve(repo, base)
        if target in ci.arch_table_at(repo, meta["base"]).accepted:
            base_bins = build(repo, meta["base"], target, work, out / "build.log")
    plan = out / "gpu-phase.json"
    plan.write_text(json.dumps({**meta, "repo": str(repo), "work": str(work), "row": row,
                                "bins": str(bins), "base_bins": str(base_bins) if base_bins else None}))
    command = [sys.executable, str(Path(__file__).resolve()), "gpu-phase", "--plan", str(plan)]
    if lock:
        command = [*row["lock"], *command]
    log("  $ " + " ".join(command))
    result = subprocess.run(command)
    started = out / "gpu-phase.started"
    if not started.exists():
        plan.unlink()
        raise DeviceBusy(f"device lock not obtained ({' '.join(row['lock'][:1])} exit {result.returncode}); nothing ran")
    started.unlink()
    meta = json.loads(plan.read_text())
    plan.unlink()
    require(result.returncode == 0 and "device" in meta, f"device phase did not finish (exit {result.returncode})")
    record = seal(out, meta)
    for path in (meta.get("src"), meta.get("base_src")):
        if path:
            git(repo, "worktree", "remove", "--force", path, check=False)
    log(f"  record: {out / 'record.json'} status={record['status']} {'; '.join(record['failures'])}")
    return record


def gpu_phase(plan_path):
    """Runs under the device lock. Writes its observations back into the plan file."""
    plan = Path(plan_path)
    meta = json.loads(plan.read_text())
    repo, work, out, row = Path(meta["repo"]), Path(meta["work"]), plan.parent, meta["row"]
    (out / "gpu-phase.started").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    meta["device"] = observe_device(row)
    env = clean_env(row.get("env"))
    src = stage(repo, meta["commit"], Path(meta["bins"]), work / f"run-{meta['commit'][:12]}")
    meta["src"] = str(src)
    roster = src / row.get("roster", "tools/release-roster.tsv")
    shutil.copy2(roster, out / "roster.tsv")
    if "battery" in meta["kinds"]:
        meta["battery_exit"] = run_battery(src, roster, out, env)
    if "serving" in meta["kinds"]:
        meta["serving_exit"] = run_serving(src, row, out, env)
    if "perf" in meta["kinds"]:
        sources = {"B": src, "A": None}
        if meta.get("base_bins"):
            sources["A"] = stage(repo, meta["base"], Path(meta["base_bins"]), work / f"run-{meta['base'][:12]}")
            meta["base_src"] = str(sources["A"])
        meta["perf_baseline"] = sources["A"] is not None
        meta["perf_runs"] = run_perf(sources, roster_rows(roster), out, env)
    after = observe_device(row)
    require(after == meta["device"], "device observation changed during the run")
    plan.write_text(json.dumps(meta))
    return 0


def bank(repo, out, name=None):
    out = Path(out)
    record = json.loads((out / "record.json").read_text())
    require(record["status"] == "pass", "refusing to bank a failed record; keep it under research/ as a receipt")
    name = name or f"{record['tested_commit'][:12]}-{record['device']['class']}-{'-'.join(sorted(record['kinds']))}"
    require(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name), "invalid record name")
    dest = Path(repo) / CELLS / name
    require(not dest.exists(), f"{dest} exists; records are immutable")
    for rel in [*record["payloads"], "record.json"]:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / rel, target)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("need", "check"):
        p = sub.add_parser(mode)
        p.add_argument("--repo", type=Path, default=HERE.parent)
        p.add_argument("--base", default="origin/main")
        p.add_argument("--head", default="HEAD")
    p = sub.add_parser("run")
    p.add_argument("--repo", type=Path, default=HERE.parent)
    p.add_argument("--device", required=True)
    p.add_argument("--commit", default="HEAD")
    p.add_argument("--kinds", default="battery")
    p.add_argument("--base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path)
    p.add_argument("--accept-regression", help="record a known, reviewed perf regression and why")
    p.add_argument("--no-lock", action="store_true", help="the caller already holds the device lock")
    p = sub.add_parser("bank")
    p.add_argument("--repo", type=Path, default=HERE.parent)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name")
    p = sub.add_parser("gpu-phase")
    p.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "gpu-phase":
            return gpu_phase(args.plan)
        if args.mode in ("need", "check"):
            base = git(args.repo, "merge-base", args.base, args.head).decode().strip()
            head = resolve(args.repo, args.head)
            if args.mode == "check":
                gate(args.repo, base, head)
            else:
                impact, _, _ = ci.impact_between(args.repo, base, head)
                print(ci.render(impact))
                for _, need in sorted(ci.requirements(impact).items()):
                    print(f"  {need['name']}: {','.join(sorted(need['kinds']))} "
                          f"({'push-gated' if need['push_gate'] else 'daily/pre-tag'})")
            return 0
        if args.mode == "run":
            record = produce(args.repo, args.commit, args.device, args.kinds.split(","), args.out,
                             base=args.base, work=args.work, accept_regression=args.accept_regression,
                             lock=not args.no_lock)
            return 0 if record["status"] == "pass" else 1
        print(bank(args.repo, args.out, args.name))
        return 0
    except (EvidenceError, ci.ImpactError, OSError, subprocess.SubprocessError) as error:
        print(f"UNQUALIFIED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
