#!/usr/bin/env python3
"""Per-cell device evidence: produce it, bank it, verify it, and gate a push on it.

A change needs evidence only for the cells tools/change_impact.py says it reaches, and only
on build targets whose `push_gate` is true (tools/change-impact.toml). One record is one
run on one physical device of one build target, for one exact source-input digest:

    research/release-qualification/cells/<name>/record.json   + its raw logs

That namespace is metadata (excluded from the source-input digest), so committing the
record does not change what it qualifies. A record also proves its cells for a later commit
that differs from its tested commit only in research or docs paths no crate or gate tool
reads (`research_drift`), so a research-only or docs-only commit on main does not void pending
landings.
Evidence kinds:

    battery   tools/release-battery.sh on the device roster (kernel-check, run-gen argmax,
              the served-batch argmax, run-spec K=1..8), raw cell logs re-judged here by
              tools/release-coverage.py and the tag checker's argmax rules
    serving   tools/serve-smoke.sh, judged by its own summary line
    perf      run-gen decode tok/s, base vs candidate interleaved A B B A A B per roster
              model; REGRESSED when every candidate run is slower than every base run
              (complete separation: one-sided exact p = 1/20 for 3 runs per arm) AND the
              candidate median is at least PERF_MIN_REGRESSION_PCT below the base median

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
import functools
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
# --quiet drops cargo's per-crate progress lines. The banked build.log is a public receipt,
# and some dependency names in those lines (a TLS crypto backend named after a cloud
# provider) trip the public-boundary provider rule. Warnings and errors still reach the log.
BUILD_COMMAND = ["cargo", "build", "--release", "--locked", "--quiet",
                 "-p", "memra-engine", "--bin", "kernel-check", "--bin", "run-gen", "--bin", "run-spec",
                 "--bin", "argmax-margin-probe", "-p", "memra-server", "--bin", "memra-server",
                 "-p", "memra-tokenizer", "--bin", "tok-parity"]
# Rig build policy: one build at a time, low priority, fat LTO with one codegen unit.
BUILD_ENV = {"CARGO_BUILD_JOBS": "2", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
             "CARGO_PROFILE_RELEASE_LTO": "fat"}
BUILD_LOCK = Path.home() / ".cache/memra-rig-build.lock"
PERF_ORDER = ("A", "B", "B", "A", "A", "B")
PERF_NGEN = "128"
# A perf regression needs complete separation AND a median drop of at least this much.
# Rank alone fires on identical code with p = 1/20 per model (about 1 in 10 heads with two
# roster models), and the daily run keeps its last pass as the base, so a false alarm stayed
# failing. The floor comes from the run-to-run noise in the banked records (the 6 records
# under research/release-qualification/cells/ and the first gb10 daily record; derivation in
# research/perf-verdict-noise-20261106/): per-run relative sd 0.23% on GB10 and 0.99% on the
# RTX 5090 laptop, single runs up to 2.5% off their arm mean. Resampling that noise as two
# arms of one program, rank alone flags 4.7% of models on the 5090; rank + 2% flags 0.32%
# (0.6% of two-model heads) and none on GB10. The largest median delta in the 10 recorded
# base vs candidate pairs is 1.3%. With 2%, a true 3% regression is caught 83% of the time
# on the 5090 and always on GB10; 5% is caught at least 99% on both.
PERF_MIN_REGRESSION_PCT = 2.0
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


def git(repo, *args, check=True, input=None):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input)
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
    # tools/release-battery.sh runs three cells per roster model: the eager argmax gate, the
    # served-batch argmax gate (`argmax-margin-gate.sh --decode batch`) and run-spec K=1..8.
    expected = {("kernel", "kernel")} | {(kind, mid) for _, mid in roster
                                         for kind in ("argmax", "argmax-batch", "spec")}
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
        elif kind == "argmax-batch":
            # The tag checker's rule: the calibrated margin verdict plus exactly one
            # SERVED-SUMMARY line, MATCH (the batched program at B=1 sampled the host argmax of
            # every row) or EAGER-ROUTED (the scheduler decodes this model eagerly).
            served = [l for l in lines if l.startswith("SERVED-SUMMARY served=")]
            require(sum(l.startswith("PASS:") for l in lines) == 1
                    and any(re.search(r"SUMMARY flips=\d+ bad=0\b", l) for l in lines)
                    and served in (["SERVED-SUMMARY served=MATCH"], ["SERVED-SUMMARY served=EAGER-ROUTED"])
                    and not any("SKIP" in l or l.startswith("FAIL") for l in lines),
                    f"served-batch argmax cell {model} did not pass its calibrated gate")
            verdict = ("calibrated served-batch argmax PASS" if served == ["SERVED-SUMMARY served=MATCH"]
                       else "calibrated served-decode argmax PASS (eager-routed: decode_step serves)")
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
    """runs: [{"arm","model","tps"}] -> per-model verdicts. A regression needs both complete
    separation and a median drop of at least PERF_MIN_REGRESSION_PCT."""
    out = {}
    for model in sorted({r["model"] for r in runs}):
        a = [r["tps"] for r in runs if r["model"] == model and r["arm"] == "A"]
        b = [r["tps"] for r in runs if r["model"] == model and r["arm"] == "B"]
        require(len(a) == len(b) == PERF_ORDER.count("A"), f"perf arms incomplete for {model}")
        delta = round((statistics.median(b) / statistics.median(a) - 1.0) * 100.0, 2)
        regressed = max(b) < min(a) and delta <= -PERF_MIN_REGRESSION_PCT
        out[model] = {"base_median": statistics.median(a), "candidate_median": statistics.median(b),
                      "delta_pct": delta, "regressed": regressed}
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


# Impact classes whose files a cell record may outlive: research receipts and documentation that no
# crate and no gate tool reads (a file either reads is classified code or gate, not these).
# .gitattributes is docs-class but can change the bytes a checkout gives the build (eol, filters),
# so it never drifts under a record.
DRIFT_CLASSES = ("research", "docs")
DRIFT_NEVER = (".gitattributes",)


@functools.lru_cache(maxsize=None)
def research_drift(repo, old, new):
    """The source inputs that differ between commits `old` and `new`, as a tuple, when every one
    of them is a research or documentation path the impact classifier calls research or docs (no
    crate reads it and no gate tool reads it); None when any differing input is anything else,
    or when the crate include census cannot answer at either commit.

    A record bound to `old` then still describes `new`: the compiled program, the battery, its
    manifests, the roster, the gate tools and their inputs are the same bytes. Without this, a
    research-only or docs-only commit on main voided every pending landing record (each lane
    rebases onto it, its inputs digest moves, and it re-runs about an hour of device time for
    bytes the build and the gate never see; a README edit on main did that to two landings on
    2026-11-08). The classifier decides (`change_impact.impact_between` under the rules in
    force), so a file a crate reads (the CI include census, e.g. build.rs reading docs/FLAGS.md)
    or a gate tool reads (a `gate` rule in tools/change-impact.toml) counts as code. When the
    census cannot answer, its literal fallback misses directory and runtime reads, so this fails
    closed.
    """
    import release_qualification as q
    for ref in (old, new):
        if not (isinstance(ref, str) and COMMIT.fullmatch(ref)):
            return None
        if git(repo, "cat-file", "-t", ref, check=False).strip() != b"commit":
            return None
    names = [n for n in git(repo, "diff", "-z", "--name-only", "--no-renames", old, new)
             .decode(errors="replace").split("\0") if n]
    if not _drift_names_may_pass(repo, old, names):
        return None
    impact, _, _ = ci.impact_between(repo, old, new)
    if any(item["class"] not in DRIFT_CLASSES or item["path"].rsplit("/", 1)[-1] in DRIFT_NEVER
           for item in impact["files"]):
        return None
    a = q.source_snapshot(repo, old)["files"]
    b = q.source_snapshot(repo, new)["files"]
    drift = tuple(sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p)))
    if not drift:
        return drift
    for ref in (old, new):
        inputs, owners = ci.crate_inputs_at(repo, ref, list(drift))
        if owners is None:
            return None
        if any(ci._input_match(p, pattern) for p in drift for pattern in inputs):
            return None
    return drift


def _drift_names_may_pass(repo, old, names):
    """A cheap necessary condition for `research_drift`: each changed name's first matching rule
    (the tested commit's) is research or docs. The full classifier can only strengthen a class (a
    crate read, a symlink), never weaken it, so a name this refuses can never drift; and it keeps
    a stale record from costing the census (about 13 s a pair) on every push."""
    text = ci.read_blob(repo, old, ci.RULES_PATH)
    if text is None:
        return False
    try:
        rules = ci.load_rules(text)
    except ci.ImpactError:
        return False
    for name in names:
        if name.startswith(ci.PUBLICATION) or name == "research/INDEX.md":
            continue  # qualification metadata, excluded from the inputs
        rule, _ = ci._match_rule(rules, name)
        if rule is None or rule["class"] not in DRIFT_CLASSES or name.rsplit("/", 1)[-1] in DRIFT_NEVER:
            return False
    return True


def drift_note(paths):
    shown = ", ".join(paths[:3]) + (f", ... ({len(paths)} paths)" if len(paths) > 3 else "")
    return f"only research or docs paths no crate or gate tool reads changed: {shown}"


def verify_record(record, evidence, repo, candidate, rules, arch, candidate_inputs, push_base=None):
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

    tested = record["tested_commit"]
    require(isinstance(tested, str) and COMMIT.fullmatch(tested), "invalid tested commit")
    if record["inputs_sha256"] != candidate_inputs:
        require(git(repo, "cat-file", "-t", tested, check=False).strip() == b"commit",
                f"record is for other source inputs and its tested commit {tested[:12]} is not present; "
                "fetch it to compare")
        require(research_drift(repo, tested, candidate) is not None,
                "record is for different source inputs (more changed since its tested commit than "
                "research or docs paths no crate or gate tool reads)")
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
        base = perf.get("base_commit")
        # The baseline is the push base's program: an older ancestor or a commit inside the
        # pushed range would compare against the wrong program. A base that differs from the
        # push base only in research or docs paths no crate reads is that same program.
        require(isinstance(base, str) and COMMIT.fullmatch(base) and isinstance(push_base, str)
                and (base == push_base or research_drift(repo, base, push_base) is not None),
                f"perf base {str(base)[:12]} is not this push's base {str(push_base)[:12]} (or "
                "differs from it in more than research or docs paths no crate or gate tool reads)")
        roster = {mid for _, mid in roster_ids(git(repo, "show", f"{candidate}:tools/release-roster.tsv").decode())}
        arms = {"none": {"B": 3}}.get(perf.get("baseline"), {"A": PERF_ORDER.count("A"), "B": PERF_ORDER.count("B")})
        counts = {}
        for run in perf["runs"]:
            counts[(run["model"], run["arm"])] = counts.get((run["model"], run["arm"]), 0) + 1
        require(counts == {(m, a): n for m in roster for a, n in arms.items()},
                "perf runs do not cover every roster model with every arm")
        if perf.get("baseline") == "none":
            require(target not in ci.arch_table_at(repo, base).accepted,
                    "perf without a baseline is admitted only when the base cannot build this target")
            for run in perf["runs"]:
                decode_tps(payload(run["log"]))
        else:
            runs = [{"arm": r["arm"], "model": r["model"], "tps": decode_tps(payload(r["log"]))} for r in perf["runs"]]
            verdicts = judge_perf(runs)
            require(verdicts == perf["verdicts"], "perf verdicts differ from raw evidence")
            regressed = sorted(m for m, v in verdicts.items() if v["regressed"])
            require(not regressed or perf.get("accepted_regression"),
                    "perf REGRESSED on " + ", ".join(regressed) + " (every candidate run slower than every base "
                    f"run, median at least {PERF_MIN_REGRESSION_PCT:g}% lower)")
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


def device_class_of(rules, name):
    found = [cls for cls, row in rules.get("device", {}).items() if row["name"] in name]
    return found[0] if len(found) == 1 else None


def proven_full_records(repo, candidate, rules, arch, log=print):
    """A sealed full release record is the device battery for its card's target."""
    import release_qualification as q
    proven, by_device = {}, {}
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
        for model in result["card_models"]:
            if device_class_of(rules, model):
                by_device.setdefault(device_class_of(rules, model), set()).add("battery")
        log(f"  full record {root}: battery on {', '.join(result['card_models'])}")
    return proven, by_device


def proven_cells(repo, candidate, rules, arch, log=print, push_base=None):
    """-> ({target: kinds}, {device class: kinds}) proven for the candidate's inputs: a record of
    the exact inputs, or of a commit that differs from the candidate in `research_drift` alone."""
    import release_qualification as q
    candidate_inputs = inputs_digest(repo, candidate)
    proven, by_device = proven_full_records(repo, candidate, rules, arch, log)
    for path in records_in_tree(repo, candidate):
        root = path.rsplit("/", 1)[0]
        try:
            record = json.loads(git(repo, "show", f"{candidate}:{path}"))
            across = None
            if record.get("inputs_sha256") != candidate_inputs:
                # Older records of other inputs are the normal case; only one whose tested commit
                # differs from the candidate in research or docs paths alone is still evidence for it.
                tested = record.get("tested_commit")
                if not (isinstance(tested, str) and COMMIT.fullmatch(tested)):
                    continue
                if git(repo, "cat-file", "-t", tested, check=False).strip() != b"commit":
                    log(f"  record {root}: tested commit {tested[:12]} is not present; fetch it to "
                        "check whether this record still applies")
                    continue
                across = research_drift(repo, tested, candidate)
                if across is None:
                    continue
            target, kinds = verify_record(record, TreeEvidence(repo, candidate, root), repo, candidate,
                                          rules, arch, candidate_inputs, push_base)
        except (EvidenceError, q.GateError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
            log(f"  record {root}: not admitted: {error}")
            continue
        proven.setdefault(target, set()).update(kinds)
        by_device.setdefault(record["device"]["class"], set()).update(kinds)
        notes = []
        if across:
            notes.append(f"tested {record['tested_commit'][:12]}; since then {drift_note(across)}")
        base = record.get("kinds", {}).get("perf", {}).get("base_commit")
        if "perf" in kinds and push_base and base and base != push_base:
            notes.append(f"perf base {base[:12]}: against push base {push_base[:12]}, "
                         + drift_note(research_drift(repo, base, push_base) or ()))
        log(f"  record {root}: {ci.target_name(rules, target)} {','.join(sorted(kinds))} "
            f"on {record['device']['class']}" + "".join(f" ({n})" for n in notes))
    return proven, by_device


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
    proven, by_device = proven_cells(repo, oid, rules, arch, log, push_base=base)
    missing = []
    for target, need in sorted(gated.items()):
        lack = sorted(need["kinds"] - proven.get(target, set()))
        cells = ",".join(sorted(set(need["cells"])))
        if lack:
            missing.append(f"{need['name']} [{cells}] needs {'+'.join(lack)} evidence from one of: "
                           f"{', '.join(devices_for(rules, arch, target)) or 'no registered device'}")
        for device in sorted(need["devices"]):
            lack = sorted({"battery", "perf"} - by_device.get(device, set()))
            if lack:
                missing.append(f"{need['name']} [defaults keyed on {device}] needs {'+'.join(lack)} "
                               f"evidence from {device} itself")
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


# A research path as code spells it: up to a quote, a space or a delimiter.
RESEARCH_PATH = re.compile(r"research/[^\s\"'`<>(),;]+")


@functools.lru_cache(maxsize=None)
def research_kept(repo, commit):
    """The research/ files a cell tree at `commit` checks out, as a tuple: those the impact
    rules at that commit classify as anything but research (a `gate` rule: a gate tool reads
    them), those the CI include census counts as crate inputs, and those under a research path
    a code line in crates/ spells (comment lines and data files read nothing, as in the census's
    literal check). The rest are receipts no build and no gate reads (the contract
    `research_drift` relies on too). None, so the tree is checked out whole, when the census
    cannot answer (it needs git 2.45 for `--no-lazy-fetch`) or a kept name cannot be written as
    a sparse pattern."""
    names = [n for n in git(repo, "ls-tree", "-r", "-z", "--name-only", commit, "--", "research")
             .decode("utf-8", errors="replace").split("\0") if n]
    text = ci.read_blob(repo, commit, ci.RULES_PATH)
    if text is None:
        return _whole("no impact rules at the commit")
    try:
        rules = ci.load_rules(text)
        import validation_plan as vp
        tree = vp.Tree(repo, commit)
        _, owners = vp.workspace(tree)
        census = [k for k in vp.included_inputs(tree, owners) if k.startswith("research/")]
    except (ImportError, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        return _whole(f"the include census cannot answer ({type(error).__name__}: {error})")
    grep = subprocess.run(["git", "-C", str(repo), "grep", "-n", "-I", "-F", "-e", "research/", commit,
                           "--", "crates"], capture_output=True)
    if grep.returncode not in (0, 1):
        return _whole("git grep failed: " + grep.stderr.decode(errors="replace").strip())
    spelled = set()
    for line in grep.stdout.decode(errors="replace").splitlines():
        _, source, _, code = line.split(":", 3)
        # The census's own literal rule: comment lines and data files read nothing.
        if ci._comment_line(code) or not ci._may_read(source, code):
            continue
        spelled.update(literal.rstrip(".") for literal in RESEARCH_PATH.findall(code))
    kept = []
    for name in names:
        rule, _ = ci._match_rule(rules, name)
        if ((rule is not None and rule["class"] != "research")
                or any(ci._input_match(name, pattern) for pattern in census)
                or any(name == p or name.startswith(p.rstrip("/") + "/") for p in spelled)):
            if "\n" in name or name != name.rstrip() or "\ufffd" in name:
                return _whole(f"a kept name cannot be a sparse pattern: {name!r}")
            kept.append(name)
    return tuple(kept)


def _whole(reason):
    print(f"  cell tree: whole checkout ({reason})", file=sys.stderr)
    return None


def sparse_patterns(names, kept):
    """Non-cone sparse-checkout patterns: everything but research/, plus the kept research files,
    as the highest directories whose tracked files are all kept (`/dir/**`) and single files
    otherwise. Per-file patterns cost minutes: git matches every path against every pattern."""
    total, held = {}, {}
    for name in names:
        for d in _research_dirs(name):
            total[d] = total.get(d, 0) + 1
    for name in kept:
        for d in _research_dirs(name):
            held[d] = held.get(d, 0) + 1
    patterns = set()
    for name in kept:
        full = [d for d in _research_dirs(name) if held.get(d) == total.get(d)]
        patterns.add(f"/{glob_escape(full[0])}/**" if full else "/" + glob_escape(name))
    return ["/*", "!/research/**", *sorted(patterns)]


def _research_dirs(name):
    """The directories of `name` below research/, outermost first."""
    parts = name.split("/")
    return ["/".join(parts[:i]) for i in range(2, len(parts))]


def glob_escape(path):
    return re.sub(r"([\\*?\[\]!#])", r"\\\1", path)


def tree_at(repo, commit, path):
    """A stable per-device worktree checked out clean at `commit`. Its in-tree target/ stays,
    so builds are incremental and the battery and serve-smoke (which runs its own no-op
    `cargo build`) use the very binaries that were hashed, at a fixed source path.

    The checkout is sparse: research/ receipts no build and no gate reads are left out
    (`research_kept`). They were 7.1 of the 7.96 GB a full checkout took on 2026-11-08, and
    each lane keeps two to four such trees. The record does not change: its inputs digest comes
    from git objects (`inputs_digest`), not from the checkout. A sparse cell tree is for building
    and running cells; the pre-push censuses read research/ and run on a full checkout."""
    kept = research_kept(repo, commit)
    if not (path / ".git").exists():
        if path.exists():
            shutil.rmtree(path)
        git(repo, "worktree", "prune")
        git(repo, "worktree", "add", "--no-checkout", "--detach", str(path), commit)
    # Sparse checkout in a worktree needs per-worktree config; set it once, not racing two runs.
    if git(repo, "config", "--get", "extensions.worktreeConfig", check=False).strip() != b"true":
        git(repo, "config", "extensions.worktreeConfig", "true")
    if kept is not None:
        names = [n for n in git(repo, "ls-tree", "-r", "-z", "--name-only", commit, "--", "research")
                 .decode("utf-8", errors="replace").split("\0") if n]
        patterns = "\n".join(sparse_patterns(names, kept)) + "\n"
        result = subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", "--no-cone", "--stdin"],
                                input=patterns.encode("utf-8"), capture_output=True)
        if result.returncode != 0:  # `set --no-cone` needs git 2.35
            kept = _whole("sparse-checkout set refused: " + result.stderr.decode(errors="replace").strip())
    if kept is None:
        git(path, "sparse-checkout", "disable")
    git(path, "checkout", "-q", "--detach", "--force", commit)
    git(path, "clean", "-qffdx", "-e", "/target")
    require(git(path, "rev-parse", "HEAD").decode().strip() == commit, f"{path} is not at {commit[:12]}")
    require(not git(path, "status", "--porcelain", "--untracked-files=normal").strip(), f"{path} is not clean")
    return path


def build_env(target, tree):
    return clean_env({**BUILD_ENV, "MEMRA_CUDA_ARCH": target, "CARGO_TARGET_DIR": str(tree / "target")})


def binary_ids(directory):
    return {n: {"bytes": (directory / n).stat().st_size, "sha256": digest((directory / n).read_bytes())}
            for n in BINARIES}


def shown_command(command):
    """A command as logged into banked evidence: the home directory reads `~`, so a published
    build.log names no user."""
    return " ".join(command).replace(str(Path.home()), "~")


def build(repo, commit, target, work, log, role="tree", bins=None):
    """Build the six battery binaries in work/<role>-<target> at `commit`; copy them to a
    per-commit cache under `bins` (default work/bins). Returns (tree, cache)."""
    tree = tree_at(repo, commit, work / f"{role}-{target}")
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    command = ["flock", str(BUILD_LOCK), "nice", "-n", "15", *BUILD_COMMAND]
    with open(log, "ab") as out:
        out.write(f"$ {shown_command(command)}  (commit {commit}, target {target}, tree {tree.name})\n".encode())
        out.flush()
        result = subprocess.run(command, cwd=tree, env=build_env(target, tree), stdout=out, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"build failed for {commit[:12]} (see {log})")
    cache = Path(bins or work / "bins") / f"{commit}-{target}"
    cache.mkdir(parents=True, exist_ok=True)
    for name in BINARIES:
        shutil.copy2(tree / "target/release" / name, cache / name)
    return tree, cache


def toolchain(tree, env):
    """Compiler versions the build used, for the record (not a qualification input). They
    run in the build tree with the build's environment, so rustup resolves the tree's
    rust-toolchain.toml as cargo did, not the caller's directory default."""
    out = {}
    for name, command in (("rustc", ["rustc", "-V"]), ("nvcc", ["nvcc", "--version"])):
        try:
            text = subprocess.run(command, cwd=tree, capture_output=True, text=True, env=env).stdout.strip()
            out[name] = text.splitlines()[-1] if text else "unavailable"
        except OSError:
            out[name] = "unavailable"
    return out


LOCK_VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def lock_command(device_class, row, environ=None):
    """The device row's lock prefix with each ${NAME} filled from the environment. One row
    can serve several boxes (the gb10 row serves two Sparks, each its own sparklock unit), so
    the box-specific part comes from a per-box override. An unset one refuses before anything
    builds or runs: there is no fallback lock."""
    environ = os.environ if environ is None else environ
    filled = []
    for arg in row["lock"]:
        missing = [name for name in LOCK_VARIABLE.findall(arg) if not environ.get(name)]
        require(not missing, f"device {device_class} lock needs {', '.join(missing)} (per box: the environment, "
                f"or ~/.config/memra-daily/{device_class}.env for the daily run); nothing ran")
        filled.append(LOCK_VARIABLE.sub(lambda m: environ[m.group(1)], arg))
    return filled


def run_battery(src, roster, out, env):
    with open(out / "battery.log", "wb") as log:
        result = subprocess.run(["bash", "tools/release-battery.sh", "--roster", str(roster),
                                 "--evidence-dir", str(out / "battery")],
                                cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def shown_bytes(line, src):
    """One line of banked output with the build tree as `<tree>` and the home directory as `~`
    (cargo names the crate path it compiles), so a published log names no user or mount."""
    return line.replace(str(src).encode(), b"<tree>").replace(str(Path.home()).encode(), b"~")


def run_serving(src, row, out, env):
    args = [a for a in (row.get("serve_model"), row.get("serve_draft")) if a]
    with open(out / "serving.log", "wb") as log:
        proc = subprocess.Popen(["bash", "tools/serve-smoke.sh", *args], cwd=src, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            log.write(shown_bytes(line, src))
        return proc.wait()


def run_perf(sources, roster_rows, out, env, cwd):
    """sources: {"A": run-gen path or None, "B": run-gen path}. Interleaved per model."""
    (out / "perf").mkdir()
    prompt = out / "perf" / "prompt.txt"
    # Ends mid-sentence so the next token is near-certain: run-gen asserts prefill == decode
    # argmax at the last position, and an open-ended prompt can be a documented near-tie.
    prompt.write_text("The quick brown fox jumps over the lazy dog. " * 40 + "The quick brown fox jumps over the lazy")
    runs, seq = [], 0
    arms = PERF_ORDER if sources.get("A") is not None else ("B", "B", "B")
    for _, model_id, path in roster_rows:
        for arm in arms:
            seq += 1
            name = f"perf/{seq:02d}-{arm}-{re.sub(r'[^a-z0-9.-]+', '_', model_id.lower())}.log"
            runs.append({"arm": arm, "model": model_id, "log": name})
            if not Path(path).is_file():
                # A missing roster model is a failed perf cell, never a silently shorter one.
                (out / name).write_text(f"perf: roster model file absent: {path}\n")
                continue
            with open(out / name, "wb") as log:
                subprocess.run([str(sources[arm]), path], cwd=cwd,
                               env={**env, "MEMRA_NGEN": PERF_NGEN, "MEMRA_PROMPT_FILE": str(prompt)},
                               stdout=log, stderr=subprocess.STDOUT)
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
    failures.extend(meta.get("binary_drift", []))
    record = {"schema": SCHEMA, "status": "fail" if failures else "pass", "failures": failures,
              "tested_commit": meta["commit"], "inputs_sha256": meta["inputs"], "target": meta["target"],
              "device": {"class": meta["device_class"], **meta["device"]},
              "build": {"target": meta["target"], "env": BUILD_ENV, "binaries": meta["binaries"],
                        "toolchain": meta.get("toolchain", {})},
              "kinds": kinds, "started": meta["started"], "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
              "payloads": {name: digest((out / name).read_bytes()) for name in files}}
    (out / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def produce(repo, commit, device_class, kinds, out, *, base=None, work=None, bins=None, accept_regression=None,
            lock=True, log=print, reference_perf=False):
    """Build, run the requested kinds on this device under its lock, seal. Returns the record.
    `bins` is the binary cache root (default work/bins); a caller that prunes its cache passes
    its own, so it never deletes what other runs left in a shared work dir."""
    commit = resolve(repo, commit)
    arch = ci.arch_table_at(repo, commit)
    rules = ci.rules_at(repo, commit, arch)
    row = rules["device"].get(device_class)
    require(row is not None, f"unknown device class {device_class}")
    target = arch.target_for_cap(row["compute_cap"])
    require(set(kinds) <= set(KINDS) and kinds, f"kinds must be a subset of {KINDS}")
    # Push evidence compares against the push base; only the daily run's first day may
    # record reference rates with no base (that record never satisfies the push gate).
    require("perf" not in kinds or base or reference_perf, "perf evidence needs --base (the push base)")
    lock_prefix = lock_command(device_class, row) if lock else []
    work = Path(work or Path.home() / ".cache/memra-cells" / device_class)
    work.mkdir(parents=True, exist_ok=True)
    bins = Path(bins or work / "bins")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    meta = {"commit": commit, "inputs": inputs_digest(repo, commit), "target": target, "device_class": device_class,
            "kinds": list(kinds), "started": dt.datetime.now(dt.timezone.utc).isoformat(),
            "accept_regression": accept_regression}
    log(f"cell evidence: {device_class} ({ci.target_name(rules, target)}) {','.join(kinds)} at {commit[:12]}")
    tree, head_bins = build(repo, commit, target, work, out / "build.log", bins=bins)
    meta["binaries"] = binary_ids(head_bins)
    meta["toolchain"] = toolchain(tree, build_env(target, tree))
    base_bins = None
    if "perf" in kinds and base:
        meta["base"] = resolve(repo, base)
        if target in ci.arch_table_at(repo, meta["base"]).accepted:
            base_bins = bins / f"{meta['base']}-{target}"
            if not all((base_bins / b).is_file() for b in BINARIES):
                _, base_bins = build(repo, meta["base"], target, work, out / "build.log", role="base", bins=bins)
    plan = out / "gpu-phase.json"
    plan.write_text(json.dumps({**meta, "repo": str(repo), "work": str(work), "row": row, "tree": str(tree),
                                "base_bins": str(base_bins) if base_bins else None}))
    command = [*lock_prefix, sys.executable, str(Path(__file__).resolve()), "gpu-phase", "--plan", str(plan)]
    log("  $ " + " ".join(command))
    result = subprocess.run(command)
    started = out / "gpu-phase.started"
    if not started.exists():
        plan.unlink()
        # gpulock, sparklock and `flock -w` exit 1 when the device stays held; any other code
        # (sparklock's 2 for an unknown unit, a missing lock command) is a configuration error
        # that a retry will not fix, so it must not read as "busy, the next run retries".
        if result.returncode != 1:
            raise EvidenceError(f"device lock command failed ({' '.join(lock_prefix[:1])} exit "
                                f"{result.returncode}); nothing ran")
        raise DeviceBusy(f"device lock not obtained ({' '.join(lock_prefix[:1])} exit {result.returncode}); nothing ran")
    started.unlink()
    meta = json.loads(plan.read_text())
    plan.unlink()
    require(result.returncode == 0 and "device" in meta, f"device phase did not finish (exit {result.returncode})")
    record = seal(out, meta)
    log(f"  record: {out / 'record.json'} status={record['status']} {'; '.join(record['failures'])}")
    return record


def gpu_phase(plan_path):
    """Runs under the device lock. Writes its observations back into the plan file."""
    plan = Path(plan_path)
    meta = json.loads(plan.read_text())
    out, row, tree = plan.parent, meta["row"], Path(meta["tree"])
    (out / "gpu-phase.started").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    meta["device"] = observe_device(row)
    meta["src"] = str(tree)
    require(git(tree, "rev-parse", "HEAD").decode().strip() == meta["commit"], "build tree moved")
    release = tree / "target/release"
    # Build settings stay in the environment, so serve-smoke's own `cargo build` is a no-op
    # on the hashed binary (and nice/two-job if it ever is not).
    env = {**build_env(meta["target"], tree), **row.get("env", {})}
    drift = []

    def unchanged(stage):
        if binary_ids(release) != meta["binaries"]:
            drift.append(f"binaries changed during {stage}; the tested bytes are not the hashed build")
    unchanged("staging")
    roster = tree / row.get("roster", "tools/release-roster.tsv")
    shutil.copy2(roster, out / "roster.tsv")
    if "battery" in meta["kinds"]:
        meta["battery_exit"] = run_battery(tree, roster, out, env)
        unchanged("battery")
    if "serving" in meta["kinds"]:
        meta["serving_exit"] = run_serving(tree, row, out, env)
        unchanged("serve-smoke")
    if "perf" in meta["kinds"]:
        sources = {"B": release / "run-gen",
                   "A": Path(meta["base_bins"]) / "run-gen" if meta.get("base_bins") else None}
        meta["perf_baseline"] = sources["A"] is not None
        meta["perf_runs"] = run_perf(sources, roster_rows(roster), out, env, tree)
        unchanged("perf")
    meta["binary_drift"] = drift
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
