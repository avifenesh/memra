#!/usr/bin/env python3
"""Classify a diff by what it can affect, per CUDA build target.

The output is the set of (build target, scope) cells a change needs evidence for. Rules
live in tools/change-impact.toml; the build targets and the compute-capability map come
from the engine's own arch table (crates/memra-engine/build.rs, src/cuda_arch.rs). The
classifier is pure: `classify()` takes parsed changes and returns cells. The git adapter
below only reads the diff, the rules and the arch table from immutable commits.

    python3 tools/change_impact.py --base origin/main --head HEAD
    python3 tools/change_impact.py --base origin/main --head HEAD --json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib

RULES_PATH = "tools/change-impact.toml"
BUILD_RS = "crates/memra-engine/build.rs"
CUDA_ARCH_RS = "crates/memra-engine/src/cuda_arch.rs"
SCHEMA = "memra-change-impact-v1"
# Cell scopes. `perf` is a cell like the others; it asks for performance evidence.
CORRECTNESS = ("kernel", "build", "numeric", "serving", "defaults", "gate")
SCOPES = CORRECTNESS + ("perf",)
NO_CELL = ("docs", "tests", "research", "cpu", "diagnostic", "comment")
UNKNOWN = "unknown"
# An unknown path reaches everything a served binary can contain.
UNKNOWN_SCOPES = ("kernel", "build", "numeric", "serving", "perf")
CLASSES = SCOPES + NO_CELL + (UNKNOWN,)
# Classes whose files compile into the served program: perf-relevant by package default and
# subject to device-key content detection.
PROGRAM = ("kernel", "build", "numeric", "serving")
# An Impact-Targets trailer can narrow program code, never the build, the gate or the unknown.
UNNARROWED = ("build", "gate", UNKNOWN)
ZERO = "0" * 40
PUBLICATION = "research/release-qualification/"
# Weakest to strongest; a path read by several crates takes the strongest class.
# The release battery's binaries (tools/cell_evidence.py BINARIES: kernel-check, run-gen, run-spec
# and argmax-margin-probe are memra-engine targets, tok-parity a memra-tokenizer one) and the perf
# A/B (run-gen) can move only with code linked into those packages: the packages and their
# workspace dependencies. memra-server sits above the engine (it depends on the engine, never the
# reverse), so a change to its code leaves every battery binary as it was; serve-smoke is the one
# test that runs that code. Its manifest is different: see EVIDENCE_BUILD_PACKAGES.
BATTERY_PACKAGES = ("memra-engine", "memra-tokenizer")
# Packages the evidence build (tools/cell_evidence.py BUILD_COMMAND) selects in one cargo call.
# Cargo unifies dependency features across them, so every manifest of a package in their
# closure shapes the battery binaries, even where the package's code is not linked into them.
EVIDENCE_BUILD_PACKAGES = ("memra-engine", "memra-server", "memra-tokenizer")
# Files a crate compiles. The literal crate-input census reads only these, outside test targets.
CODE_SUFFIXES = (".rs", ".cu", ".cuh", ".h", ".c", ".cc", ".cpp")
TEST_TARGET = re.compile(r"^crates/[^/]+/(tests|benches|examples)/")
STRENGTH = ("docs", "research", "tests", "comment", "diagnostic", "cpu", "gate", "serving",
            "numeric", "defaults", "build", "kernel", "perf", UNKNOWN)


class ImpactError(ValueError):
    """The change cannot be classified; callers refuse rather than guess."""


@dataclass(frozen=True)
class Change:
    path: str
    old_mode: str = "100644"
    new_mode: str = "100644"
    added: tuple = ()
    removed: tuple = ()


@dataclass(frozen=True)
class ArchTable:
    accepted: tuple            # build targets build.rs accepts, e.g. ("120a", "100a", ...)
    caps: dict                 # compute capability -> target, e.g. {"12.0": "120a"}
    families: dict = field(default_factory=dict)  # target -> policy family (identity if absent)

    def family(self, target):
        return self.families.get(target, target)

    def target_for_cap(self, cap):
        return self.caps.get(cap.strip())


# ---------------------------------------------------------------- arch table parsing

_TARGET = r'"([0-9]+[a-z]?)"'
_ARM = re.compile(r'((?:(?:Some\()?"[0-9]+\.[0-9]+"\)?\s*\|\s*)*(?:Some\()?"[0-9]+\.[0-9]+"\)?)'
                  r'\s*=>\s*(?:Some\()?"([0-9]+[a-z]?)"')


def _function_body(text, name):
    match = re.search(r"\bfn\s+" + name + r"\s*\(", text)
    if not match:
        return None
    start = text.find("{", match.end())
    depth, end = 0, start
    while end < len(text):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        end += 1
        if depth == 0:
            return text[start:end]
    return None


def parse_arch_table(build_rs, cuda_arch_rs=None):
    """Read the accepted targets and the capability map the build itself uses."""
    if not build_rs:
        raise ImpactError(f"cannot read the CUDA target table: {BUILD_RS} is missing")
    assert_match = re.search(r"matches!\(\s*[a-z_]+\.as_str\(\),\s*([^)]*)\)\s*,\s*\"MEMRA_CUDA_ARCH must be",
                             build_rs)
    if not assert_match:
        raise ImpactError("cannot find the accepted MEMRA_CUDA_ARCH set in build.rs")
    accepted = tuple(re.findall(_TARGET, assert_match.group(1)))
    body = None
    if cuda_arch_rs:
        body = _function_body(cuda_arch_rs, "target_for_compute_cap")
    if body is None:
        body = _function_body(build_rs, "detect_arch")
    if body is None:
        raise ImpactError("cannot find the compute-capability map (detect_arch / target_for_compute_cap)")
    caps = {}
    for caps_text, target in _ARM.findall(body):
        for cap in re.findall(r'"([0-9]+\.[0-9]+)"', caps_text):
            caps[cap] = target
    families = {}
    family_body = _function_body(cuda_arch_rs, "family") if cuda_arch_rs else None
    if family_body:
        for source, family in re.findall(r'"([0-9]+[a-z]?)"\s*=>\s*"([0-9]+[a-z]?)"', family_body):
            families[source] = family
    if not accepted or not caps:
        raise ImpactError("empty CUDA target table")
    unknown = sorted(set(caps.values()) - set(accepted))
    if unknown:
        raise ImpactError("capability map names targets build.rs refuses: " + ", ".join(unknown))
    return ArchTable(accepted=accepted, caps=caps, families=families)


# ---------------------------------------------------------------------- rules

def load_rules(text, arch=None):
    try:
        rules = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ImpactError(f"{RULES_PATH}: {error}") from error
    if rules.get("schema") != SCHEMA:
        raise ImpactError(f"{RULES_PATH}: unknown schema")
    for rule in rules.get("rule", []):
        if rule.get("class") not in CLASSES or not isinstance(rule.get("pattern"), str):
            raise ImpactError(f"{RULES_PATH}: invalid rule {rule!r}")
        try:
            rule["_re"] = re.compile(rule["pattern"])
        except re.error as error:
            raise ImpactError(f"{RULES_PATH}: bad pattern {rule['pattern']!r}: {error}") from error
    for name, kind in rules.get("package", {}).items():
        if name != "perf_packages" and kind not in CLASSES:
            raise ImpactError(f"{RULES_PATH}: package {name} has unknown class {kind}")
    content = rules.get("content", {})
    rules["_device_key"] = re.compile(content.get("device_key", r"(?!)"), re.IGNORECASE)
    rules["_comment_suffixes"] = tuple(content.get("comment_only_suffixes", ()))
    if arch is not None:
        missing = [t for t in arch.accepted if t not in rules.get("target", {})]
        if missing:
            raise ImpactError("build.rs accepts targets with no [target] row in "
                              f"{RULES_PATH}: {', '.join(missing)}; add a row with its push_gate decision")
        for name, device in rules.get("device", {}).items():
            if arch.target_for_cap(device.get("compute_cap", "")) is None:
                raise ImpactError(f"device {name}: compute capability {device.get('compute_cap')} "
                                  "maps to no build target")
    return rules


def target_name(rules, target):
    return rules.get("target", {}).get(target, {}).get("name", "sm_" + target)


def push_gated(rules, target):
    return bool(rules.get("target", {}).get(target, {}).get("push_gate"))


def device_target(rules, arch, device):
    row = rules.get("device", {}).get(device)
    if row is None:
        raise ImpactError(f"unknown device class {device}")
    return arch.target_for_cap(row["compute_cap"])


# --------------------------------------------------------------------- classifier

def _comment_line(line):
    """A `//` comment line. Blank lines and `//` lines ending in a backslash (a C/CUDA line
    continuation splices the next line into the comment) do not count: a blank line can sit
    inside a Rust multi-line string, and a continued line can hide code."""
    stripped = line.strip()
    return stripped.startswith("//") and not stripped.endswith("\\")


def _package_of(path, owners):
    if owners is None:
        match = re.match(r"crates/([a-z0-9-]+)/", path)
        return match.group(1) if match else None
    found = [name for prefix, name in owners.items() if path.startswith(prefix + "/")]
    return found[0] if len(found) == 1 else None


def _match_rule(rules, path):
    for rule in rules.get("rule", []):
        match = rule["_re"].search(path)
        if match:
            return rule, match
    return None, None


def _template(value, match, path):
    if not isinstance(value, str) or "{" not in value:
        return value
    fields = {"stem": PurePosixPath(path).stem, **{k: v for k, v in match.groupdict().items() if v}}
    return value.format(**fields)


def classify_file(change, rules, arch, *, owners=None, crate_inputs=None, linked=None, built=None,
                  same_registry=False):
    """One path -> {class, targets, perf, defaults, pack, route, why, linked, extra}.

    linked: the workspace packages linked into the battery binaries (link_closure); None means
    unknown, and every path then counts as linked. A path is linked unless it belongs to, or is
    read only by, packages outside that set. built: the evidence build's packages; a manifest of
    one of them is a linked build input. extra: further (class, linked) answers the path also
    needs, when a rule and the crates reading it disagree (the cells take the union)."""
    path = change.path
    result = {"path": path, "class": UNKNOWN, "targets": None, "perf": False, "defaults": False,
              "devices": [], "pack": None, "route": None, "why": "no rule matches this path", "linked": True,
              "extra": []}
    if path.startswith(PUBLICATION) or path == "research/INDEX.md":
        return {**result, "class": "research", "why": "qualification metadata (excluded from source inputs)"}
    if path == FLAGS_MD and same_registry and change.old_mode == change.new_mode == "100644":
        return {**result, "class": "docs", "why": "FLAGS.md prose: the env registry build.rs compiles from it is unchanged"}
    package = _package_of(path, owners)
    if "120000" in (change.old_mode, change.new_mode) and (package is not None or not path.startswith(("research/", "docs/"))):
        return {**result, "why": "symlink: its target is a source input the diff does not show"}
    consumers = set()
    if crate_inputs:
        consumers = set().union(*(pkgs for pattern, pkgs in crate_inputs.items() if _input_match(path, pattern)))
    # a crate's own file read by another crate (include!/include_str! across crates): linked when
    # any reader is; its class stays its own package's
    cross = consumers - {package} if package is not None else set()
    if package is not None:
        consumers = set()
    rule, match = _match_rule(rules, path)
    packages = rules.get("package", {})
    perf_packages = set(packages.get("perf_packages", ()))
    if any(p.startswith("<") for p in consumers):
        return {**result, "why": next(p for p in consumers if p.startswith("<")).strip("<>")}
    consumer_kind = max((packages.get(p, UNKNOWN) for p in consumers), key=STRENGTH.index) if consumers else None
    consumer_answer = {"class": consumer_kind, "why": "read by crate " + ", ".join(sorted(consumers)),
                       "perf": bool(consumers & perf_packages) and consumer_kind in PROGRAM,
                       "linked": linked is None or bool(consumers & linked)} if consumers else None
    own_linked = linked is None or package is None or package in linked or bool(cross & linked)
    # A path both read by a crate and matched by a rule needs both answers: Cargo.lock is hashed
    # by memra-server's build id (serve-smoke) and is a build input of every crate (battery); a gate
    # file a server read would still be the gate. The stronger class leads, the other is extra.
    if consumers and not (rule is not None and STRENGTH.index(rule["class"]) > STRENGTH.index(consumer_kind)):
        result.update(consumer_answer)
        if rule is not None and rule["class"] not in NO_CELL:
            result["extra"].append({"class": rule["class"], "linked": own_linked, "why": rule.get("why", rule["pattern"])})
    elif rule is not None:
        result.update({"class": rule["class"], "why": rule.get("why", rule["pattern"]),
                       "pack": _template(rule.get("pack"), match, path),
                       "route": _template(rule.get("route"), match, path),
                       "linked": own_linked})
        if consumers:
            result["why"] += " (also read by crate " + ", ".join(sorted(consumers)) + ")"
            if consumer_kind not in NO_CELL:
                result["extra"].append({k: consumer_answer[k] for k in ("class", "linked", "why")})
        if "perf" in rule:
            result["perf"] = bool(rule["perf"])
        else:
            result["perf"] = package in perf_packages and rule["class"] in PROGRAM
        targets = set(rule.get("targets", ()))
        families = set(rule.get("families", ()))
        if targets or families:
            result["targets"] = sorted(t for t in arch.accepted
                                       if t in targets or arch.family(t) in families)
        if (package is not None and rule["class"] not in ("tests", "diagnostic", "gate")
                and package not in packages):
            result.update({"class": UNKNOWN, "why": f"workspace package {package} has no class in {RULES_PATH}"})
    elif package is not None:
        kind = packages.get(package)
        if kind is None:
            result["why"] = f"workspace package {package} has no class in {RULES_PATH}"
        else:
            result.update({"class": kind, "why": f"{package} default ({kind})",
                           "perf": package in perf_packages and kind in PROGRAM,
                           "linked": own_linked})
    manifest = re.fullmatch(r"crates/([^/]+)/Cargo\.toml", path)
    if manifest and (built is None or manifest.group(1) in built or package in built):
        # its dependency features unify into the one cargo call that builds the battery binaries
        if STRENGTH.index(result["class"]) < STRENGTH.index("build"):
            result.update({"class": "build", "why": result["why"] + "; a manifest of the evidence build: its features "
                           "unify into the battery binaries"})
        result.update({"linked": True, "perf": True})
    if result["class"] in (*PROGRAM, "gate") and change.old_mode == change.new_mode != "000000":
        lines = [l for l in (*change.added, *change.removed)]
        if path.endswith(rules["_comment_suffixes"]) and (change.added or change.removed) and all(map(_comment_line, lines)):
            return {**result, "class": "comment", "perf": False,
                    "why": "comment-only edit: the compiled program is unchanged"}
    if result["class"] in PROGRAM:
        code = [l for l in (*change.added, *change.removed) if l.strip() and not _comment_line(l)]
        keyed = [l for l in code if rules["_device_key"].search(l)]
        if keyed:
            result.update({"defaults": True, "perf": True})
            # A line that names a card keys a default on that card's device class.
            result["devices"] = sorted(name for name, row in rules.get("device", {}).items()
                                       if any(row["name"].lower() in l.lower() for l in keyed))
    return result


def link_closure(read, roots=BATTERY_PACKAGES):
    """Workspace packages linked into `root`'s binaries: root and every path dependency reached
    through [dependencies] and [build-dependencies] (target-specific tables and [patch] path
    crates included; dev-dependencies link only into tests). read(path) -> text or None.
    Returns package names and their directory names, or None when a manifest cannot be read
    (the caller then counts every path as linked)."""
    import posixpath
    try:
        ws = tomllib.loads(read("Cargo.toml") or "")
    except tomllib.TOMLDecodeError:
        return None
    ws_deps = ws.get("workspace", {}).get("dependencies", {})
    manifests = {}

    def load(d):
        if d not in manifests:
            text = read(f"{d}/Cargo.toml")
            try:
                manifests[d] = None if text is None else tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                manifests[d] = None
        return manifests[d]

    members = ws.get("workspace", {}).get("members", [])
    if isinstance(roots, str):
        roots = (roots,)
    root_dirs = []
    for root in roots:
        found = next((m for m in members if (load(m) or {}).get("package", {}).get("name") == root), None)
        if found is None:
            return None
        root_dirs.append(found)
    todo = root_dirs + [posixpath.normpath(spec["path"]) for patch in ws.get("patch", {}).values()
                        for spec in patch.values() if isinstance(spec, dict) and spec.get("path")]
    seen, names = set(), set()
    while todo:
        d = todo.pop()
        if d in seen:
            continue
        seen.add(d)
        manifest = load(d)
        if manifest is None:
            return None
        names.update({manifest.get("package", {}).get("name"), PurePosixPath(d).name} - {None})
        tables = [manifest.get(k, {}) for k in ("dependencies", "build-dependencies", "build_dependencies")]
        for spec in manifest.get("target", {}).values():
            tables += [spec.get(k, {}) for k in ("dependencies", "build-dependencies", "build_dependencies")]
        for table in tables:
            for name, spec in table.items():
                base = d
                if isinstance(spec, dict) and spec.get("workspace"):
                    spec, base = ws_deps.get(name), ""
                if isinstance(spec, dict) and spec.get("path"):
                    todo.append(posixpath.normpath(posixpath.join(base, spec["path"])))
    return names


# docs/FLAGS.md is compiled into the engine only through the env registry build.rs generates from
# it (generate_env_registry: legal names, prefixes and retired doors; the owned families come
# from the sources). An edit that leaves that registry as it was compiles to the same program.
# flags_registry() is a port of the generator pinned to its source: when build.rs's
# generate_env_registry differs from the pinned hash, the port is not trusted and FLAGS.md keeps
# its crate class.
FLAGS_MD = "docs/FLAGS.md"
REGISTRY_GENERATOR_SHA = "18bf048eb85d32519550"
_RUST_WS = set("\t\n\x0b\x0c\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000") | {chr(c) for c in range(0x2000, 0x200B)}


def _generator_span(build_rs):
    found = list(re.finditer(r"^fn generate_env_registry\(.*?^\}$", build_rs or "", re.M | re.S))
    return found[0] if len(found) == 1 else None


def registry_generator_sha(build_rs):
    match = _generator_span(build_rs)
    return hashlib.sha256(match.group(0).encode()).hexdigest()[:20] if match else None


_READER = re.compile(r"join\(|include(_str|_bytes)?!|read_to_string|read\(|File::open|Path::new|PathBuf::from|"
                     r"rerun-if-changed|fs::|OpenOptions")


def flags_readers_ok(repo, head, build_rs):
    """FLAGS.md reaches the compiled program only through the pinned generator: build.rs names it
    (and writes the registry) nowhere else, main still calls it, and every other code line that
    builds a path to FLAGS.md sits in a test item or a test target."""
    span = _generator_span(build_rs)
    if span is None:
        return False
    rest = build_rs[:span.start()] + build_rs[span.end():]
    if not re.search(r"^\s*generate_env_registry\(&out\);", rest, re.M):
        return False
    for line in rest.splitlines():
        if not _comment_line(line) and not line.strip().startswith("///") and (
                "FLAGS.md" in line or "memra_env_registry.rs" in line):
            return False
    out = _git(repo, "grep", "-n", "-I", "-B3", "-F", "FLAGS.md", head, "--", "crates", check=False)
    if out.returncode not in (0, 1):
        return False
    window, hits = {}, []
    for raw in out.stdout.decode(errors="replace").splitlines():
        m = re.match(r"[^:]+:(.+?)([:-])(\d+)[:-](.*)$", raw)
        if not m:
            continue
        source, sep, num, text = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        window[(source, num)] = text
        if sep == ":":
            hits.append((source, num, text))
    from impact_map import delta as _delta
    scanned = {}
    for source, num, text in hits:
        if source == BUILD_RS or TEST_TARGET.match(source) or not source.endswith((".rs", "/Cargo.toml")):
            continue
        if _comment_line(text) or text.strip().startswith(("///", "#")) or '"' not in text:
            continue
        context = "\n".join(window.get((source, n), "") for n in range(num - 3, num + 1))
        if not _READER.search(context):
            continue  # a message naming the file, not a read
        if source not in scanned:
            scanned[source] = _delta.scan(source, read_blob(repo, head, source, strict=False))
        if not any(v[3] and v[4] != "outside" and v[1] <= num <= v[2] for v in scanned[source].values()):
            return False
    return True


def flags_registry(text):
    """(legal names, prefixes, retired (name, ledger heading)) as build.rs derives them."""
    def trim(s):
        i, j = 0, len(s)
        while i < j and s[i] in _RUST_WS:
            i += 1
        while j > i and s[j - 1] in _RUST_WS:
            j -= 1
        return s[i:j]

    def tokens(cell):
        out, i = [], 0
        while (start := cell.find("MEMRA_", i)) >= 0:
            end = start
            while end < len(cell) and ("A" <= cell[end] <= "Z" or "0" <= cell[end] <= "9" or cell[end] == "_"):
                end += 1
            prefix = end < len(cell) and cell[end] == "*"
            stem = cell[start:end].rstrip("_")
            if len(stem) > len("MEMRA"):
                out.append((stem + "_" if prefix else stem, prefix))
            i = max(end, start + 1)
        return out

    def heading(line):
        level = len(line) - len(line.lstrip("#"))
        if level == 0 or level > 6 or not line[level:].startswith(" "):
            return None
        return level, trim(line[level:])

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    defs, legal, prefixes, ledger = set(), set(), set(), {}
    removed, in_block = None, False
    for line in (l[:-1] if l.endswith("\r") else l for l in lines):
        hd = heading(line)
        if hd is not None:
            level, head = hd
            in_block = False
            if removed is not None and level > removed[0] and not head.startswith("Removed"):
                pass
            elif head.startswith("Removed"):
                removed = (level, head)
            else:
                removed = None
            continue
        if removed is not None:
            starts = line.startswith(("- ", "| ", "`MEMRA_"))
            continues = in_block and line.startswith(" ") and trim(line) != ""
            if starts or continues:
                in_block = True
                for name, prefix in tokens(line):
                    if not prefix:
                        ledger.setdefault(name, removed[1])
            else:
                in_block = False
            continue
        for name, prefix in tokens(line):
            (prefixes if prefix else legal).add(name)
        if line.startswith("| "):
            cells = line[2:].split("|")
            second = trim(cells[1] if len(cells) > 1 else "").lstrip("*")
            removed_row = second.startswith("REMOVED")
            for name, prefix in tokens(cells[0]):
                if prefix:
                    continue
                if removed_row:
                    ledger.setdefault(name, "row marked " + second.rstrip("*"))
                else:
                    defs.add(name)
    retired = sorted((n, h) for n, h in ledger.items() if n not in defs)
    return sorted(legal - {n for n, _ in retired}), sorted(prefixes), retired


def same_flags_registry(repo, base, head):
    """True when FLAGS.md changed but the registry build.rs compiles from it did not, and nothing
    else compiled reads FLAGS.md."""
    build_rs = read_blob(repo, head, BUILD_RS, strict=False)
    if registry_generator_sha(build_rs) != REGISTRY_GENERATOR_SHA or not flags_readers_ok(repo, head, build_rs):
        return False
    # a non-UTF-8 FLAGS.md makes the generator fall back to an empty registry: never the same
    old, new = read_blob(repo, base, FLAGS_MD, strict=False), read_blob(repo, head, FLAGS_MD, strict=False)
    return old is not None and new is not None and flags_registry(old) == flags_registry(new)


def _input_match(path, pattern):
    import fnmatch
    return (path == pattern or fnmatch.fnmatchcase(path, pattern)
            or (pattern.endswith("/**") and path.startswith(pattern[:-3] + "/")))


def parse_trailers(messages, arch):
    """Impact-Targets / Impact-Reason trailers. Narrowing applies only when EVERY commit in
    the range declares targets, so one undeclared commit keeps the full target set."""
    declared, reasons, missing = set(), [], []
    for sha, message in messages:
        targets = re.findall(r"^Impact-Targets:\s*(.+)$", message, re.MULTILINE)
        reason = re.findall(r"^Impact-Reason:\s*(.+)$", message, re.MULTILINE)
        if not targets:
            missing.append(sha)
            continue
        if not reason:
            raise ImpactError(f"commit {sha[:12]} declares Impact-Targets without an Impact-Reason")
        for value in targets:
            for target in re.split(r"[,\s]+", value.strip()):
                target = target.removeprefix("sm_")
                if target not in arch.accepted:
                    raise ImpactError(f"commit {sha[:12]} declares unknown target {target}")
                declared.add(target)
        reasons.extend(f"{sha[:12]}: {r.strip()}" for r in reason)
    if not messages or missing:
        return None, reasons, missing
    return declared, reasons, missing


def classify(changes, rules, arch, *, owners=None, crate_inputs=None, declared=None, refiner=None, linked=None,
             built=None, same_registry=False):
    """Changes -> impact: the cells, the per-file classes and what stayed uncovered.

    refiner (tools/impact_map/gate.py) narrows a Rust/CUDA program file from the items the diff
    changed: token-identical edits reach nothing, and build targets shrink to the devices the
    changed items' map reach includes. Path rules decide first; the map only narrows.

    linked: the packages linked into the battery binaries (link_closure). A cell records whether
    any of its paths is linked; requirements() asks for the battery and the perf A/B only then."""
    files = [classify_file(c, rules, arch, owners=owners, crate_inputs=crate_inputs, linked=linked, built=built,
                           same_registry=same_registry) for c in changes]
    if refiner is not None and refiner.active():
        files = [refiner.file(f) for f in files]
    covered = {name for name, pack in rules.get("pack", {}).items() if pack.get("roster")}
    cells, uncovered = {}, {}
    for item in files:
        kind = item["class"]
        targets = set(arch.accepted if item["targets"] is None else item["targets"])
        if declared is not None and kind not in UNNARROWED:
            targets &= declared
        item["targets"] = sorted(targets)
        if kind in NO_CELL or not targets:
            continue
        if kind == "numeric" and item["pack"] and item["pack"] not in covered:
            uncovered.setdefault(item["pack"], []).append(item["path"])
            continue
        if kind in PROGRAM and item.get("uncovered_families"):
            # the map places every changed item on families no battery model belongs to: the
            # battery cannot see the change; those families' own gates are the evidence
            for fam in item["uncovered_families"]:
                uncovered.setdefault(fam, []).append(item["path"])
            continue
        item_linked = kind == UNKNOWN or item.get("linked", True)
        # the perf A/B times run-gen, which links none of this path's code
        perf = item["perf"] and item_linked
        scopes = UNKNOWN_SCOPES if kind == UNKNOWN else (kind,)
        answers = [(s, item_linked) for s in scopes]
        answers += [("defaults", item_linked)] if item["defaults"] else []
        answers += [("perf", True)] if perf else []
        for extra in item.get("extra", ()):
            if extra["class"] not in NO_CELL:
                answers += [(s, extra["class"] == UNKNOWN or extra["linked"])
                            for s in (UNKNOWN_SCOPES if extra["class"] == UNKNOWN else (extra["class"],))]
        for target in targets:
            for scope, is_linked in answers:
                cell = cells.setdefault((target, scope), {"paths": [], "packs": set(), "routes": set(),
                                                          "devices": {}, "linked": False, "unlinked": False})
                cell["paths"].append(item["path"])
                cell["linked" if is_linked else "unlinked"] = True
                if scope == "defaults":
                    for d in item["devices"]:
                        if device_target(rules, arch, d) == target:
                            flags = cell["devices"].setdefault(d, set())
                            flags.add(is_linked)
                if item["pack"]:
                    cell["packs"].add(item["pack"])
                if item["route"]:
                    cell["routes"].add(item["route"])
    ordered = []
    for (target, scope) in sorted(cells, key=lambda k: (arch.accepted.index(k[0]), SCOPES.index(k[1]))):
        cell = cells[(target, scope)]
        ordered.append({"target": target, "name": target_name(rules, target), "scope": scope,
                        "push_gate": push_gated(rules, target), "paths": sorted(set(cell["paths"])),
                        "packs": sorted(cell["packs"]), "routes": sorted(cell["routes"]),
                        "devices": sorted(cell["devices"]), "battery_linked": cell["linked"],
                        "outside_battery": cell["unlinked"],
                        "linked_devices": sorted(d for d, v in cell["devices"].items() if True in v),
                        "unlinked_devices": sorted(d for d, v in cell["devices"].items() if False in v)})
    return {"schema": SCHEMA, "targets": list(arch.accepted), "cells": ordered,
            "files": files, "uncovered": {k: sorted(v) for k, v in sorted(uncovered.items())},
            "declared_targets": None if declared is None else sorted(declared)}


def requirements(impact):
    """Cells -> what evidence each target needs: {target: {"kinds": set, "cells": [...]}}.

    A correctness cell needs the battery when its change is linked into the battery binaries
    (battery_linked); a cell whose paths all sit outside them (memra-server, memra-lanes) needs
    serve-smoke instead, the one test that runs that code. serving adds serve-smoke; perf adds the
    A/B, which times run-gen and so applies to linked cells only; a gate change needs all three
    (the changed gate itself must run). `devices` lists device classes a card-keyed default names;
    `device_kinds` says what each needs from that card itself."""
    needs = {}
    for cell in impact["cells"]:
        entry = needs.setdefault(cell["target"], {"name": cell["name"], "push_gate": cell["push_gate"],
                                                  "kinds": set(), "cells": [], "devices": set(),
                                                  "device_kinds": {}})
        entry["cells"].append(cell["scope"])
        linked = cell.get("battery_linked", True)
        outside = cell.get("outside_battery", False)
        if cell["scope"] in CORRECTNESS:
            if linked:
                entry["kinds"].add("battery")
            if outside:
                entry["kinds"].add("serving")
        if cell["scope"] == "serving":
            entry["kinds"].add("serving")
        if cell["scope"] == "perf" and linked:
            entry["kinds"].add("perf")
        if cell["scope"] == "gate":
            entry["kinds"].update(("battery", "serving", "perf"))
        linked_devices = set(cell.get("linked_devices", cell.get("devices", ())))
        unlinked_devices = set(cell.get("unlinked_devices", ()))
        for device in cell.get("devices", ()):
            entry["devices"].add(device)
            kinds = entry["device_kinds"].setdefault(device, set())
            if device in linked_devices:
                kinds.update(("battery", "perf"))
            if device in unlinked_devices:
                kinds.add("serving")
    return needs


def render(impact):
    lines = []
    if not impact["cells"]:
        lines.append("change impact: none (no device cell)")
    for cell in impact["cells"]:
        gate = "push-gated" if cell["push_gate"] else "daily/pre-tag"
        extra = []
        if cell["packs"]:
            extra.append("packs=" + ",".join(cell["packs"]))
        if cell["routes"]:
            extra.append("routes=" + ",".join(cell["routes"]))
        if cell.get("devices"):
            extra.append("devices=" + ",".join(cell["devices"]))
        if not cell.get("battery_linked", True):
            extra.append("outside the battery binaries")
        elif cell.get("outside_battery"):
            extra.append("partly outside the battery binaries")
        sample = ", ".join(cell["paths"][:3]) + (f" (+{len(cell['paths']) - 3})" if len(cell["paths"]) > 3 else "")
        lines.append(f"  cell {cell['name']}:{cell['scope']:<8} [{gate}] {' '.join(extra)} <- {sample}")
    for pack, paths in impact["uncovered"].items():
        lines.append(f"  UNCOVERED pack {pack}: the device battery runs no {pack} model; "
                     f"its pack gate is the evidence ({len(paths)} path(s), e.g. {paths[0]})")
    if impact["declared_targets"] is not None:
        lines.append("  narrowed by Impact-Targets: " + ", ".join(impact["declared_targets"]))
    if impact.get("map"):
        from impact_map import gate as impact_gate
        lines.extend(impact_gate.render_plan(impact["map"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------- git adapter

def _git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), "-c", "core.quotePath=false", *args],
                            capture_output=True)
    if check and result.returncode != 0:
        raise ImpactError("git " + " ".join(args[:3]) + ": " + result.stderr.decode(errors="replace").strip())
    return result


def read_blob(repo, ref, path, strict=True):
    """Text of path at ref, or None when absent. Non-UTF-8 text refuses (strict) or is None."""
    result = _git(repo, "show", f"{ref}:{path}", check=False)
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode()
    except UnicodeDecodeError as error:
        if strict:
            raise ImpactError(f"{path} at {str(ref)[:12]} is not UTF-8: {error}") from error
        return None


def arch_table_at(repo, ref):
    return parse_arch_table(read_blob(repo, ref, BUILD_RS), read_blob(repo, ref, CUDA_ARCH_RS))


def rules_at(repo, ref, arch):
    text = read_blob(repo, ref, RULES_PATH)
    if text is None:
        raise ImpactError(f"{RULES_PATH} is missing at {ref}")
    return load_rules(text, arch)


def diff_changes(repo, base, head, suffixes=(".rs", ".cu", ".cuh", ".h", ".c", ".cc", ".cpp")):
    raw = _git(repo, "diff", "--raw", "-z", "--no-renames", "--no-ext-diff", base, head).stdout.decode(errors="replace")
    fields = raw.split("\0")[:-1]
    modes = {}
    for meta, path in zip(fields[0::2], fields[1::2]):
        old_mode, new_mode = meta.lstrip(":").split()[:2]
        modes[path] = (old_mode, new_mode)
    code = [p for p in modes if p.startswith("crates/") and p.endswith(suffixes)]
    hunks = {p: ([], []) for p in code}
    if code:
        patch = _git(repo, "diff", "-U0", "--no-renames", "--no-color", "--no-ext-diff", "--no-textconv",
                     "--src-prefix=a/", "--dst-prefix=b/", base, head,
                     "--", *code).stdout.decode(errors="replace")
        # Hunk bodies are read by their header counts, so a content line that happens to
        # look like a file header ("--- a/x") is still counted as content.
        current, old_left, new_left = None, 0, 0
        for line in patch.split("\n"):
            if old_left or new_left:
                if line.startswith("-") and old_left:
                    hunks.setdefault(current, ([], []))[1].append(line[1:])
                    old_left -= 1
                elif line.startswith("+") and new_left:
                    hunks.setdefault(current, ([], []))[0].append(line[1:])
                    new_left -= 1
                elif line.startswith("\\"):
                    pass
                else:
                    raise ImpactError(f"unparseable hunk in {current}")
            elif line.startswith("diff --git "):
                current = None
            elif line.startswith("--- a/") and current is None:
                current = line[6:]
            elif line.startswith("+++ b/"):
                current = line[6:]
            elif line.startswith("@@ "):
                header = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
                if header is None:
                    raise ImpactError(f"unparseable hunk header in {current}")
                old_left = int(header.group(1) if header.group(1) is not None else 1)
                new_left = int(header.group(2) if header.group(2) is not None else 1)
    return [Change(p, *modes[p], tuple(hunks.get(p, ((), ()))[0]), tuple(hunks.get(p, ((), ()))[1]))
            for p in sorted(modes)]


def crate_inputs_at(repo, ref, paths, force=False):
    """Paths a crate reads beyond its own sources (include_str!, build-script inputs, another
    crate's files), via the CI census. force: also for a crate-only diff (a path outside the
    battery binaries' closure, whose readers decide whether it is linked)."""
    if not force and not any(not p.startswith(("crates/", "research/release-qualification/")) for p in paths):
        return {}, None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # The census and the literal check are unioned: tools/validation_plan.py is CI tooling
    # (class cpu), so a change to it must not be able to hide a literal include from the gate.
    literal = literal_inputs(repo, ref, paths)
    try:
        import validation_plan as vp
        tree = vp.Tree(repo, ref)
        _, owners = vp.workspace(tree)
        inputs = {k: set(v) for k, v in vp.included_inputs(tree, owners).items()}
        for path, packages in literal.items():
            inputs.setdefault(path, set()).update(packages)
        return inputs, owners
    except (ImportError, subprocess.SubprocessError, OSError, ValueError, KeyError):
        # vp.Refused is a ValueError.
        # The census cannot answer (typically: the running tools and the head tree differ).
        # Fall back to the literal check: a non-crate path is an input when a non-comment
        # line under crates/ spells it. Includes are literal paths in this repo.
        return literal, None


def literal_inputs(repo, ref, paths):
    wanted = sorted({p for p in paths if not p.startswith(("crates/", PUBLICATION))})
    if not wanted:
        return {}
    # the patterns go through stdin: one -e per path overflows the argument list on long ranges
    result = subprocess.run(["git", "-C", str(repo), "-c", "core.quotePath=false", "grep", "-n", "-F", "-I", "-f", "-",
                             ref, "--", "crates"], input="\n".join(wanted).encode(), capture_output=True)
    if result.returncode not in (0, 1):
        return {"**": {"<literal input census failed>"}}
    inputs = {}
    for line in result.stdout.decode(errors="replace").splitlines():
        _, source, _, text = line.split(":", 3)
        # only sources a crate compiles and its manifest ([[bin]] path, build), outside its test
        # targets: a path named in a fixture, a manifest comment or a test script is not compiled
        # in (a README listed in a model's artifact lock, AGENTS.md cited by a battery test)
        manifest = source.endswith("/Cargo.toml")
        if (_comment_line(text) or (manifest and text.lstrip().startswith("#"))
                or not (source.endswith(CODE_SUFFIXES) or manifest) or TEST_TARGET.match(source)):
            continue
        quoted = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', text))
        package = _package_of(source, None)
        for path in wanted:
            if path in quoted:
                inputs.setdefault(path, set()).add(package or "<unowned>")
    return inputs


def range_messages(repo, base, head):
    if set(base) == {"0"} or _git(repo, "cat-file", "-t", base, check=False).stdout.strip() != b"commit":
        return []
    out = _git(repo, "log", "--format=%H%x00%B%x01", f"{base}..{head}").stdout.decode(errors="replace")
    messages = []
    for entry in out.split("\x01"):
        if not entry.strip():
            continue
        sha, message = entry.strip("\n").split("\0", 1)
        touched = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-z", "-r", "-m", "--root",
                       sha).stdout.decode(errors="replace").split("\0")
        touched = [p for p in touched if p]
        # An evidence commit (qualification metadata only) declares nothing and narrows nothing.
        if touched and all(p.startswith(PUBLICATION) or p == "research/INDEX.md" for p in touched):
            continue
        messages.append((sha, message))
    return messages


def rules_in_force(repo, base, head, arch):
    """The push base's rules decide what a change needs, so a change cannot relax its own
    requirements (push_gate, path rules, packages). A build target new in the head takes the
    head's row with push_gate forced on, and devices of that new target come from the head.
    With no rules at the base (the change that introduces them), the head's rules apply."""
    head_text = read_blob(repo, head, RULES_PATH)
    base_text = read_blob(repo, base, RULES_PATH) if _git(repo, "cat-file", "-t", base, check=False).stdout.strip() == b"commit" else None
    if base_text is None:
        if head_text is None:
            raise ImpactError(f"{RULES_PATH} is missing at {head}")
        return load_rules(head_text, arch)
    rules = load_rules(base_text)
    head_rules = load_rules(head_text) if head_text is not None else {}
    for target in arch.accepted:
        if target not in rules.get("target", {}):
            row = dict(head_rules.get("target", {}).get(target, {"name": "sm_" + target}))
            rules.setdefault("target", {})[target] = {**row, "push_gate": True}
            for name, device in head_rules.get("device", {}).items():
                if arch.target_for_cap(device.get("compute_cap", "")) == target:
                    rules.setdefault("device", {}).setdefault(name, device)
    rules["_from"] = "base"
    return validate_rules(rules, arch)


def validate_rules(rules, arch):
    for name, device in rules.get("device", {}).items():
        if arch.target_for_cap(device.get("compute_cap", "")) is None:
            raise ImpactError(f"device {name}: compute capability {device.get('compute_cap')} maps to no build target")
    return rules


def battery_linked(repo, base, head):
    """(linked, built): packages linked into the battery binaries, and packages in the evidence
    build's closure, each over the base and the head (a change that adds a link changes a
    manifest, which is a build input anyway). None for either when a side cannot say."""
    sides = [head] + ([base] if _git(repo, "cat-file", "-t", base, check=False).stdout.strip() == b"commit" else [])
    linked, built = set(), set()
    for ref in sides:
        read = lambda p, ref=ref: read_blob(repo, ref, p, strict=False)
        a, b = link_closure(read, BATTERY_PACKAGES), link_closure(read, EVIDENCE_BUILD_PACKAGES)
        if a is None or b is None:
            return None, None
        linked |= a
        built |= b
    return linked, built


def impact_between(repo, base, head, rules_text=None, use_map=True, map_ref=None, map_data=None):
    """map_ref pins the impact map to a commit (default: the base's, as for the rules); map_data
    is an explicit map for what-if replays and is never taken from the environment."""
    arch = arch_table_at(repo, head)
    rules = load_rules(rules_text, arch) if rules_text is not None else rules_in_force(repo, base, head, arch)
    changes = diff_changes(repo, base, head)
    linked, built = battery_linked(repo, base, head)
    outside = [c.path for c in changes if linked is not None and _package_of(c.path, None) not in (None, *linked)]
    inputs, owners = crate_inputs_at(repo, head, [c.path for c in changes], force=bool(outside))
    declared, reasons, _ = parse_trailers(range_messages(repo, base, head), arch)
    refiner = None
    if use_map:
        from impact_map import gate as impact_gate
        try:
            refiner = impact_gate.Refiner(repo, base, head, rules, arch, map_ref=map_ref, map_data=map_data)
        except impact_gate.MapError as error:
            raise ImpactError(str(error)) from error
    same_registry = any(c.path == FLAGS_MD for c in changes) and same_flags_registry(repo, base, head)
    impact = classify(changes, rules, arch, owners=owners, crate_inputs=inputs, declared=declared, refiner=refiner,
                      linked=linked, built=built, same_registry=same_registry)
    impact["declared_reasons"] = reasons
    impact["base"], impact["head"] = base, head
    impact["map"] = refiner.report() if refiner is not None else None
    return impact, rules, arch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rules", type=Path, help="what-if: classify with this rules file instead of the head's")
    parser.add_argument("--no-map", action="store_true", help="path rules only (ignore docs/impact-map/map.json)")
    args = parser.parse_args()
    try:
        base = _git(args.repo, "merge-base", args.base, args.head).stdout.decode().strip()
        impact, _, _ = impact_between(args.repo, base, args.head,
                                      args.rules.read_text() if args.rules else None, use_map=not args.no_map)
    except ImpactError as error:
        print(f"change impact: REFUSED: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(impact, indent=1, sort_keys=True, default=sorted))
    else:
        print(render(impact))
        for target, need in sorted(requirements(impact).items()):
            state = "required at push" if need["push_gate"] else "daily regression / pre-tag only"
            print(f"  {need['name']}: {', '.join(sorted(need['kinds']))} evidence, {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
