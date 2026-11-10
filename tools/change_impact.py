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


def device_class_of(rules, name):
    """The device class of a device name as nvidia-smi reports it: the one `[device.X]` row
    whose `name` appears in it. None when no row or more than one matches."""
    found = [cls for cls, row in rules.get("device", {}).items() if row["name"] in name]
    return found[0] if len(found) == 1 else None


def maintained_classes(rules, arch):
    """The device classes of push-gated build targets: the ones every push is proven on."""
    return sorted(cls for cls in rules.get("device", {}) if push_gated(rules, device_target(rules, arch, cls)))


def device_status(rules, arch, name):
    """`maintained <class>`, `unmaintained <class>` or `unknown` for a device name. A device
    of a maintained class is one whose gates must be calibrated; a device of a class with no
    push-gated target is not maintained; a name no row matches is unknown, never unmaintained."""
    cls = device_class_of(rules, name)
    if cls is None:
        return "unknown"
    return ("maintained " if cls in maintained_classes(rules, arch) else "unmaintained ") + cls


def tree_rules(repo):
    """The rules and the arch table of a working tree (not of a commit)."""
    repo = Path(repo)
    try:
        cuda_arch = (repo / CUDA_ARCH_RS).read_text() if (repo / CUDA_ARCH_RS).exists() else None
        arch = parse_arch_table((repo / BUILD_RS).read_text(), cuda_arch)
        return load_rules((repo / RULES_PATH).read_text(), arch), arch
    except OSError as error:
        raise ImpactError(f"cannot read the rules of {repo}: {error}") from error


# --------------------------------------------------------------------- classifier

def _comment_line(line):
    """A `//` comment line. Blank lines and `//` lines ending in a backslash (a C/CUDA line
    continuation splices the next line into the comment) do not count: a blank line can sit
    inside a Rust multi-line string, and a continued line can hide code."""
    stripped = line.strip()
    return stripped.startswith("//") and not stripped.endswith("\\")


# Files that cannot read a path themselves: data a crate may load, whose path strings name files
# of some other tree (a model fixture's `"path": "README.md"`), never this repository's.
DATA_SUFFIXES = (".json", ".md", ".txt", ".lock", ".tsv", ".csv", ".yaml", ".yml", ".jsonl")


def _may_read(source, text):
    """Whether a line of `source` that spells a path can be a read of it: not a data file, and not
    a `#` comment in Python, shell or TOML."""
    if source.endswith(DATA_SUFFIXES):
        return False
    if source.endswith((".py", ".sh", ".toml")) and text.strip().startswith("#"):
        return False
    return True


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


def classify_file(change, rules, arch, *, owners=None, crate_inputs=None):
    """One path -> {class, targets, perf, defaults, pack, route, why}."""
    path = change.path
    result = {"path": path, "class": UNKNOWN, "targets": None, "perf": False, "defaults": False,
              "devices": [], "pack": None, "route": None, "why": "no rule matches this path"}
    if path.startswith(PUBLICATION) or path == "research/INDEX.md":
        return {**result, "class": "research", "why": "qualification metadata (excluded from source inputs)"}
    package = _package_of(path, owners)
    if "120000" in (change.old_mode, change.new_mode) and (package is not None or not path.startswith(("research/", "docs/"))):
        return {**result, "why": "symlink: its target is a source input the diff does not show"}
    consumers = set()
    if package is None and crate_inputs:
        consumers = set().union(*(pkgs for pattern, pkgs in crate_inputs.items() if _input_match(path, pattern)))
    rule, match = _match_rule(rules, path)
    packages = rules.get("package", {})
    perf_packages = set(packages.get("perf_packages", ()))
    if any(p.startswith("<") for p in consumers):
        return {**result, "why": next(p for p in consumers if p.startswith("<")).strip("<>")}
    if consumers:
        kind = max((packages.get(p, UNKNOWN) for p in consumers), key=STRENGTH.index)
        result.update({"class": kind, "why": "read by crate " + ", ".join(sorted(consumers)),
                       "perf": bool(consumers & perf_packages) and kind in PROGRAM})
    elif rule is not None:
        result.update({"class": rule["class"], "why": rule.get("why", rule["pattern"]),
                       "pack": _template(rule.get("pack"), match, path),
                       "route": _template(rule.get("route"), match, path)})
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
                           "perf": package in perf_packages and kind in PROGRAM})
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


def classify(changes, rules, arch, *, owners=None, crate_inputs=None, declared=None):
    """Changes -> impact: the cells, the per-file classes and what stayed uncovered."""
    files = [classify_file(c, rules, arch, owners=owners, crate_inputs=crate_inputs) for c in changes]
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
        scopes = UNKNOWN_SCOPES if kind == UNKNOWN else (kind,)
        scopes = set(scopes) | ({"defaults"} if item["defaults"] else set()) | ({"perf"} if item["perf"] else set())
        for target in targets:
            for scope in scopes:
                cell = cells.setdefault((target, scope), {"paths": [], "packs": set(), "routes": set(),
                                                          "devices": set()})
                cell["paths"].append(item["path"])
                if scope == "defaults":
                    cell["devices"].update(d for d in item["devices"] if device_target(rules, arch, d) == target)
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
                        "devices": sorted(cell["devices"])})
    return {"schema": SCHEMA, "targets": list(arch.accepted), "cells": ordered,
            "files": files, "uncovered": {k: sorted(v) for k, v in sorted(uncovered.items())},
            "declared_targets": None if declared is None else sorted(declared)}


def requirements(impact):
    """Cells -> what evidence each target needs: {target: {"kinds": set, "cells": [...]}}.

    battery: any correctness scope; serving adds serve-smoke; perf adds the A/B; a gate change
    needs all three (the changed gate itself must run). `devices` lists device classes a
    card-keyed default names; each needs its own battery + perf."""
    needs = {}
    for cell in impact["cells"]:
        entry = needs.setdefault(cell["target"], {"name": cell["name"], "push_gate": cell["push_gate"],
                                                  "kinds": set(), "cells": [], "devices": set()})
        entry["cells"].append(cell["scope"])
        if cell["scope"] in CORRECTNESS:
            entry["kinds"].add("battery")
        if cell["scope"] == "serving":
            entry["kinds"].add("serving")
        if cell["scope"] == "perf":
            entry["kinds"].add("perf")
        if cell["scope"] == "gate":
            entry["kinds"].update(("battery", "serving", "perf"))
        entry["devices"].update(cell.get("devices", ()))
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
        sample = ", ".join(cell["paths"][:3]) + (f" (+{len(cell['paths']) - 3})" if len(cell["paths"]) > 3 else "")
        lines.append(f"  cell {cell['name']}:{cell['scope']:<8} [{gate}] {' '.join(extra)} <- {sample}")
    for pack, paths in impact["uncovered"].items():
        lines.append(f"  UNCOVERED pack {pack}: the device battery runs no {pack} model; "
                     f"its pack gate is the evidence ({len(paths)} path(s), e.g. {paths[0]})")
    if impact["declared_targets"] is not None:
        lines.append("  narrowed by Impact-Targets: " + ", ".join(impact["declared_targets"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------- git adapter

def _git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), "-c", "core.quotePath=false", *args],
                            capture_output=True)
    if check and result.returncode != 0:
        raise ImpactError("git " + " ".join(args[:3]) + ": " + result.stderr.decode(errors="replace").strip())
    return result


def read_blob(repo, ref, path):
    result = _git(repo, "show", f"{ref}:{path}", check=False)
    return result.stdout.decode() if result.returncode == 0 else None


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


def crate_inputs_at(repo, ref, paths):
    """Non-crate paths a crate reads (include_str!, build-script inputs), via the CI census."""
    if not any(not p.startswith(("crates/", "research/release-qualification/")) for p in paths):
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
    args = ["grep", "-n", "-F", "-I"]
    for path in wanted:
        args += ["-e", path]
    result = _git(repo, *args, ref, "--", "crates", check=False)
    if result.returncode not in (0, 1):
        return {"**": {"<literal input census failed>"}}
    inputs = {}
    for line in result.stdout.decode(errors="replace").splitlines():
        _, source, _, text = line.split(":", 3)
        if _comment_line(text) or not _may_read(source, text):
            continue
        package = _package_of(source, None)
        for path in wanted:
            if path in text:
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


def impact_between(repo, base, head, rules_text=None):
    arch = arch_table_at(repo, head)
    rules = load_rules(rules_text, arch) if rules_text is not None else rules_in_force(repo, base, head, arch)
    changes = diff_changes(repo, base, head)
    inputs, owners = crate_inputs_at(repo, head, [c.path for c in changes])
    declared, reasons, _ = parse_trailers(range_messages(repo, base, head), arch)
    impact = classify(changes, rules, arch, owners=owners, crate_inputs=inputs, declared=declared)
    impact["declared_reasons"] = reasons
    impact["base"], impact["head"] = base, head
    return impact, rules, arch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rules", type=Path, help="what-if: classify with this rules file instead of the head's")
    parser.add_argument("--device-status", metavar="NAME",
                        help="print `maintained <class>`, `unmaintained <class>` or `unknown` for a device "
                             "name as nvidia-smi reports it, from the working tree's rules, and exit")
    args = parser.parse_args()
    if args.device_status is not None:
        try:
            rules, arch = tree_rules(args.repo)
        except ImpactError as error:
            print(f"change impact: REFUSED: {error}", file=sys.stderr)
            return 1
        print(device_status(rules, arch, args.device_status))
        return 0
    try:
        base = _git(args.repo, "merge-base", args.base, args.head).stdout.decode().strip()
        impact, _, _ = impact_between(args.repo, base, args.head,
                                      args.rules.read_text() if args.rules else None)
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
