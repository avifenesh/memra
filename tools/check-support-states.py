#!/usr/bin/env python3
"""Support-state census: a pack declares only what its own records prove (memra#551).

Reads every ModelPack static and speech SUPPORT const under crates/memra-gguf/src/model_packs,
the records in docs/support-records.toml, the tracked gates.txt receipts those records cite,
and every NativeReference/NativeQualified/NativeTuned token in the published docs. Exits 1 on
any mismatch, 2 when the inputs cannot be parsed (a census that parses nothing verifies
nothing). Text only, no build. The rules are written at the top of docs/support-records.toml.

Usage: tools/check-support-states.py [--root DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib

STATES = ("NativeReference", "NativeQualified", "NativeTuned")
RANK = {None: 0, "NativeReference": 1, "NativeQualified": 2, "NativeTuned": 3}
GATES = (
    "Config",
    "TokenizerTemplate",
    "TensorCensus",
    "TinyParity",
    "CheckpointParity",
    "RewriteParity",
    "Serve",
)
REQUIRED = {
    "NativeReference": {"Config", "TinyParity"},
    "NativeQualified": {
        "Config",
        "TokenizerTemplate",
        "TensorCensus",
        "TinyParity",
        "CheckpointParity",
        "Serve",
    },
}
REQUIRED["NativeTuned"] = REQUIRED["NativeQualified"] | {"RewriteParity"}
CI_TINY = "ci:verify-tiny"
CI_TINY_GATES = {"Config", "TinyParity"}
CLI_TEST = "supported_packs_write_deterministic_native_oracles"
PACK_DIR = "crates/memra-gguf/src/model_packs"
CLI_SRC = "crates/memra-cli/src/lib.rs"
RECORDS = "docs/support-records.toml"
DOC_FILES = ("README.md", "STATUS.md", "CLAUDE.md")
DOC_DIRS = ("docs",)
DOC_EXCLUDE = ("docs/archive/",)
TOKEN_RE = re.compile(r"\b(NativeReference|NativeQualified|NativeTuned)\b")
MARKER_RE = re.compile(r"<!--\s*support:\s*([^>]*?)\s*-->")
# A `none` line must read as a definition, negation or pending mention, not as a claim.
NONE_CUES = re.compile(
    r"\b(means?|not|no|never|unset|pending|requires?|required|until|next|still need|"
    r"exactly three|states?|minimum|additionally|plus|only after|before|downgraded|"
    r"reference only|is not|alone)\b",
    re.IGNORECASE,
)
# A definition entry: the line opens with the bold state name, e.g. "- **NativeReference**: ...".
DEFINITION = re.compile(r"^\s*(?:- )?\*\*`?Native(?:Reference|Qualified|Tuned)`?\*\*[:,]")
HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


class InputError(Exception):
    pass


def parse_packs(root: pathlib.Path) -> dict[str, dict]:
    """family -> {state, static, module, tiny_plan, path}"""
    packs: dict[str, dict] = {}
    pack_dir = root / PACK_DIR
    files = sorted(pack_dir.glob("*/mod.rs"))
    if not files:
        raise InputError(f"no pack sources under {PACK_DIR}")
    opened = 0
    for path in files:
        text = path.read_text()
        module = path.parent.name
        opened += text.count(": ModelPack = ModelPack {")
        for m in re.finditer(r"pub static (\w+): ModelPack = ModelPack \{", text):
            body_start = m.end()
            depth, i = 1, body_start
            while depth and i < len(text):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            body = text[body_start:i]
            fam = re.search(r'^\s*family:\s*"([^"]+)"', body, re.M)
            sup = re.search(r"^\s*support:\s*(.+?),\s*$", body, re.M)
            tiny = re.search(r"^\s*tiny_plan:\s*(.+?),\s*$", body, re.M)
            if not fam or not sup or not tiny:
                raise InputError(f"{path}: {m.group(1)} has no parseable family/support/tiny_plan")
            packs[fam.group(1)] = {
                "state": parse_support(sup.group(1), path),
                "static": m.group(1),
                "module": module,
                "tiny_plan": tiny.group(1) != "None",
                "path": str(path.relative_to(root)),
                "speech": False,
            }
        for m in re.finditer(r"pub const SUPPORT: Option<NativeSupport> = (.+?);", text):
            fam = re.search(r'pub const FAMILY: &\'static str = "([^"]+)";', text)
            if not fam:
                raise InputError(f"{path}: SUPPORT const without a FAMILY const")
            packs[fam.group(1)] = {
                "state": parse_support(m.group(1), path),
                "static": "SUPPORT",
                "module": module,
                "tiny_plan": False,
                "path": str(path.relative_to(root)),
                "speech": True,
            }
    parsed = sum(1 for p in packs.values() if not p["speech"])
    if parsed != opened or parsed == 0:
        raise InputError(f"parsed {parsed} ModelPack statics but the sources open {opened}")
    return packs


def parse_support(expr: str, path: pathlib.Path) -> str | None:
    expr = expr.strip()
    if expr == "None":
        return None
    m = re.fullmatch(r"Some\(NativeSupport::(\w+)\)", expr)
    if not m or m.group(1) not in STATES:
        raise InputError(f"{path}: unparseable support expression {expr!r}")
    return m.group(1)


def pack_lists(root: pathlib.Path) -> dict[str, set[tuple[str, str]]]:
    text = (root / PACK_DIR / "mod.rs").read_text()
    lists = {}
    for name in ("PACKS", "ONBOARDING_PROFILES"):
        m = re.search(rf"pub const {name}: &\[&ModelPack\] = &\[(.*?)\];", text, re.S)
        if not m:
            raise InputError(f"{PACK_DIR}/mod.rs: no {name} list")
        lists[name] = set(re.findall(r"&(\w+)::(\w+)", m.group(1)))
    return lists


def ci_tiny_lists(root: pathlib.Path) -> set[str]:
    """Which pack lists the hosted verify-tiny test iterates."""
    text = (root / CLI_SRC).read_text()
    start = text.find(f"fn {CLI_TEST}()")
    if start < 0:
        return set()
    end = text.find("\n    #[test]", start)
    body = text[start : end if end > 0 else len(text)]
    covered = set()
    if "model_packs::PACKS" in body:
        covered.add("PACKS")
        if "model_packs::ONBOARDING_PROFILES" in body:
            covered.add("ONBOARDING_PROFILES")
    return covered


def read_gates(root: pathlib.Path, rel: str) -> tuple[str | None, dict[str, str]]:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    gates = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            gates[k.strip()] = v.strip()
    family = None
    lock = path.with_name("artifact.lock")
    tiny = path.with_name("tiny-gate.tsv")
    if lock.is_file():
        m = re.search(r"^family=(\S+)$", lock.read_text(), re.M)
        family = m.group(1) if m else None
    elif tiny.is_file():
        m = re.search(r"^family\t(\S+)$", tiny.read_text(), re.M)
        family = m.group(1) if m else None
    return family, gates


def check_records(root, packs, errors) -> dict[str, dict]:
    path = root / RECORDS
    if not path.is_file():
        raise InputError(f"missing {RECORDS}")
    data = tomllib.loads(path.read_text())
    records = data.get("record", [])
    if not records:
        raise InputError(f"{RECORDS} has no [[record]] entries")
    lists = pack_lists(root)
    ci_lists = ci_tiny_lists(root)
    by_id: dict[str, dict] = {}
    for rec in records:
        rid = rec.get("id", "<no id>")
        where = f"{RECORDS}: record {rid}"
        if rid in by_id:
            errors.append(f"{where}: duplicate id")
        by_id[rid] = rec
        fam, state = rec.get("pack"), rec.get("state")
        if fam not in packs:
            errors.append(f"{where}: pack {fam!r} is not a declared pack family")
            continue
        if state not in STATES:
            errors.append(f"{where}: state {state!r} is not one of {', '.join(STATES)}")
            continue
        for field in ("artifact", "numeric_program", "topology", "route"):
            if not str(rec.get(field, "")).strip():
                errors.append(f"{where}: missing {field}")
        gates = rec.get("gates", {})
        evidence = rec.get("evidence", {})
        for gate in GATES:
            if gates.get(gate) not in ("passed", "pending"):
                errors.append(f"{where}: gate {gate} must be \"passed\" or \"pending\"")
        for gate in set(gates) | set(evidence):
            if gate not in GATES:
                errors.append(f"{where}: unknown gate {gate}")
        for gate in REQUIRED[state]:
            if gates.get(gate) != "passed":
                errors.append(f"{where}: {state} requires {gate} passed, record says {gates.get(gate)!r}")
        for gate, status in gates.items():
            if status != "passed":
                if evidence.get(gate):
                    errors.append(f"{where}: {gate} is pending but cites evidence")
                continue
            sources = evidence.get(gate, [])
            if not sources:
                errors.append(f"{where}: {gate} passed with no evidence")
            for src in sources:
                if src == CI_TINY:
                    info = packs[fam]
                    member = [n for n, s in lists.items() if (info["module"], info["static"]) in s]
                    if gate not in CI_TINY_GATES:
                        errors.append(f"{where}: {CI_TINY} backs only Config and TinyParity, not {gate}")
                    elif info["state"] is None or not info["tiny_plan"]:
                        errors.append(f"{where}: {CI_TINY} skips {fam}: it needs support Some and a tiny_plan")
                    elif not member or member[0] not in ci_lists:
                        errors.append(
                            f"{where}: {CI_TINY} does not reach {fam} ({info['static']} is in "
                            f"{member[0] if member else 'no list'}; {CLI_TEST} iterates "
                            f"{', '.join(sorted(ci_lists)) or 'nothing'})"
                        )
                    continue
                try:
                    rfam, rgates = read_gates(root, src)
                except FileNotFoundError:
                    errors.append(f"{where}: {gate} evidence {src} is not a tracked file")
                    continue
                if rfam != fam:
                    errors.append(f"{where}: {gate} evidence {src} is for family {rfam!r}, not {fam!r}")
                elif rgates.get(gate) != "passed":
                    errors.append(f"{where}: {gate} evidence {src} says {rgates.get(gate)!r}")
        if RANK[state] >= RANK["NativeQualified"]:
            art = str(rec.get("artifact", ""))
            if re.search(r"tiny", art, re.I):
                errors.append(f"{where}: a tiny fixture never qualifies a pack")
            if not (HEX40.search(art) or SHA256.search(art)):
                errors.append(f"{where}: {state} needs a pinned artifact (revision or sha256)")
            if not HEX40.fullmatch(str(rec.get("memra_commit", ""))):
                errors.append(f"{where}: {state} needs a 40-hex memra_commit")
    # Declared state == highest recorded state, per exact family.
    for fam, info in sorted(packs.items()):
        recorded = max(
            (r["state"] for r in records if r.get("pack") == fam and r.get("state") in STATES),
            key=lambda s: RANK[s],
            default=None,
        )
        if info["state"] != recorded:
            errors.append(
                f"{info['path']}: {fam} declares support {info['state'] or 'None'} but its "
                f"records prove {recorded or 'nothing'}"
            )
    return by_id


def doc_paths(root: pathlib.Path):
    for name in DOC_FILES:
        if (root / name).is_file():
            yield root / name
    for d in DOC_DIRS:
        for p in sorted((root / d).rglob("*.md")):
            rel = str(p.relative_to(root))
            if not any(rel.startswith(x) for x in DOC_EXCLUDE):
                yield p


def check_docs(root, by_id, errors) -> int:
    checked = 0
    for path in doc_paths(root):
        rel = path.relative_to(root)
        for n, line in enumerate(path.read_text().splitlines(), 1):
            tokens = set(TOKEN_RE.findall(MARKER_RE.sub("", line)))
            markers = MARKER_RE.findall(line)
            if not tokens:
                if markers:
                    errors.append(f"{rel}:{n}: support marker on a line with no state token")
                continue
            checked += 1
            if len(markers) != 1:
                errors.append(
                    f"{rel}:{n}: {', '.join(sorted(tokens))} needs one "
                    f"<!-- support: <record id> --> or <!-- support: none --> marker"
                )
                continue
            head, _, denied_part = markers[0].partition(";")
            ids = [x.strip() for x in head.split(",") if x.strip()]
            denied: set[str] = set()
            if denied_part:
                words = denied_part.split()
                if not words or words[0] != "not" or len(words) < 2:
                    errors.append(f"{rel}:{n}: after ';' a support marker takes 'not <State> ...'")
                    continue
                denied = set(words[1:])
                bad = denied - set(STATES)
                if bad:
                    errors.append(f"{rel}:{n}: 'not' names unknown state {', '.join(sorted(bad))}")
                    continue
                if denied - tokens:
                    errors.append(f"{rel}:{n}: 'not {' '.join(sorted(denied - tokens))}' names a state the line does not mention")
                    continue
            if ids == ["none"]:
                bare = MARKER_RE.sub("", line)
                if not (NONE_CUES.search(bare) or DEFINITION.match(bare)):
                    errors.append(f"{rel}:{n}: support: none on a line that reads as a claim")
                continue
            if "none" in ids or not ids:
                errors.append(f"{rel}:{n}: support marker mixes none with record ids or is empty")
                continue
            unknown = [i for i in ids if i not in by_id]
            if unknown:
                errors.append(f"{rel}:{n}: unknown support record {', '.join(unknown)}")
                continue
            backed = {by_id[i]["state"] for i in ids}
            top = max(RANK[s] for s in backed)
            for tok in sorted(denied):
                if RANK[tok] <= top:
                    errors.append(f"{rel}:{n}: denies {tok} but the named records prove {', '.join(sorted(backed))}")
            for tok in sorted(tokens - backed - denied):
                errors.append(
                    f"{rel}:{n}: says {tok} but the named records prove {', '.join(sorted(backed))}"
                )
    return checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent, type=pathlib.Path)
    root = ap.parse_args().root
    errors: list[str] = []
    try:
        packs = parse_packs(root)
        by_id = check_records(root, packs, errors)
        lines = check_docs(root, by_id, errors)
    except (InputError, tomllib.TOMLDecodeError) as exc:
        print(f"check-support-states: {exc}", file=sys.stderr)
        return 2
    if lines == 0:
        print("check-support-states: VACUOUS: no state tokens found in the docs", file=sys.stderr)
        return 2
    for e in errors:
        print(f"check-support-states: {e}", file=sys.stderr)
    if errors:
        print(f"check-support-states: FAIL ({len(errors)} violations)", file=sys.stderr)
        return 1
    positive = sum(1 for p in packs.values() if p["state"])
    print(
        f"check-support-states: {len(packs)} packs ({positive} with a support state), "
        f"{len(by_id)} records, {lines} doc lines, all backed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
