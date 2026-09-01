#!/usr/bin/env python3
"""Public-boundary policy check.

Enforces `tools/public-boundary-policy.toml` against tracked files, using
`tools/public-boundary-allowlist.jsonl` as a hash-pinned, RULE-SCOPED grandfather list.

Modes:
  check   (default) — scan tracked files, print violations, exit 1 on any unmatched
                      violation. With --commits-file, scan the changed blobs in those
                      commits instead of the checkout. With --refs, scan every blob
                      version carried by the matching published refs. With
                      --summary-only, report per-rule counts and no paths.
  seed              — regenerate the allowlist so every current violation is pinned by
                      SHA-256; used to bootstrap. Refuses to run if the allowlist file
                      already exists unless `--force` is given.
  verify-allowlist  — confirm every allowlist entry still points at a tracked file with
                      the recorded SHA-256; entries that no longer apply are removed
                      when `--prune` is given. Exit 1 if any entry drifts.

The policy file, this scanner, the allowlist, and the progress dir are exempt from
the secret regex so they can document what is blocked without triggering themselves.

Why --refs exists: the pre-push hook only judges blob versions introduced by the push, so a
rule added after a branch was pushed never looks at that branch again. Two lanes were pushed
one day before the account-identity rules landed and stayed unexamined until a review read
them by hand. --refs re-scans published history under the CURRENT policy, which is the only
mode that catches a leak the policy learned about late.

Two reporting invariants, both from defects the 2026-08-19 public-ref audit proved on real
bytes rather than argued from:

  * An allowlist entry exempts a blob for the RULES IT NAMES, never for every rule. Keyed on
    (path, sha256) alone, one grandfathered file was exempt from rules added after it: a
    recon journal pinned for `production_endpoint` also carried a capacity-block id at offset
    9,616, and that id was permanently invisible to every report.
  * A blob reports EVERY rule it matches, ordered worst-rule-first by the policy's severity
    ranks. First-match-only was correct for the enforce/don't-enforce decision and wrong for
    triage: it labelled the worst file in the repo — a cloud account id plus a privilege level
    plus an IAM principal — `personal_email`, one line inside 56 other `personal_email` hits.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11 (serving/spot boxes ship 3.10)
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        sys.exit(
            "check-public-boundary: needs tomllib (Python >= 3.11) or the tomli package.\n"
            "This interpreter has neither — a push from this machine would be UNGATED\n"
            "(that is how a lane branch reached the public remote once, see\n"
            "research/public-ref-sweep-20260819/CREDENTIAL-VERDICT.md). Fix: `pip install\n"
            "tomli` here, or push from the rig."
        )
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "tools" / "public-boundary-policy.toml"
ALLOWLIST_PATH = ROOT / "tools" / "public-boundary-allowlist.jsonl"

SEVERITY_FLOOR = 1
SEVERITY_CEILING = 5

# Allowlist field naming the rules on an entry that pin a finding nobody has ruled on yet. It
# exists because the two ways to hold a known-but-undecided finding are both bad on their own:
# leave the gate red and it gets bypassed or ignored, pin it silently and green means nobody looks
# — which is how the last leak hid inside a 578-entry list. Pinned entries keep the gate
# deployable; this field makes every run of the check state how many are outstanding, so the
# backlog cannot go quiet.
UNREMEDIATED_KEY = "unremediated"
# A rule the policy does not rank cannot happen (load_policy refuses), but an ad-hoc Policy
# built in a test has no ranks at all, and an unranked rule must not sort BELOW a ranked one.
DEFAULT_SEVERITY = SEVERITY_CEILING


class _EveryRule(FrozenSet[str]):
    """Sentinel exemption set: a legacy allowlist entry that names no rule.

    `rule in UNSCOPED` is always true, which is exactly the old (path, sha256) behaviour —
    kept nameable so the one place that still grants it is visible instead of implied.
    """

    def __contains__(self, item: object) -> bool:  # noqa: D105 - see class docstring
        return True

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<every rule>"


UNSCOPED: FrozenSet[str] = _EveryRule()


@dataclass
class Policy:
    secret_patterns: Dict[str, re.Pattern]
    secret_sources: Dict[str, str]
    secret_union: re.Pattern
    secret_groups: Dict[str, str]
    private_paths: Dict[str, str]
    bypass_paths: List[str]
    severity: Dict[str, int] = field(default_factory=dict)

    def rank(self, rule: str) -> int:
        return self.severity.get(rule, DEFAULT_SEVERITY)

    def worst_first(self, rules: Iterable[str]) -> Tuple[str, ...]:
        """Order rule names worst-rank-first, then alphabetically for a stable report."""
        return tuple(sorted(rules, key=lambda name: (-self.rank(name), name)))


@dataclass
class Violation:
    path: str
    sha256: str
    category: str
    detail: str
    # Every rule this blob matches, worst-first. Empty on hand-built Violations, in which case
    # `violation_rules` recovers the names from `detail`, which has always carried them.
    rules: Tuple[str, ...] = ()
    severity: int = 0
    # rule name -> earliest line, so a partially-exempt blob can be re-reported against only
    # the rules its allowlist entry does not cover, without rescanning the bytes.
    lines: Dict[str, int] = field(default_factory=dict)


def rules_from_detail(detail: str) -> Tuple[str, ...]:
    """Recover rule names from a `detail`/`reason` string.

    Both the allowlist's `reason` and a Violation's `detail` lead with the comma-joined rule
    names before any ` (…)` note, and have since the allowlist was first seeded. That makes the
    existing 578 entries machine-readable for the rule-scoping migration, rather than needing a
    re-seed that would relabel every line and hide the migration inside the noise.
    """
    head = detail.split(" (")[0]
    return tuple(name for name in (part.strip() for part in head.split(",")) if name)


def violation_rules(v: Violation) -> Tuple[str, ...]:
    return v.rules or rules_from_detail(v.detail)


def load_policy(path: Path) -> Policy:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    secret_sources = dict(raw.get("secret_patterns", {}))
    secrets = {name: re.compile(pattern) for name, pattern in secret_sources.items()}
    union_parts: List[str] = []
    secret_groups: Dict[str, str] = {}
    for index, (name, source) in enumerate(secret_sources.items()):
        # Inline global flags are only legal at the start of a regex. Scope them before
        # embedding each policy rule in the combined matcher.
        if source.startswith("(?i)"):
            source = f"(?i:{source[4:]})"
        group = f"rule_{index}"
        union_parts.append(f"(?P<{group}>{source})")
        secret_groups[group] = name
    secret_union = re.compile("|".join(union_parts) if union_parts else r"(?!x)x")
    private = dict(raw.get("private_paths", {}))
    bypass = list(raw.get("bypass_paths", {}).values())
    severity = load_severity(path, raw, set(secret_sources) | set(private))
    return Policy(
        secret_patterns=secrets,
        secret_sources=secret_sources,
        secret_union=secret_union,
        secret_groups=secret_groups,
        private_paths=private,
        bypass_paths=bypass,
        severity=severity,
    )


def load_severity(path: Path, raw: dict, named: set[str]) -> Dict[str, int]:
    """Every rule must be ranked, and every rank must name a rule.

    Fail-closed both ways on purpose. An unranked rule would sort by an implicit default, and a
    default is how a rule that should outrank 56 lines of authorship noise ends up buried under
    them; a rank naming no rule is a typo that silently ranks nothing.
    """
    severity = dict(raw.get("severity", {}))
    unknown = sorted(set(severity) - named)
    if unknown:
        raise SystemExit(
            f"{path}: [severity] ranks names that are not rules or globs: {','.join(unknown)}"
        )
    missing = sorted(named - set(severity))
    if missing:
        raise SystemExit(
            f"{path}: every rule needs a [severity] rank so a long report can be ordered "
            f"worst-first; unranked: {','.join(missing)}"
        )
    for name, level in sorted(severity.items()):
        if not isinstance(level, int) or isinstance(level, bool):
            raise SystemExit(f"{path}: [severity].{name} must be an integer")
        if not SEVERITY_FLOOR <= level <= SEVERITY_CEILING:
            raise SystemExit(
                f"{path}: [severity].{name}={level} outside "
                f"{SEVERITY_FLOOR}..{SEVERITY_CEILING}"
            )
    return severity


def load_allowlist(path: Path) -> Dict[Tuple[str, str], dict]:
    if not path.exists():
        return {}
    entries: Dict[Tuple[str, str], dict] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
        required = {"path", "sha256", "category", "reason", "rules"}
        missing = required.difference(entry)
        if missing:
            # `rules` became required 2026-08-19. Without it an entry is an exemption from
            # EVERY rule, including rules added after it was written, which is how a
            # `production_endpoint` grandfather permanently absorbed a capacity-block id in the
            # same bytes. There is no legacy-tolerant read of this field: the allowlist ships in
            # the repo alongside the scanner, so the two are never out of step.
            raise SystemExit(
                f"{path}:{line_no}: missing required keys: {','.join(sorted(missing))}"
                + (
                    "  — `rules` must name the rule(s) this entry grandfathers; an entry that"
                    " names none is an exemption from every rule"
                    if "rules" in missing
                    else ""
                )
            )
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise SystemExit(f"{path}:{line_no}: malformed sha256")
        rules = entry["rules"]
        if not isinstance(rules, list) or not rules:
            raise SystemExit(f"{path}:{line_no}: rules must be a non-empty list")
        if not all(isinstance(name, str) and name for name in rules):
            raise SystemExit(f"{path}:{line_no}: rules must be non-empty strings")
        pending = entry.get(UNREMEDIATED_KEY)
        if pending is not None:
            if not isinstance(pending, list) or not all(
                isinstance(name, str) for name in pending
            ):
                raise SystemExit(
                    f"{path}:{line_no}: {UNREMEDIATED_KEY} must be a list of rule names"
                )
            # A pending rule the entry does not actually grandfather is a bookkeeping error that
            # would inflate the outstanding count forever without a finding behind it.
            stray = sorted(set(pending) - set(rules))
            if stray:
                raise SystemExit(
                    f"{path}:{line_no}: {UNREMEDIATED_KEY} names rules the entry does not "
                    f"grandfather: {','.join(stray)}"
                )
        key = (entry["path"], entry["sha256"])
        if key in entries:
            raise SystemExit(f"{path}:{line_no}: duplicate allowlist key: {key[0]}")
        entries[key] = entry
    return entries


def exempt_rules(allowlist: Dict[Tuple[str, str], dict], v: Violation) -> Optional[
    FrozenSet[str]
]:
    """Which rules this blob is grandfathered for — None when it is not listed at all."""
    entry = allowlist.get((v.path, v.sha256))
    if entry is None:
        return None
    listed = entry.get("rules")
    if listed is None:
        return UNSCOPED
    return frozenset(listed)


def tracked_files() -> List[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def secret_candidate_files(patterns: Dict[str, str]) -> set[str]:
    """Use git's optimized PCRE walker to prefilter files with any structural secret hit."""
    if not patterns:
        return set()
    args = ["git", "grep", "--text", "-z", "-l", "-P"]
    for source in patterns.values():
        args.extend(("-e", source))
    args.append("--")
    result = subprocess.run(args, cwd=ROOT, capture_output=True)
    if result.returncode == 1:
        return set()
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git grep secret prefilter failed: {message}")
    return {
        raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    }


# Heads, tags and GitHub's PR refs, because the leak does not care which namespace carries it.
# Tags were out of scope until 2026-08-19 and carried MORE violations than heads (151 vs 131
# under the pushed policy) including a rule class heads did not have — and they are the worst
# place for one: immutable release markers are what forks and package consumers pull, and
# rewriting them is worse than the disclosure, so detection is the only lever that works on
# them at all. Comma-separated because `git for-each-ref` takes many patterns and one glob
# cannot span three namespaces.
DEFAULT_REF_GLOB = (
    "refs/remotes/origin/**,refs/remotes/origin-pull/**,refs/tags/**"
)


def published_refs(glob: str) -> List[str]:
    patterns = [part.strip() for part in glob.split(",") if part.strip()]
    if not patterns:
        return []
    out = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", *patterns],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted({line for line in out.splitlines() if line})


def short_ref(ref: str) -> str:
    if ref.startswith("refs/tags/"):
        # Keep the namespace: in a per-ref report a tag and a branch of the same name are two
        # different remediation decisions (a tag is not rewritable without breaking consumers).
        return f"tags/{ref[len('refs/tags/'):]}"
    for prefix in ("refs/remotes/", "refs/heads/"):
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return ref


def ref_namespaces(refs: Iterable[str]) -> Dict[str, int]:
    """Count refs per namespace, so a log reader can see the scan was not silently narrowed."""
    tally: Dict[str, int] = {"heads": 0, "tags": 0, "pull": 0, "other": 0}
    for ref in refs:
        if ref.startswith("refs/tags/"):
            tally["tags"] += 1
        elif "origin-pull/" in ref or ref.startswith("refs/pull/"):
            tally["pull"] += 1
        elif ref.startswith(("refs/remotes/", "refs/heads/")):
            tally["heads"] += 1
        else:
            tally["other"] += 1
    return tally


def secret_candidates_at_refs(
    patterns: Dict[str, str], refs: List[str]
) -> set[Tuple[str, str]]:
    """Return (ref, path) pairs with a structural hit, from ONE multi-tree git grep.

    git grep accepts many trees in a single invocation, which matters: one process over all
    published refs costs seconds, while one process per ref per rule costs minutes and would
    make this mode too slow to keep in CI — the way a gate rots.
    """
    if not patterns or not refs:
        return set()
    args = ["git", "grep", "--text", "-z", "-l", "-P"]
    for source in patterns.values():
        args.extend(("-e", source))
    args.extend(refs)
    result = subprocess.run(args, cwd=ROOT, capture_output=True)
    if result.returncode == 1:
        return set()
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git grep ref prefilter failed: {message}")
    pairs: set[Tuple[str, str]] = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        # `<ref>:<path>`. A ref name cannot contain a colon, so the first one splits it.
        entry = raw.decode("utf-8", errors="surrogateescape")
        ref, _, path = entry.partition(":")
        if path:
            pairs.add((ref, path))
    return pairs


def private_candidates_at_ref(policy: Policy, ref: str) -> List[str]:
    """Paths at `ref` under a private_paths glob, narrowed by git rather than by scanning."""
    if not policy.private_paths:
        return []
    pathspecs = [f":(glob){glob}" for glob in policy.private_paths.values()]
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", ref, "--", *pathspecs],
        cwd=ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return [
        raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def blobs_at(specs: List[str]) -> Dict[str, bytes]:
    """Map each `<ref>:<path>` spec to its bytes in one `git cat-file --batch` pass."""
    if not specs:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input="\n".join(specs).encode() + b"\n",
        capture_output=True,
        check=True,
    )
    out = result.stdout
    loaded: Dict[str, bytes] = {}
    pos = 0
    for spec in specs:
        end = out.find(b"\n", pos)
        if end == -1:
            break
        header = out[pos:end].decode("utf-8", errors="replace").split()
        pos = end + 1
        # A missing or non-blob object prints a header and NO body, so the stream stays
        # aligned with `specs` as long as we only advance past a body when one exists.
        if len(header) != 3 or header[1] != "blob":
            continue
        size = int(header[2])
        loaded[spec] = out[pos : pos + size]
        pos += size + 1  # git terminates each body with a newline.
    return loaded


SCANNED_BLOBS = "scanned_blobs"


def evaluate_refs(
    policy: Policy, refs: List[str], chunk: int = 256, stats: Optional[Dict[str, int]] = None
) -> Tuple[List[Violation], Dict[Tuple[str, str], set[str]]]:
    """Evaluate every candidate blob version carried by `refs`.

    Deduplicated by (path, sha256): branches share history, so the same blob shows up on many
    refs and the allowlist is keyed on content anyway. The refs carrying each violation are
    returned alongside it so a report can say where to go and remove it.
    """
    candidates: set[Tuple[str, str]] = set()
    for ref in refs:
        for path in private_candidates_at_ref(policy, ref):
            candidates.add((ref, path))
    candidates.update(secret_candidates_at_refs(policy.secret_sources, refs))

    violations: Dict[Tuple[str, str], Violation] = {}
    carriers: Dict[Tuple[str, str], set[str]] = {}
    ordered = sorted(candidates)
    read = 0
    for start in range(0, len(ordered), chunk):
        batch = ordered[start : start + chunk]
        loaded = blobs_at([f"{ref}:{path}" for ref, path in batch])
        for ref, path in batch:
            data = loaded.get(f"{ref}:{path}")
            if data is None:
                continue
            read += 1
            violation = evaluate_content(policy, path, data)
            if violation is None:
                continue
            key = (violation.path, violation.sha256)
            violations.setdefault(key, violation)
            carriers.setdefault(key, set()).add(short_ref(ref))
    if stats is not None:
        # A scan that read zero blobs is a scan that proved nothing, and the difference between
        # "clean" and "never ran" has to be visible in a log rather than inferred from silence.
        stats[SCANNED_BLOBS] = read
        stats["candidates"] = len(ordered)
    return list(violations.values()), carriers


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def changed_blobs(commits: Iterable[str]) -> List[Tuple[str, str]]:
    """Return each changed (path, blob oid) version reachable through the commits."""
    blobs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for commit in commits:
        result = subprocess.run(
            [
                "git",
                "diff-tree",
                "--raw",
                "-z",
                "-r",
                "-m",
                "--root",
                "--no-commit-id",
                "--no-renames",
                commit,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        fields = result.stdout.split(b"\0")
        if fields and not fields[-1]:
            fields.pop()
        if len(fields) % 2:
            raise RuntimeError(f"git diff-tree returned malformed output for {commit}")
        for metadata, raw_path in zip(fields[0::2], fields[1::2]):
            parts = metadata.split()
            if len(parts) != 5 or not parts[0].startswith(b":"):
                raise RuntimeError(f"git diff-tree returned malformed metadata for {commit}")
            new_mode = parts[1]
            oid = parts[3].decode("ascii")
            status = parts[4]
            if status.startswith(b"D") or new_mode == b"160000" or not oid.strip("0"):
                continue
            path = raw_path.decode("utf-8", errors="surrogateescape")
            key = (path, oid)
            if key not in seen:
                seen.add(key)
                blobs.append(key)
    return blobs


def matches_glob(path: str, glob: str) -> bool:
    # fnmatch's `*` matches path separators, so `deploy/gateway/**` works for any depth.
    return fnmatch.fnmatchcase(path, glob) or fnmatch.fnmatchcase(path, glob.rstrip("*"))


def is_bypass(path: str, bypass_globs: Iterable[str]) -> bool:
    return any(matches_glob(path, g) for g in bypass_globs)


def scan_secrets(
    path: Path,
    matcher: re.Pattern,
    group_names: Dict[str, str],
    patterns: Optional[Dict[str, re.Pattern]] = None,
) -> List[Tuple[str, int, str]]:
    """Return every structural secret rule matching a tracked file."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    return scan_secret_bytes(data, matcher, group_names, patterns)


def _hit_at(text: str, name: str, start: int, end: int) -> Tuple[str, int, str]:
    line_no = text.count("\n", 0, start) + 1
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return (name, line_no, text[line_start:line_end].strip()[:120])


def scan_secret_bytes(
    data: bytes,
    matcher: re.Pattern,
    group_names: Dict[str, str],
    patterns: Optional[Dict[str, re.Pattern]] = None,
) -> List[Tuple[str, int, str]]:
    """Return the earliest hit for EVERY rule matching these bytes, ordered by line.

    Two passes, because they answer different questions. The union matcher is the cheap gate
    that keeps the 1+ GB research tree scannable — one traversal instead of one per rule — and
    the overwhelming majority of blobs stop there. Only a blob that already hit gets the
    per-rule pass, and that pass is what makes the report honest: alternation reports only the
    branch that won at the earliest position, so two rules matching the same span, or a
    narrower rule inside a span a wider rule already consumed, are invisible to the union no
    matter how it is iterated. Measured on the real repo the second pass costs 24 searches on
    ~580 blobs of ~40,000.

    Reporting one rule per file was correct for the enforce/don't-enforce decision — one hit is
    enough to make a blob a violation — and it is why the worst file in this repo read as
    authorship noise: `personal_email` matched 189 bytes before the account id did, so the
    account id, the AdministratorAccess assertion and the IAM principal beside it were never
    named in any report. Enforcement takes one rule; triage needs all of them.
    """
    text = data.decode("utf-8", errors="ignore")
    if matcher.search(text) is None:
        return []
    hits: List[Tuple[str, int, str]] = []
    if patterns:
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match is not None:
                hits.append(_hit_at(text, name, match.start(), match.end()))
    else:
        # No per-rule dict available (an ad-hoc matcher). Walk the union and deduplicate by
        # rule: still every rule the union can see, just not the ones it structurally cannot.
        seen: set[str] = set()
        for match in matcher.finditer(text):
            name = group_names[match.lastgroup or ""]
            if name in seen:
                continue
            seen.add(name)
            hits.append(_hit_at(text, name, match.start(), match.end()))
    hits.sort(key=lambda hit: (hit[1], hit[0]))
    return hits


def build_violation(
    policy: Policy,
    rel: str,
    sha256: str,
    category: str,
    hits: List[Tuple[str, int, str]],
) -> Violation:
    """Assemble a violation naming every matched rule, worst-rank-first.

    `detail` keeps the shape the allowlist's `reason` has always had — comma-joined rule names,
    then a parenthetical note — so migrating the 578 existing entries is a field addition rather
    than a re-seed that relabels every line. What changed is that the list is now complete and
    ordered by severity instead of being the single rule that happened to match first.
    """
    lines = {name: line for name, line, _ in hits}
    rules = policy.worst_first(lines)
    located = [line for line in lines.values() if line]
    detail = ",".join(rules)
    if located:
        detail = f"{detail} (first hit line {min(located)})"
    return Violation(
        path=rel,
        sha256=sha256,
        category=category,
        detail=detail,
        rules=rules,
        severity=max((policy.rank(name) for name in rules), default=0),
        lines=lines,
    )


def evaluate_content(policy: Policy, rel: str, data: bytes) -> Optional[Violation]:
    bypass = is_bypass(rel, policy.bypass_paths)
    matched_globs = [
        name for name, glob in policy.private_paths.items() if matches_glob(rel, glob)
    ]
    if matched_globs and not bypass:
        # A private path is private by location, so there is no line to report — but the glob
        # names go through the same rule machinery, which is what lets an entry grandfathered
        # for one glob stay accountable to a second one.
        return build_violation(
            policy,
            rel,
            hashlib.sha256(data).hexdigest(),
            "private_path",
            [(name, 0, "") for name in matched_globs],
        )
    if bypass:
        return None
    hits = scan_secret_bytes(
        data, policy.secret_union, policy.secret_groups, policy.secret_patterns
    )
    if not hits:
        return None
    return build_violation(
        policy, rel, hashlib.sha256(data).hexdigest(), "secret_pattern", hits
    )


def evaluate(policy: Policy) -> List[Violation]:
    violations: List[Violation] = []
    secret_candidates = secret_candidate_files(policy.secret_sources)
    for rel in tracked_files():
        full = ROOT / rel
        if not full.is_file():
            continue
        bypass = is_bypass(rel, policy.bypass_paths)
        # Path glob rule (private serving material).
        matched_globs = [
            name
            for name, glob in policy.private_paths.items()
            if matches_glob(rel, glob)
        ]
        if matched_globs and not bypass:
            violations.append(
                build_violation(
                    policy,
                    rel,
                    sha256_of(full),
                    "private_path",
                    [(name, 0, "") for name in matched_globs],
                )
            )
            continue  # one violation per file — path glob wins over secret hits.
        # Secret regex rule.
        if not bypass and rel in secret_candidates:
            hits = scan_secrets(
                full, policy.secret_union, policy.secret_groups, policy.secret_patterns
            )
            if hits:
                violations.append(
                    build_violation(
                        policy, rel, sha256_of(full), "secret_pattern", hits
                    )
                )
    return violations


def evaluate_commits(policy: Policy, commits: Iterable[str]) -> List[Violation]:
    violations: List[Violation] = []
    for rel, oid in changed_blobs(commits):
        data = subprocess.run(
            ["git", "cat-file", "blob", oid],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        violation = evaluate_content(policy, rel, data)
        if violation is not None:
            violations.append(violation)
    return violations


def print_violation(v: Violation, status: str, refs: Optional[set[str]] = None) -> None:
    where = ""
    if refs:
        shown = sorted(refs)
        extra = f" (+{len(shown) - 4} more)" if len(shown) > 4 else ""
        where = f"  on: {', '.join(shown[:4])}{extra}"
    rank = f" sev{v.severity}" if v.severity else ""
    print(f"  [{status}{rank}] {v.path}  ({v.category}: {v.detail}){where}")


def narrow_to(policy: Policy, v: Violation, rules: Tuple[str, ...]) -> Violation:
    """Re-report a partially-grandfathered blob against only the rules it is NOT exempt for."""
    hits = [(name, v.lines.get(name, 0), "") for name in rules]
    narrowed = build_violation(policy, v.path, v.sha256, v.category, hits)
    covered = ",".join(sorted(set(violation_rules(v)) - set(rules)))
    if covered:
        narrowed.detail = f"{narrowed.detail} [allowlisted only for: {covered}]"
    return narrowed


def cmd_check(
    policy: Policy,
    commits: Optional[List[str]] = None,
    refs: Optional[List[str]] = None,
    summary_only: bool = False,
) -> int:
    allowlist = load_allowlist(ALLOWLIST_PATH)
    carriers: Dict[Tuple[str, str], set[str]] = {}
    if refs is not None:
        if not refs:
            # Fail closed. A ref scan that matched nothing is indistinguishable from a clean
            # one in an exit code, and "the glob matched one tracking ref" is exactly how a
            # nightly audit job reports green forever without ever having looked.
            print("public-boundary: no refs matched; nothing scanned.")
            return 1
        spread = ref_namespaces(refs)
        print(
            f"public-boundary: scanning {len(refs)} refs "
            f"({spread['heads']} heads, {spread['tags']} tags, {spread['pull']} pull"
            + (f", {spread['other']} other" if spread["other"] else "")
            + ")."
        )
        if not spread["tags"]:
            print(
                "public-boundary: WARNING — 0 tags in scope. Tags carried more violations "
                "than heads in the 2026-08-19 audit; fetch them (`+refs/tags/*:refs/tags/*`) "
                "or the scan is not the audit it claims to be."
            )
        stats: Dict[str, int] = {}
        violations, carriers = evaluate_refs(policy, refs, stats=stats)
        print(
            f"public-boundary: read {stats.get(SCANNED_BLOBS, 0)} candidate blob versions "
            f"of {stats.get('candidates', 0)} prefiltered."
        )
    elif commits is None:
        violations = evaluate(policy)
    else:
        violations = evaluate_commits(policy, commits)
    unmatched: List[Violation] = []
    grandfathered = 0
    partial = 0
    for v in violations:
        exempt = exempt_rules(allowlist, v)
        if exempt is not None:
            # Rule-scoped: the entry covers the rules it names and nothing else. A blob whose
            # entry covers some of its rules is still a violation for the rest — that is the
            # whole defect, and it is why this is not `if key in allowlist: continue`.
            remaining = tuple(r for r in violation_rules(v) if r not in exempt)
            if not remaining:
                grandfathered += 1
                continue
            partial += 1
            v = narrow_to(policy, v, remaining)
        unmatched.append(v)
    # Worst rule first, then path. A report ordered by scan order is how a severity-5 finding
    # ends up as line 34 of 56 in a class the reader has already learned to skim.
    unmatched.sort(key=lambda item: (-item.severity, item.path))
    total = len(violations)
    scope = f" across {len(refs)} published refs" if refs is not None else ""
    print(
        f"public-boundary: {total} matches{scope} ({grandfathered} grandfathered, "
        f"{len(unmatched)} new)."
    )
    if partial:
        print(
            f"public-boundary: {partial} of those are allowlisted for a DIFFERENT rule than "
            f"the one they now match (rule-scoped exemptions, 2026-08-19)."
        )
    outstanding = unremediated_tally(allowlist, policy)
    if outstanding:
        # Printed on EVERY run, pass or fail. A pinned finding is a decision deferred, not a
        # decision made, and the only thing that keeps a deferral honest is that it is loud.
        total_pinned = sum(outstanding.values())
        worst = max(policy.rank(rule) for rule in outstanding)
        print(
            f"public-boundary: {total_pinned} allowlisted findings are marked "
            f"{UNREMEDIATED_KEY} (worst sev{worst}); owner decision pending. By rule: "
            + ", ".join(
                f"{rule}={count}"
                for rule, count in sorted(
                    outstanding.items(),
                    key=lambda kv: (-policy.rank(kv[0]), -kv[1], kv[0]),
                )
            )
        )
    if unmatched:
        if summary_only:
            # For a job whose log is public: naming the file and line of an unremediated leak
            # is a map to it. Counts are enough to fail the job; a maintainer re-runs without
            # this flag, locally, to get the paths.
            print("New public-boundary violations by rule (re-run without --summary-only):")
            tally: Dict[str, int] = {}
            for v in unmatched:
                for name in violation_rules(v):
                    tally[name] = tally.get(name, 0) + 1
            for name, count in sorted(
                tally.items(), key=lambda kv: (-policy.rank(kv[0]), -kv[1], kv[0])
            ):
                print(f"  sev{policy.rank(name)}  {count:5d}  {name}")
            if carriers:
                # Ref names are already public on GitHub; the path and line are what must not
                # be. A per-ref count is what makes the job decisive rather than a single
                # number: it says which refs to go clean, and it cannot be read as a map.
                print("Refs carrying at least one new violation (counts only):")
                per_ref: Dict[str, int] = {}
                for v in unmatched:
                    for ref in carriers.get((v.path, v.sha256), ()):
                        per_ref[ref] = per_ref.get(ref, 0) + 1
                for ref, count in sorted(
                    per_ref.items(), key=lambda kv: (-kv[1], kv[0])
                ):
                    print(f"  {count:5d}  {ref}")
                print(f"  ({len(per_ref)} refs of {len(refs or ())} scanned)")
        else:
            print("New public-boundary violations (add to allowlist or migrate):")
            for v in unmatched:
                print_violation(v, "NEW", carriers.get((v.path, v.sha256)))
        return 1
    if refs is not None:
        # Published refs are a different corpus from the checkout, so a live allowlist entry
        # that no ref carries is not stale here — only the full-tree check owns that invariant.
        return 0
    if commits is not None:
        # A commit range is intentionally partial. Full-tree check/CI retains the stale-entry
        # invariant; the pre-push range check only judges blob versions introduced by the push.
        return 0
    # Stale entries must fail closed. Otherwise a removed private file can be restored later
    # with the same bytes and silently regain its historical exemption.
    stale = stale_entries(allowlist, violations)
    if stale:
        print(f"public-boundary: {len(stale)} stale allowlist entries (prune before merge):")
        for path, digest, why in stale:
            print(f"  [STALE] {path}  sha256={digest[:12]}…  ({why})")
        return 1
    return 0


def unremediated_tally(
    allowlist: Dict[Tuple[str, str], dict], policy: Policy
) -> Dict[str, int]:
    """Count pinned-but-undecided findings per rule, from the `unremediated` field.

    A named field rather than a marker parsed out of the reason prose: an entry pinned long ago
    for `serve_data_root` that later gained an undecided `serve_prefix` finding is ONE
    outstanding finding, not two, and a count derived from free text drifts from the sentence a
    reviewer reads the moment either is edited.
    """
    tally: Dict[str, int] = {}
    for entry in allowlist.values():
        for rule in entry.get(UNREMEDIATED_KEY) or ():
            tally[rule] = tally.get(rule, 0) + 1
    return tally


def stale_entries(
    allowlist: Dict[Tuple[str, str], dict], violations: List[Violation]
) -> List[Tuple[str, str, str]]:
    """Entries that no longer describe a live finding, per rule.

    Rule scoping sharpens this invariant as well as the exemption. Under (path, sha256) an entry
    stayed live as long as the bytes tripped ANY rule, so remediating the rule the entry was
    granted for left the exemption in place, still covering everything. Now the grant expires
    with the finding it was granted for.
    """
    live_rules = {
        (v.path, v.sha256): set(violation_rules(v)) for v in violations
    }
    stale: List[Tuple[str, str, str]] = []
    for (path, digest), entry in allowlist.items():
        live = live_rules.get((path, digest))
        if live is None:
            stale.append((path, digest, "no live violation for these bytes"))
            continue
        dead = sorted(set(entry.get("rules") or ()) - live)
        if dead:
            stale.append((path, digest, f"rules no longer match: {','.join(dead)}"))
    return stale


def cmd_seed(policy: Policy, force: bool) -> int:
    if ALLOWLIST_PATH.exists() and not force:
        print(
            f"seed: refusing to overwrite existing {ALLOWLIST_PATH.name}; "
            f"pass --force to regenerate."
        )
        return 1
    violations = evaluate(policy)
    ALLOWLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# public-boundary allowlist — pinned SHA-256 grandfather list.",
        "# Generated by tools/check-public-boundary.py --seed. Every entry represents a",
        "# private surface still tracked in the public repo, pending migration to",
        "# darklanes. Changing a file makes its (path, sha256) tuple miss the list, so",
        "# the check fails and forces a review — either update the hash intentionally",
        "# or migrate the file. The extraction plan lives in",
        "# research/public-boundary-20260814/PROGRESS.md.",
        "#",
        "# `rules` is the scope of the exemption and is REQUIRED. An entry grandfathers only",
        "# the rules it names; the same bytes matching any other rule is a fresh violation.",
    ]
    for v in sorted(violations, key=lambda x: (x.category, x.path)):
        entry = {
            "path": v.path,
            "sha256": v.sha256,
            "category": v.category,
            "rules": list(violation_rules(v)),
            "reason": v.detail,
        }
        lines.append(json.dumps(entry, sort_keys=True))
    ALLOWLIST_PATH.write_text("\n".join(lines) + "\n")
    print(
        f"seed: wrote {len(violations)} entries to {ALLOWLIST_PATH.relative_to(ROOT)}"
    )
    return 0


def cmd_verify(policy: Policy, prune: bool) -> int:
    allowlist = load_allowlist(ALLOWLIST_PATH)
    live_violations = evaluate(policy)
    drifted = stale_entries(allowlist, live_violations)
    if not drifted:
        print(
            f"verify: {len(allowlist)} allowlist entries all pin live tracked files "
            f"for the rules they name."
        )
        return 0
    print(f"verify: {len(drifted)} allowlist entries no longer match tracked files:")
    for path, digest, why in drifted:
        print(f"  [DRIFT] {path}  sha256={digest[:12]}…  ({why})")
    if prune:
        # Prune the RULES that went dead, not the whole entry: an entry granted for two rules
        # where one has been remediated must keep covering the other, or --prune would silently
        # re-open a finding the owner never decided on. An entry whose bytes no longer produce
        # any violation, or whose every rule died, drops out entirely.
        live_rules = {
            (v.path, v.sha256): set(violation_rules(v)) for v in live_violations
        }
        pruned: Dict[Tuple[str, str], dict] = {}
        dropped = 0
        narrowed = 0
        for key, entry in allowlist.items():
            live = live_rules.get(key, set())
            keep = [rule for rule in entry.get("rules") or () if rule in live]
            if not keep:
                dropped += 1
                continue
            if len(keep) != len(entry.get("rules") or ()):
                narrowed += 1
                entry = {**entry, "rules": keep}
            pruned[key] = entry
        header = ALLOWLIST_PATH.read_text().splitlines()
        preamble = [ln for ln in header if ln.startswith("#") or not ln.strip()]
        lines = preamble + [
            json.dumps(entry, sort_keys=True)
            for entry in sorted(
                pruned.values(),
                key=lambda e: (e.get("category", ""), e["path"]),
            )
        ]
        ALLOWLIST_PATH.write_text("\n".join(lines) + "\n")
        print(f"verify: dropped {dropped} entries, narrowed {narrowed}")
        return 0
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="check",
        choices=("check", "seed", "verify-allowlist"),
    )
    parser.add_argument("--force", action="store_true", help="seed: overwrite existing")
    parser.add_argument("--prune", action="store_true", help="verify-allowlist: prune")
    parser.add_argument(
        "--commits-file",
        type=Path,
        help="check: newline-delimited commit ids whose changed blobs should be scanned",
    )
    parser.add_argument(
        "--refs",
        nargs="?",
        const=DEFAULT_REF_GLOB,
        metavar="GLOB",
        help=(
            "check: re-scan every blob version carried by the refs matching GLOB "
            f"(default {DEFAULT_REF_GLOB}) under the current policy"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "check: print per-rule counts instead of paths — for a job whose log is "
            "public, where the findings list is itself a map to unremediated leaks"
        ),
    )
    args = parser.parse_args(argv)
    if args.commits_file is not None and args.mode != "check":
        parser.error("--commits-file is only valid with check")
    if args.refs is not None and args.mode != "check":
        parser.error("--refs is only valid with check")
    if args.summary_only and args.mode != "check":
        parser.error("--summary-only is only valid with check")
    if args.refs is not None and args.commits_file is not None:
        parser.error("--refs and --commits-file scan different corpora; pick one")
    policy = load_policy(POLICY_PATH)
    if args.mode == "check":
        commits = None
        if args.commits_file is not None:
            commits = [
                line.strip()
                for line in args.commits_file.read_text().splitlines()
                if line.strip()
            ]
        refs = published_refs(args.refs) if args.refs is not None else None
        return cmd_check(policy, commits, refs, args.summary_only)
    if args.mode == "seed":
        return cmd_seed(policy, args.force)
    if args.mode == "verify-allowlist":
        return cmd_verify(policy, args.prune)
    return 2


if __name__ == "__main__":
    sys.exit(main())
