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
    plus an identity principal — `personal_email`, one line inside 56 other `personal_email` hits.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zlib

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
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Optional, Tuple

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
# A whole private SURFACE parked in the public repo (the severity-4 private_path class:
# ledger, admin, darklane, lanes) is an owner decision with a date, not a permanent fact.
# Before 2026-08-29 these entries were re-blessed hash-bump after hash-bump — four lanes in
# one week — and nothing ever asked when the parking ends. Entries in this class must name
# the extraction lane and an expiry date; past the date the exemption stops working and the
# finding is live again, so extending the parking is a conscious edit, never a side effect
# of re-pinning a hash. (The exception-lists-need-expiry lesson, applied to this gate.)
EXPIRY_KEY = "expires"
LANE_KEY = "lane"
# An entry for a reviewed HISTORICAL version: bytes that exist only on published refs (old lane
# branches, immutable release tags), never in the checkout. Every edit to an allowlisted private
# file leaves its old version on the tags that carry it, and the ref scan (`check --refs`) would
# report it forever, since tags are not rewritten. Such an entry exempts that version in the ref
# scan only: the checkout and commit scans ignore it, so the same bytes coming back into the tree
# are a new finding, and the checkout scan never calls it stale. `verify-allowlist` checks that
# the version still exists at some published ref and still trips the rules the entry names.
SCOPE_KEY = "scope"
REFS_ONLY = "refs_only"
# The comment over the file's refs_only section. `seed --force` and `verify-allowlist --prune`
# rewrite the file and put the section, under this comment, after the checkout's entries, so a
# rewrite never strands the comment away from the entries it describes.
REFS_ONLY_SECTION = (
    "# refs_only entries: reviewed HISTORICAL versions that exist only on published refs, never in",
    "# the checkout. Each one was read before it was listed and holds no secret; private content",
    "# (such as the business-tier files) is never listed here. `verify-allowlist` holds each one",
    "# to some published ref, run in a clone that mirrors origin's refs (boundary-refs.yml).",
)
EXPIRY_SEVERITY = 4
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
        expiry = entry.get(EXPIRY_KEY)
        if expiry is not None:
            if not isinstance(expiry, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", expiry
            ):
                raise SystemExit(
                    f"{path}:{line_no}: {EXPIRY_KEY} must be an ISO date (YYYY-MM-DD)"
                )
        lane = entry.get(LANE_KEY)
        if lane is not None and (not isinstance(lane, str) or not lane):
            raise SystemExit(f"{path}:{line_no}: {LANE_KEY} must be a non-empty string")
        scope = entry.get(SCOPE_KEY)
        if scope is not None and scope != REFS_ONLY:
            raise SystemExit(f"{path}:{line_no}: {SCOPE_KEY} must be {REFS_ONLY!r} when present")
        key = (entry["path"], entry["sha256"])
        if key in entries:
            raise SystemExit(f"{path}:{line_no}: duplicate allowlist key: {key[0]}")
        entries[key] = entry
    return entries


def is_refs_only(entry: dict) -> bool:
    return entry.get(SCOPE_KEY) == REFS_ONLY


def checkout_entries(allowlist: Dict[Tuple[str, str], dict]) -> Dict[Tuple[str, str], dict]:
    """The entries the checkout and commit scans honour: every one but the `refs_only` ones."""
    return {key: entry for key, entry in allowlist.items() if not is_refs_only(entry)}


def needs_expiry(entry: dict, policy: Policy) -> bool:
    """Whether this entry parks a structural private surface and so must carry a date.

    Keyed on the RULE'S class and declared rank, not on the entry's self-declared
    category: the demand covers whole private SURFACES (private_path rules ranked
    severity-4+), never secret-pattern pins, whose safety case lives in the reason text.
    The rank must be EXPLICIT: `load_policy` refuses an unranked rule, so in production
    every rule has one; a rule absent from the table only exists in hand-built test
    policies, and an implicit DEFAULT_SEVERITY here would put the expiry demand on
    fixtures that never declared a severity at all.

    A `refs_only` entry parks nothing: its bytes are a version on immutable refs, with no lane
    that could extract them, so it carries no date.
    """
    if is_refs_only(entry):
        return False
    return any(
        rule in policy.private_paths
        and policy.severity.get(rule, 0) >= EXPIRY_SEVERITY
        for rule in entry.get("rules") or ()
    )


def enforce_expiry_metadata(
    allowlist: Dict[Tuple[str, str], dict], policy: Policy
) -> None:
    """Fail closed when a severity-4 private-surface entry has no expiry or lane."""
    bad = [
        entry["path"]
        for entry in allowlist.values()
        if needs_expiry(entry, policy)
        and (entry.get(EXPIRY_KEY) is None or entry.get(LANE_KEY) is None)
    ]
    if bad:
        raise SystemExit(
            f"allowlist: {len(bad)} severity-{EXPIRY_SEVERITY}+ private-surface "
            f"entries lack {EXPIRY_KEY!r} and/or {LANE_KEY!r} (a parked private surface "
            f"is a dated owner decision, not a permanent fact): " + ", ".join(sorted(bad))
        )


def entry_expired(entry: dict, today: str) -> bool:
    """ISO dates compare lexically, so string comparison IS date comparison here."""
    expiry = entry.get(EXPIRY_KEY)
    return expiry is not None and expiry < today


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


def secret_candidate_files(
    patterns: Dict[str, str], paths: Optional[Iterable[str]] = None
) -> set[str]:
    """Use git's optimized PCRE walker to prefilter files with any structural secret hit."""
    selected = None if paths is None else list(paths)
    if not patterns or selected == []:
        return set()
    args = ["git"]
    if selected is not None:
        args.append("--literal-pathspecs")
    args.extend(("grep", "--text", "-z", "-l", "-P"))
    for source in patterns.values():
        args.extend(("-e", source))
    args.append("--")
    if selected is not None:
        args.extend(selected)
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


# ---------------------------------------------------------------------------------------------
# GZIP BLOBS ARE JUDGED BY THEIR TEXT (memra#1198). Deflate bytes are close to random, so a short
# name rule matches somewhere in a few hundred kilobytes of them by chance (15 of 90
# recompressions of ten receipts in the issue did), and a real leak inside a compressed receipt is
# invisible to a scan of the compressed bytes. A blob that starts with the gzip magic and
# decompresses is judged by its decompressed members instead: a tar's regular files, or the one
# stream. Binary content is left out by git's own test (a NUL in the first 8000 bytes), so tensor
# and float dumps, which would match by chance just as deflate bytes do, stay quiet. A blob that
# does not decompress is judged by its raw bytes, as before. The decompressed text is about seven
# times the compressed bytes in this repo (4 GB from 575 MB), so the prefilter over it is git's
# PCRE walker (`git grep --no-index` over temporary copies), not the per-rule Python pass, which
# only reads the members git grep flags.
GZIP_MAGIC = b"\x1f\x8b"
BINARY_PROBE = 8000
GREP_BATCH_BYTES = 512 << 20


def is_gzip_name(path: str) -> bool:
    return path.endswith((".gz", ".tgz"))


def has_gzip_magic(path: Path) -> bool:
    """The checkout scan finds a gzip blob by its magic, whatever its name; commit mode reads every
    changed blob and sees the magic in its bytes. The ref scan takes gzip blobs by name, plus any
    the raw prefilter already made candidates."""
    try:
        with path.open("rb") as fh:
            return fh.read(2) == GZIP_MAGIC
    except OSError:
        return False


# A blob's decompressed members may hold at most this many bytes (review round 1): a few MB of
# zeros expand to GBs, and the scan holds a member in memory. Over it, the blob is a finding of its
# own (`archive_over_scan_cap`), never a pass. Nested archives share their outer blob's budget, and
# a plain tar opened from a tar member is charged again for its copy (review round 3), so nested
# tars cannot hold a copy of the cap per level.
SCAN_CAP_BYTES = 1 << 30
# Containers opened below a blob's own gzip: a gzip and a tar each count one, so 7 is the outer
# tar.gz's tar and three nested tar.gz levels inside it. A gzip past the limit is judged by its
# bytes, as one that does not decompress is (`text_member`).
GZIP_NEST_DEPTH = 7
GZIP_HEADER_MEMBER = "<gzip header>"
TAR_TAIL_MEMBER = "<after the tar>"


class ScanCapExceeded(Exception):
    """A gzip blob decompresses past `SCAN_CAP_BYTES`."""


def gzip_header_fields(data: bytes, pos: int) -> Tuple[int, bytes]:
    """Parse one gzip member header at `pos` (RFC 1952): its end, and its FEXTRA, FNAME and
    FCOMMENT fields, which a reader of the decompressed stream never sees but the blob carries.
    Raises ValueError on a malformed header."""
    if data[pos : pos + 2] != GZIP_MAGIC or len(data) < pos + 10 or data[pos + 2] != 8:
        raise ValueError("not a deflate gzip member")
    flags = data[pos + 3]
    at = pos + 10
    fields: List[bytes] = []
    if flags & 0x04:  # FEXTRA
        if len(data) < at + 2:
            raise ValueError("truncated FEXTRA")
        xlen = int.from_bytes(data[at : at + 2], "little")
        fields.append(data[at + 2 : at + 2 + xlen])
        at += 2 + xlen
    for bit in (0x08, 0x10):  # FNAME, FCOMMENT: zero-terminated
        if flags & bit:
            end = data.find(b"\0", at)
            if end < 0:
                raise ValueError("unterminated header field")
            fields.append(data[at:end])
            at = end + 1
    if flags & 0x02:  # FHCRC
        at += 2
    if at > len(data):
        raise ValueError("truncated header")
    return at, b"\n".join(field for field in fields if field)


def gunzip_capped(data: bytes, budget: List[int]) -> Tuple[bytes, bytes]:
    """Decompress every member of a gzip stream within `budget[0]` bytes (decremented): the
    content and the members' header fields. Raises ScanCapExceeded past the budget, and
    ValueError, EOFError or zlib.error on a stream that does not decompress."""
    out: List[bytes] = []
    headers: List[bytes] = []
    pos = 0
    while pos < len(data):
        if data[pos : pos + 2] != GZIP_MAGIC:
            if data[pos:].strip(b"\0"):
                raise ValueError("trailing bytes after the gzip stream")
            break
        pos, fields = gzip_header_fields(data, pos)
        if fields:
            headers.append(fields)
        inflate = zlib.decompressobj(-zlib.MAX_WBITS)
        pending = data[pos:]
        while True:
            chunk = inflate.decompress(pending, budget[0] + 1)
            if len(chunk) > budget[0]:
                raise ScanCapExceeded()
            budget[0] -= len(chunk)
            out.append(chunk)
            if inflate.eof:
                break
            pending = inflate.unconsumed_tail
            if not pending:
                raise EOFError("truncated gzip member")
        pos = len(data) - len(inflate.unused_data) + 8  # the CRC32 and ISIZE trailer
    return b"".join(out), b"\n".join(headers)


def looks_like_tar(raw: bytes) -> bool:
    """A tar stream: the `ustar` magic, or a first 512-byte header whose checksum holds (a v7
    tar has no magic). Text and binary payloads fail the checksum."""
    if raw[257:262] == b"ustar":
        return True
    if len(raw) < 512:
        return False
    try:
        tarfile.TarInfo.frombuf(raw[:512], tarfile.ENCODING, "surrogateescape")
    except tarfile.HeaderError:
        return False
    return True


def container_members(
    raw: bytes, depth: int, budget: List[int], copied: bool = False
) -> Optional[List[Tuple[str, bytes]]]:
    """The members of a decompressed stream or of a tar member: a gzip inside it is opened in
    turn and a tar is read by its files, up to `GZIP_NEST_DEPTH` levels of nesting; None when
    `raw` is neither (it is then one member, `text_member`). `copied` says `raw` is a tar
    member's copy, which no decompression charged: a plain tar opened from it is charged its
    length first. Raises ScanCapExceeded."""
    if depth >= GZIP_NEST_DEPTH:
        return None
    if raw[:2] == GZIP_MAGIC:
        return gzip_members(raw, depth + 1, budget)
    if looks_like_tar(raw):
        if copied:
            if len(raw) > budget[0]:
                raise ScanCapExceeded()
            budget[0] -= len(raw)
        return tar_members(raw, depth + 1, budget)
    return None


def text_member(name: str, body: bytes) -> List[Tuple[str, bytes]]:
    """One member, left out when it is binary by git's own test (a NUL in the first 8000 bytes):
    tensor and float dumps would match by chance just as deflate bytes do. A gzip that was not
    opened (it does not decompress, or it is past `GZIP_NEST_DEPTH`) is kept and judged by its
    bytes, as a top-level one is (review round 3)."""
    if body[:2] == GZIP_MAGIC:
        return [(name, body)]
    return [] if b"\0" in body[:BINARY_PROBE] else [(name, body)]


def tar_members(raw: bytes, depth: int, budget: List[int]) -> List[Tuple[str, bytes]]:
    """A tar's regular files by their path inside it, each opened in turn when it is an archive,
    and the bytes after the tar's end that are not padding as a member of their own. A tar that
    does not parse is judged as one stream, whatever it holds. The tar's bytes were charged to the
    budget when they were decompressed; only a sparse member, which expands past its stored bytes,
    and a plain tar member opened in turn (`container_members`) are charged again."""
    files: List[Tuple[str, bytes]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for info in tar:
                if not info.isfile():
                    continue
                if info.issparse() and info.size > budget[0]:
                    raise ScanCapExceeded()
                handle = tar.extractfile(info)
                body = handle.read() if handle else b""
                if info.issparse():
                    budget[0] -= len(body)
                files.append((info.name, body))
            tail = raw[tar.offset :]
    except tarfile.TarError:
        return [("", raw)]
    members: List[Tuple[str, bytes]] = []
    for name, body in files:
        nested = container_members(body, depth, budget, copied=True)
        if nested is None:
            members.extend(text_member(name, body))
        else:
            members.extend((f"{name}/{inner}" if inner else name, b) for inner, b in nested)
    if tail.strip(b"\0"):
        members.append((TAR_TAIL_MEMBER, tail))
    return members


def gzip_members(
    data: bytes, depth: int = 0, budget: Optional[List[int]] = None
) -> Optional[List[Tuple[str, bytes]]]:
    """The members a gzip blob is judged by, or None when `data` is not a gzip blob that
    decompresses (it is then judged by its raw bytes). Raises ScanCapExceeded past the cap.

    - The gzip header's FEXTRA, FNAME and FCOMMENT fields, as a member of their own.
    - Every gzip member of the stream, decompressed and joined.
    - A gzip or a tar inside it (a plain or v7 tar too) is opened in turn, up to
      `GZIP_NEST_DEPTH` levels (`container_members`, `tar_members`).
    - Otherwise the one stream, named '', unless it is binary (`text_member`).

    The header fields, a tar's tail and a tar that does not parse are judged whatever they hold,
    as the raw scan judged them before."""
    if data[:2] != GZIP_MAGIC:
        return None
    budget = [SCAN_CAP_BYTES] if budget is None else budget
    try:
        raw, header = gunzip_capped(data, budget)
    except (ValueError, EOFError, zlib.error):
        return None
    members: List[Tuple[str, bytes]] = []
    if header:
        members.append((GZIP_HEADER_MEMBER, header))
    nested = container_members(raw, depth, budget)
    if nested is None:
        members.extend(text_member("", raw))
    else:
        members.extend(nested)
    return members


def scratch_dir() -> Optional[str]:
    """Where the decompressed copies go (review round 1): the git common dir, on the repository's
    disk, rather than a RAM-backed /tmp; the system's temporary dir outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return str((ROOT / result.stdout.strip()).resolve())


def members_with_hits(
    sources: Dict[str, str], blobs: List[Tuple[Hashable, List[Tuple[str, bytes]]]]
) -> Dict[Hashable, List[Tuple[str, bytes]]]:
    """The members, by blob key, in which any rule source matches, as `git grep --text -P` reads
    them. One git grep per batch of about 512 MiB of temporary copies."""
    flagged: Dict[Hashable, List[Tuple[str, bytes]]] = {}
    if not sources:
        return flagged
    args = ["git", "grep", "--no-index", "--text", "-z", "-l", "-P"]
    for source in sources.values():
        args.extend(("-e", source))
    args.extend(("--", "."))
    pending: List[Tuple[str, Hashable, Tuple[str, bytes]]] = []

    def run(tmp: str) -> None:
        result = subprocess.run(args, cwd=tmp, capture_output=True)
        if result.returncode not in (0, 1):
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git grep gzip prefilter failed: {message}")
        hit = {raw.decode() for raw in result.stdout.split(b"\0") if raw}
        for file_name, key, member in pending:
            if file_name in hit:
                flagged.setdefault(key, []).append(member)
            os.unlink(os.path.join(tmp, file_name))
        pending.clear()

    with tempfile.TemporaryDirectory(prefix="public-boundary-gz-", dir=scratch_dir()) as tmp:
        size = 0
        for key, members in blobs:
            for name, body in members:
                file_name = str(len(pending))
                with open(os.path.join(tmp, file_name), "wb") as fh:
                    fh.write(body)
                pending.append((file_name, key, (name, body)))
                size += len(body)
                if size >= GREP_BATCH_BYTES:
                    run(tmp)
                    size = 0
        if pending:
            run(tmp)
    return flagged


def scan_members(
    policy: Policy, members: List[Tuple[str, bytes]]
) -> List[Tuple[str, int, str]]:
    """The per-rule pass over the flagged members of one gzip blob, each hit's text prefixed with
    its member's name so a report says where inside the archive it is."""
    hits: List[Tuple[str, int, str]] = []
    for name, body in members:
        for rule, line, text in scan_secret_bytes(
            body, policy.secret_union, policy.secret_groups, policy.secret_patterns
        ):
            hits.append((rule, line, f"{name}: {text}" if name else text))
    hits.sort(key=lambda hit: (hit[1], hit[0]))
    return hits


def blobs_at(specs: List[str]) -> Dict[str, bytes]:
    """Map each `<ref>:<path>` spec to its bytes in one `git cat-file --batch` pass."""
    if not specs:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        # Paths come from git as bytes and are decoded with surrogateescape, so encode them back
        # the same way: a non-UTF-8 path name must not stop the pass.
        input="\n".join(specs).encode("utf-8", "surrogateescape") + b"\n",
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


def tree_candidates_at_refs(
    policy: Policy, refs: List[str]
) -> Dict[Tuple[str, str], set[str]]:
    """(path, oid) of every blob at `refs` that is gzip-named or under a private_paths glob, with
    the refs that carry it: one `git ls-tree` per ref, keeping the oids, so a blob shared by many
    refs is one entry.

    Private paths are matched here with `matches_glob`, as the checkout and commit scans match
    them (memra#1356): the old per-ref `git ls-tree -- ':(glob)...'` was refused by git
    ("pathspec magic not supported by this command: 'glob'"), and its silent `return []` meant
    the ref scan never reported a private path. A failed `git ls-tree` now stops the scan.

    The globs are one compiled regex (`glob_matcher`), and its verdict is memoized per path:
    published refs repeat the same paths (41 refs listed 6.7 M entries on the rig), and an
    fnmatch call per entry and glob took the pass from 4 s to 36 s there.
    """
    private_re = glob_matcher(policy.private_paths.values())
    private: Dict[bytes, bool] = {}
    groups: Dict[Tuple[str, str], set[str]] = {}
    for ref in refs:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "-z", ref], cwd=ROOT, capture_output=True
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git ls-tree {ref} failed: {message}")
        for entry in result.stdout.split(b"\0"):
            meta, _, raw_path = entry.partition(b"\t")
            if not raw_path:
                continue
            if not raw_path.endswith((b".gz", b".tgz")):
                hit = private.get(raw_path)
                if hit is None:
                    path = raw_path.decode("utf-8", errors="surrogateescape")
                    hit = private_re is not None and private_re.match(path) is not None
                    private[raw_path] = hit
                if not hit:
                    continue
            parts = meta.split()
            if len(parts) != 3 or parts[1] != b"blob":
                continue
            path = raw_path.decode("utf-8", errors="surrogateescape")
            groups.setdefault((path, parts[2].decode("ascii")), set()).add(ref)
    return groups


def oids_at(specs: List[str]) -> Dict[str, str]:
    """Map each `<ref>:<path>` spec to its blob oid in one `git cat-file --batch-check` pass."""
    if not specs:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=ROOT,
        # Paths come from git as bytes and are decoded with surrogateescape, so encode them back
        # the same way: a non-UTF-8 path name must not stop the pass.
        input="\n".join(specs).encode("utf-8", "surrogateescape") + b"\n",
        capture_output=True,
        check=True,
    )
    oids: Dict[str, str] = {}
    for spec, line in zip(specs, result.stdout.decode("utf-8", errors="replace").splitlines()):
        parts = line.split()
        if len(parts) == 2 and parts[1] == "blob":
            oids[spec] = parts[0]
    return oids


def evaluate_refs(
    policy: Policy, refs: List[str], chunk: int = 256, stats: Optional[Dict[str, int]] = None
) -> Tuple[List[Violation], Dict[Tuple[str, str], set[str]]]:
    """Evaluate every candidate blob version carried by `refs`.

    Each (path, oid) is read and judged once (review round 1 of memra#1198): published refs share
    almost all of their blobs, and judging per (ref, path) read the same blob once per ref that
    carries it (1,119 refs held 4,385 distinct gzip blobs in 1.27 M (ref, path) pairs). The refs
    carrying each violation come from the groups, so a report can still say where to go and remove
    it. Gzip blobs are candidates by name at every ref, and by their magic when the raw
    prefilter already made them candidates (the checkout scan sniffs every file's magic). Private
    paths are candidates by their glob at every ref (memra#1356).
    """
    groups: Dict[Tuple[str, str], set[str]] = tree_candidates_at_refs(policy, refs)
    ordered_pairs = sorted(secret_candidates_at_refs(policy.secret_sources, refs))
    oids = oids_at([f"{ref}:{path}" for ref, path in ordered_pairs])
    for ref, path in ordered_pairs:
        oid = oids.get(f"{ref}:{path}")
        if oid is not None:
            groups.setdefault((path, oid), set()).add(ref)

    violations: Dict[Tuple[str, str], Violation] = {}
    carriers: Dict[Tuple[str, str], set[str]] = {}
    ordered = sorted(groups)
    read = 0
    for start in range(0, len(ordered), chunk):
        batch = ordered[start : start + chunk]
        loaded = blobs_at([oid for _, oid in batch])
        # Keyed by (path, oid), never the oid alone (review round 2): one blob at two paths is two
        # verdicts, since the allowlist pins (path, sha256) and a private path keeps its own.
        gz = gzip_batch(
            policy,
            [
                ((path, oid), path, lambda oid=oid: loaded.get(oid))
                for path, oid in batch
                if is_gzip_name(path) or loaded.get(oid, b"")[:2] == GZIP_MAGIC
            ],
        )
        for path, oid in batch:
            data = loaded.get(oid)
            if data is None:
                continue
            read += 1
            key_blob = (path, oid)
            violation = gz[key_blob] if key_blob in gz else evaluate_content(policy, path, data)
            if violation is None:
                continue
            key = (violation.path, violation.sha256)
            violations.setdefault(key, violation)
            carriers.setdefault(key, set()).update(short_ref(ref) for ref in groups[(path, oid)])
    if stats is not None:
        # A scan that read zero blobs is a scan that proved nothing, and the difference between
        # "clean" and "never ran" has to be visible in a log rather than inferred from silence.
        stats[SCANNED_BLOBS] = read
        stats["candidates"] = len(ordered)
    return list(violations.values()), carriers




def worktree_blob_bytes(path: Path) -> bytes:
    """Read the bytes Git publishes for one tracked worktree entry.

    A symlink is a Git blob whose content is its target string. Following it is both wrong and
    non-portable: an absolute receipt link may resolve on the machine that committed it, fail on a
    clean clone, or raise PermissionError while pathlib asks whether the target is a file. Scan the
    link text itself, exactly as the commit/ref modes already scan the blob from git cat-file.
    """
    if path.is_symlink():
        return os.readlink(path).encode("utf-8", errors="surrogateescape")
    try:
        return path.read_bytes()
    except OSError as exc:
        rel = path.relative_to(ROOT)
        raise RuntimeError(f"cannot read tracked worktree entry {rel}: {exc}") from exc


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


def glob_matcher(globs: Iterable[str]) -> Optional[re.Pattern]:
    """One compiled regex that matches a path exactly when `matches_glob` matches it for one of
    `globs`, or None for no globs: the ref scan tests every entry of every ref, and one regex is
    much cheaper there than an fnmatch call per glob (review of memra#1356)."""
    forms = [fnmatch.translate(form) for glob in globs for form in (glob, glob.rstrip("*"))]
    return re.compile("|".join(f"(?:{form})" for form in forms)) if forms else None


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
    account id, the AdministratorAccess assertion and the identity principal beside it were never
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
    try:
        members = gzip_members(data)
    except ScanCapExceeded:
        return cap_violation(policy, rel, data)
    if members is not None:
        flagged = members_with_hits(policy.secret_sources, [(rel, members)]).get(rel, [])
        return gzip_violation(policy, rel, data, flagged)
    if not raw_bytes_prefilter(data, policy.secret_sources):
        # Same gate the checkout scan applies through `git grep --text -P` before it decodes
        # anything: a rule must match the RAW bytes. Without it the commit and ref scans judged
        # a blob the checkout scan never looked at, and `decode(errors="ignore")` glued the
        # neighbours of undecodable bytes into a provider name no reader sees (2026-09-20: two
        # gzipped trace logs refused by the pre-push hook at "line 1810" of the deflate stream,
        # then called stale pins by `check` and `verify-allowlist`). One candidate set, both scans.
        return None
    hits = scan_secret_bytes(
        data, policy.secret_union, policy.secret_groups, policy.secret_patterns
    )
    if not hits:
        return None
    return build_violation(
        policy, rel, hashlib.sha256(data).hexdigest(), "secret_pattern", hits
    )


SCAN_CAP_RULE = "archive_over_scan_cap"


def cap_violation(policy: Policy, rel: str, data: bytes) -> Violation:
    """A gzip blob that decompresses past the scan cap is a finding, not a pass: what lies past
    the cap was not read."""
    return Violation(
        path=rel,
        sha256=hashlib.sha256(data).hexdigest(),
        category="secret_pattern",
        detail=f"{SCAN_CAP_RULE} (decompresses past {SCAN_CAP_BYTES >> 20} MiB; not scanned)",
        rules=(SCAN_CAP_RULE,),
        severity=policy.rank(SCAN_CAP_RULE),
    )


def gzip_violation(
    policy: Policy, rel: str, data: bytes, flagged: List[Tuple[str, bytes]]
) -> Optional[Violation]:
    hits = scan_members(policy, flagged)
    if not hits:
        return None
    return build_violation(
        policy, rel, hashlib.sha256(data).hexdigest(), "secret_pattern", hits
    )


def gzip_batch(
    policy: Policy, items: Iterable[Tuple[Hashable, str, Callable[[], Optional[bytes]]]]
) -> Dict[Hashable, Optional[Violation]]:
    """Judge the gzip blobs among `items` (key, path, a reader of its bytes) a batch at a time:
    one prefilter per batch of about 512 MiB of decompressed members, so the decompressed text is
    never all in memory. Returns a verdict by key for every blob judged here; a private path, a
    bypassed path or a blob that does not decompress is left to the raw scan."""
    verdicts: Dict[Hashable, Optional[Violation]] = {}
    judged: Dict[Hashable, Tuple[str, bytes]] = {}
    blobs: List[Tuple[Hashable, List[Tuple[str, bytes]]]] = []
    size = 0

    def flush() -> None:
        flagged = members_with_hits(policy.secret_sources, blobs)
        for key, (rel, data) in judged.items():
            verdicts[key] = gzip_violation(policy, rel, data, flagged.get(key, []))
        judged.clear()
        blobs.clear()

    for key, rel, read in items:
        if is_bypass(rel, policy.bypass_paths):
            continue
        if any(matches_glob(rel, glob) for glob in policy.private_paths.values()):
            continue
        data = read()
        if data is None:
            continue
        try:
            members = gzip_members(data)
        except ScanCapExceeded:
            verdicts[key] = cap_violation(policy, rel, data)
            continue
        if members is None:
            continue
        judged[key] = (rel, data)
        blobs.append((key, members))
        size += len(data) + sum(len(body) for _, body in members)
        if size >= GREP_BATCH_BYTES:
            flush()
            size = 0
    if judged:
        flush()
    return verdicts


_RAW_RULES: Dict[str, Optional[re.Pattern[bytes]]] = {}


def raw_bytes_prefilter(data: bytes, sources: Dict[str, str]) -> bool:
    """True when any rule source matches the raw bytes, as `git grep --text -P` would.

    The checkout scan prefilters candidates with git's PCRE walker over bytes; this is the same
    question asked of a loose blob. A source that does not compile as a bytes pattern keeps the
    blob a candidate (fail open into the text scan, never silently out of it).
    """
    for name, source in sources.items():
        if name not in _RAW_RULES:
            try:
                _RAW_RULES[name] = re.compile(source.encode("utf-8"))
            except (re.error, UnicodeEncodeError):
                _RAW_RULES[name] = None
        pattern = _RAW_RULES[name]
        if pattern is None or pattern.search(data) is not None:
            return True
    return False


def evaluate(
    policy: Policy, paths: Optional[Iterable[str]] = None
) -> List[Violation]:
    violations: List[Violation] = []
    files = tracked_files()
    if paths is None:
        secret_candidates = secret_candidate_files(policy.secret_sources)
    else:
        selected = set(paths)
        files = [rel for rel in files if rel in selected]
        secret_candidates = secret_candidate_files(policy.secret_sources, files)
    gz_names: List[str] = []
    for rel in files:
        full = ROOT / rel
        # Do not ask is_file() to follow a symlink. Python 3.12 raises PermissionError when an
        # absolute target crosses an unreadable parent (the hosted-runner failure that motivated
        # this guard); the tracked object is still a perfectly readable symlink blob.
        is_link = full.is_symlink()
        if not is_link:
            # A NON-symlink we cannot stat is REPORTED, never raised and never skipped
            # (lane/glm5-accrace, merged with the symlink guard above). Raising kills the
            # scan for every LATER tracked file — the exact shape of the CI failure both
            # halves of this guard exist to fix — and a silent `continue` would let an
            # unreadable tracked path hide anything at all. So it becomes its own finding
            # and the census runs to the end of the tree. Teeth: `UnstatablePathTests`.
            try:
                is_file = full.is_file()
            except OSError as err:
                # Built directly rather than through `build_violation`: that helper derives
                # `detail` from matched RULE NAMES and line numbers, and the one fact worth
                # reporting here is the errno text. `detail` keeps the allowlist's
                # long-standing "rules (note)" shape, so this category needs no migration.
                violations.append(
                    Violation(
                        path=rel,
                        sha256="unstatable",
                        category="unstatable_path",
                        detail=(
                            f"unstatable_path ({type(err).__name__}: "
                            f"{err.strerror or err})"
                        ),
                        rules=("unstatable_path",),
                        severity=policy.rank("unstatable_path"),
                    )
                )
                continue
            if not is_file:
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
                    hashlib.sha256(worktree_blob_bytes(full)).hexdigest(),
                    "private_path",
                    [(name, 0, "") for name in matched_globs],
                )
            )
            continue  # one violation per file — path glob wins over secret hits.
        # Secret regex rule.
        # git grep does not prefilter symlink blobs, so the small tracked-symlink set is scanned
        # directly. This keeps the fast prefilter for ordinary files without making links a blind
        # spot.
        if not bypass and not is_link and (is_gzip_name(rel) or has_gzip_magic(full)):
            # Judged together after the walk (`gzip_batch`); one that does not decompress comes
            # back to the raw scan below.
            gz_names.append(rel)
            continue
        if not bypass and (rel in secret_candidates or is_link):
            data = worktree_blob_bytes(full)
            hits = scan_secret_bytes(
                data,
                policy.secret_union,
                policy.secret_groups,
                policy.secret_patterns,
            )
            if hits:
                violations.append(
                    build_violation(
                        policy,
                        rel,
                        hashlib.sha256(data).hexdigest(),
                        "secret_pattern",
                        hits,
                    )
                )
    gz = gzip_batch(
        policy,
        [(rel, rel, lambda rel=rel: worktree_blob_bytes(ROOT / rel)) for rel in gz_names],
    )
    for rel in gz_names:
        if rel in gz:
            if gz[rel] is not None:
                violations.append(gz[rel])
            continue
        if rel in secret_candidates:
            data = worktree_blob_bytes(ROOT / rel)
            hits = scan_secret_bytes(
                data, policy.secret_union, policy.secret_groups, policy.secret_patterns
            )
            if hits:
                violations.append(
                    build_violation(
                        policy, rel, hashlib.sha256(data).hexdigest(), "secret_pattern", hits
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
    enforce_expiry_metadata(allowlist, policy)
    today = datetime.date.today().isoformat()
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
    # A `refs_only` entry exempts a historical version in the ref scan alone: in the checkout or
    # a pushed commit the same bytes are back in the tree, which is a new finding.
    exempting = allowlist if refs is not None else checkout_entries(allowlist)
    unmatched: List[Violation] = []
    grandfathered = 0
    partial = 0
    expired = 0
    for v in violations:
        listed = exempting.get((v.path, v.sha256))
        if listed is not None and entry_expired(listed, today):
            # The exemption died on its date. Resurface the finding with the parking
            # record attached, so the reader extends the date consciously (with the
            # lane's state in hand) instead of re-pinning what looks like a fresh hit.
            expired += 1
            v.detail = (
                f"{v.detail} [grandfather EXPIRED {listed[EXPIRY_KEY]}; "
                f"lane: {listed.get(LANE_KEY, 'unnamed')}]"
            )
            unmatched.append(v)
            continue
        exempt = exempt_rules(exempting, v)
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
    if expired:
        print(
            f"public-boundary: {expired} of the new findings are EXPIRED grandfathers — "
            f"the parking date passed; extend {EXPIRY_KEY!r} deliberately or migrate."
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
    stale = stale_entries(checkout_entries(allowlist), violations)
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


def allowlist_lines(preamble: List[str], entries: Iterable[dict]) -> List[str]:
    """The allowlist file's lines: the preamble's comments, the checkout's entries, then the
    `refs_only` entries under REFS_ONLY_SECTION."""
    entries = list(entries)

    def rows(chosen: List[dict]) -> List[str]:
        ordered = sorted(chosen, key=lambda e: (e.get("category", ""), e["path"]))
        return [json.dumps(entry, sort_keys=True) for entry in ordered]

    lines = [ln for ln in preamble if ln not in REFS_ONLY_SECTION]
    lines += rows([e for e in entries if not is_refs_only(e)])
    historical = [e for e in entries if is_refs_only(e)]
    if historical:
        lines += [*REFS_ONLY_SECTION, *rows(historical)]
    return lines


def cmd_seed(policy: Policy, force: bool) -> int:
    if ALLOWLIST_PATH.exists() and not force:
        print(
            f"seed: refusing to overwrite existing {ALLOWLIST_PATH.name}; "
            f"pass --force to regenerate."
        )
        return 1
    violations = evaluate(policy)
    # Reviewed historical versions are not in the checkout, so a reseed cannot regenerate them.
    kept = [e for e in load_allowlist(ALLOWLIST_PATH).values() if is_refs_only(e)]
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
    if kept:
        lines += [*REFS_ONLY_SECTION, *(json.dumps(entry, sort_keys=True) for entry in kept)]
    ALLOWLIST_PATH.write_text("\n".join(lines) + "\n")
    print(
        f"seed: wrote {len(violations)} entries to {ALLOWLIST_PATH.relative_to(ROOT)}"
        + (f", and kept {len(kept)} {REFS_ONLY} entries" if kept else "")
    )
    return 0


def is_shallow() -> bool:
    out = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return out == "true"


def ref_set_looks_complete(refs: List[str]) -> bool:
    """Whether these published refs look like origin's whole set: its branches AND GitHub's PR
    heads, as boundary-refs.yml fetches them. A plain clone has no `refs/remotes/origin-pull/*`.
    The test cannot see a ref that is here but not on origin (a local-only tag counts as
    published), so only a clone that mirrors origin's refs judges `refs_only` entries for good."""
    return any(ref.startswith("refs/remotes/origin/") for ref in refs) and any(
        ref.startswith("refs/remotes/origin-pull/") for ref in refs
    )


def historical_violations(
    policy: Policy, entries: Dict[Tuple[str, str], dict], refs: List[str]
) -> Tuple[List[Violation], List[Tuple[str, str, str]]]:
    """The violations of the `refs_only` entries' versions, as found at the published refs: each
    entry's path is read at every ref (one `git cat-file` pass), and a version is judged when its
    bytes are the pinned ones.

    Also returns the versions that trip a rule the entry does not name. A private path's
    violation names its globs only (evaluate_content never runs the secret rules on it), so the
    secret rules are run on it here: without that, a secret inside a listed private version would
    be exempted unseen."""
    paths = sorted({path for path, _ in entries})
    oids = oids_at([f"{ref}:{path}" for ref in refs for path in paths])
    versions = sorted({(spec.split(":", 1)[1], oid) for spec, oid in oids.items()})
    loaded = blobs_at(sorted({oid for _, oid in versions}))
    found: List[Violation] = []
    unnamed: List[Tuple[str, str, str]] = []
    for path, oid in versions:
        data = loaded.get(oid)
        digest = hashlib.sha256(data).hexdigest() if data is not None else ""
        entry = entries.get((path, digest))
        if entry is None:
            continue
        violation = evaluate_content(policy, path, data)
        if violation is None:
            continue
        if violation.category == "private_path":
            hits = scan_secret_bytes(
                data, policy.secret_union, policy.secret_groups, policy.secret_patterns
            )
            if hits:
                # The secret rules these bytes trip are live rules of this version as much as
                # its globs: the entry has to name them (a reviewed placeholder, say).
                globs = [(rule, 0, "") for rule in violation_rules(violation)]
                violation = build_violation(policy, path, digest, "private_path", globs + hits)
        found.append(violation)
        extra = sorted(set(violation_rules(violation)) - set(entry.get("rules") or ()))
        if extra:
            unnamed.append(
                (path, digest, f"rules the entry does not name: {','.join(extra)}")
            )
    return found, sorted(set(unnamed))


def cmd_verify(policy: Policy, prune: bool) -> int:
    allowlist = load_allowlist(ALLOWLIST_PATH)
    enforce_expiry_metadata(allowlist, policy)
    checkout = checkout_entries(allowlist)
    historical = {key: entry for key, entry in allowlist.items() if key not in checkout}
    # Drift concerns declared exemptions only. The separate check command still scans
    # every tracked blob for new violations, including paths with no exemption.
    live_violations = evaluate(policy, paths={path for path, _ in checkout})
    drifted = stale_entries(checkout, live_violations)
    unchecked = 0
    complete = False
    unnamed: List[Tuple[str, str, str]] = []
    if historical and is_shallow():
        # A shallow clone (ci.yml's boundary job) holds no published history to look in.
        # boundary-refs.yml runs this mode on a full clone with every published ref.
        unchecked = len(historical)
        print(
            f"verify: {unchecked} {REFS_ONLY} entries not checked here: a shallow clone has no "
            f"published history (boundary-refs.yml checks them)."
        )
    elif historical:
        refs = published_refs(DEFAULT_REF_GLOB)
        if not refs:
            # Fail closed, as the ref scan does: no published ref here means nothing was judged.
            print(
                f"verify: no refs matched {DEFAULT_REF_GLOB}, so {len(historical)} {REFS_ONLY} "
                f"entries were not judged. Fetch origin's refs as boundary-refs.yml does."
            )
            return 1
        complete = ref_set_looks_complete(refs)
        found, unnamed = historical_violations(policy, historical, refs)
        live_violations += found
        drifted += [
            (path, digest, why.replace("no live violation", "at no published ref, or no violation"))
            for path, digest, why in stale_entries(historical, found)
        ]
        drifted += unnamed
    if not drifted:
        print(
            f"verify: {len(checkout)} allowlist entries all pin live tracked files "
            f"for the rules they name."
        )
        if historical and not unchecked:
            print(
                f"verify: {len(historical)} {REFS_ONLY} entries all pin versions at published "
                f"refs for the rules they name."
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
        refused = 0
        drifted_keys = {(path, digest) for path, digest, _ in drifted}
        for key, entry in allowlist.items():
            if is_refs_only(entry) and (unchecked or not complete):
                # Not judged here, or judged against a ref set that may lack the refs carrying
                # it: a clone without origin's PR heads drifts every such entry, and pruning
                # there would drop reviewed entries that CI's refs still need.
                pruned[key] = entry
                refused += not unchecked and key in drifted_keys
                continue
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
        lines = allowlist_lines(preamble, pruned.values())
        ALLOWLIST_PATH.write_text("\n".join(lines) + "\n")
        print(f"verify: dropped {dropped} entries, narrowed {narrowed}")
        if refused:
            print(
                f"verify: kept {refused} {REFS_ONLY} entries unpruned: the published refs here do "
                f"not look like origin's whole set (no refs/remotes/origin/* or no "
                f"refs/remotes/origin-pull/*). Prune them in a clone that mirrors origin's refs."
            )
        # A secret rule an entry does not name is a finding, not a dead rule: no prune fixes it.
        return 1 if refused or unnamed else 0
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
