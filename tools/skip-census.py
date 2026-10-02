#!/usr/bin/env python3
"""Skip census: a Rust test that skips must be counted, named, and fatal by default.

WHY THIS FILE EXISTS (GATE-INTEGRITY-20260819 section 10, round 2's own "still green-but-blind"
list). `nv27b_twin_parity` read:

    if !st_dir.join("model.safetensors.index.json").exists() || !twin.exists() {
        eprintln!("SKIP: ckpt/twin absent");
        return;
    }

and the test PASSES. That was defensible while nothing consumed it. Round 2 then wired
`cargo test -p memra-gguf ... --lib` into .github/workflows/ci.yml, and a hosted runner has no
checkpoints at all, so from that commit onward CI reports `90 passed` in perpetuity while the
model-backed parity never ran once. It is A-2's shape in `#[test]` form ("ALL GREEN (N cells,
M skipped)" matching `grep -q "ALL GREEN"`).

THE SHAPE, copied deliberately from round 2's kernel-check fix: the harness parses the run's own
output, counts the skips, prints them, and compares the count with a NAMED budget that defaults
to 0. Raising the budget is allowed and is the escape hatch for a developer without artifacts,
but it must name the number, and the number appears in the verdict. An accounted skip, never an
invisible one.

THE SKIP PROTOCOL (memra #484). Every test that skips prints exactly one form:

    eprintln!("SKIP[<artifact>]: <reason>");

`<artifact>` names what was missing (a path, `CUDA device`, `2 CUDA devices`, an env var) and
`<reason>` says what the test would have proven with it. A per-case test (a loop over staged
fixtures that `continue`s past a missing one) prints one line PER MISSING CASE with that case's
artifact, so a run where zero of three cases executed counts three skips, not one green test.
Before #484 the census matched only `eprintln!("SKIP...")` in capitals, and five tests in the
CUDA-free crates skipped with `skipping:` or `skip:` and were counted by nothing: the audit log
carried the skips while the static census reported zero for memra-tokenizer and memra-reference.
Now any print in test code whose literal carries the word skip (any case) is a skip site: in the
protocol form it is a census row, in any other form it FAILS as unstructured.

Three assertions, because a count alone can go blind in three different ways:

  run     the suite's own verdict is asserted first (cargo's exit status, every
          `test result:` line says ok., the run is not vacuous or name-filtered), then the
          skips are counted against the budget. A protocol line must match a declared row for
          the same test IN FULL (format placeholders match any text); a skip-shaped line that is
          not in the protocol form FAILS.
  verify  the STATIC census over EVERY crate under crates/ (src/ and tests/), compared with
          tools/skip-census.tsv in BOTH directions: an undeclared skipping test FAILS, a stale
          row FAILS, a row whose file no longer holds the test FAILS, an unstructured skip print
          FAILS, and a skip print in a test helper that the census cannot attribute FAILS.
  report  the census file the shell gates append to (MEMRA_SKIP_CENSUS) asserted against an
          expected count, for batteries that mix Rust tests and generated gate scripts.

NO LINE ANCHORS (memra #484). The manifest used to carry `file:line` and reported drift as a
NOTE; by 2026-09-19 all 14 memra-gguf rows had drifted (several by ~1,900 lines) and the NOTE
fired on every row, so nobody read it. A row is identified by crate + test path + message, and
its third column is the FILE only: a file that no longer holds the test is a failure, and a
`file:line` anchor is refused outright so a stale one cannot be paired silently.

`run` uses `--test-threads=1 --nocapture`, and that is load-bearing rather than tidy: with
parallel threads libtest interleaves un-attributed output, so a SKIP line cannot be tied to the
test that emitted it. Single-threaded, libtest writes `test <path> ... ` with no newline before
the test body runs, so the first SKIP text lands on that same line and attribution is exact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tools" / "skip-census.tsv"
CRATES_DIR = "crates"

# A print macro with a string literal as its first argument, possibly on the next line.
PRINT_RE = re.compile(r'\b(?:e?print(?:ln)?)!\(\s*"((?:[^"\\]|\\.)*)"', re.S)
# The word skip in any case and inflection. Deliberately broad: a false positive is a test that
# says "skip" in a print, which is cheap to rephrase; a false negative is an invisible skip.
SKIP_WORD_RE = re.compile(r"(?i)\bskip(?:s|ped|ping)?\b")
# The one accepted form. The artifact is non-empty and bracket-free; the reason is non-empty.
PROTOCOL_RE = re.compile(r"^SKIP\[(?P<artifact>[^\]\s][^\]]*)\]: (?P<reason>\S.*)$")
FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)\s*[<(]")
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*\{")
ATTR_OR_DOC_RE = re.compile(r"^\s*(?:#\[|#!\[|///|//!|//|$)")
TEST_ATTR_RE = re.compile(r"#\[\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*test\s*[\](]")
CFG_TEST_RE = re.compile(r"#\[\s*cfg\s*\(\s*(?:all\s*\(\s*)?test\b")
TEST_LINE_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s*(.*)$")
RESULT_RE = re.compile(
    r"^test result:\s+(?P<verdict>\S+)\.?\s+(?P<passed>\d+) passed;\s+(?P<failed>\d+) failed;"
    r"\s+(?P<ignored>\d+) ignored;\s+(?P<measured>\d+) measured;\s+(?P<filtered>\d+) filtered"
)
# A run-side line that LOOKS like a skip: it starts with the word. Anchored at the start so a
# production log line that mentions a skipped step mid-sentence ("[kv-vmm] trim skipped: ...")
# is not a test skip; the static census is the strong guard for anything phrased otherwise.
RUN_SKIP_START_RE = re.compile(r"(?i)^skip(?:s|ped|ping)?\b")
LINE_ANCHOR_RE = re.compile(r":\d+$")

# Registered skip helpers: a function in test code that prints the protocol line on behalf of
# its caller. A #[test] that calls one is a census row carrying the helper's message; a skip
# print in any OTHER non-test function in test code fails, because the census cannot attribute
# it to a test and so cannot hold it to the manifest.
SKIP_HELPERS = {"skip_unless_native_pair"}

# The tree carries dozens of artifact-gated tests. A static census that finds nothing is a
# broken scan, and it would certify a blind suite as fully covered.
STATIC_FLOOR = 1


class CensusError(RuntimeError):
    """The census cannot be trusted, so it refuses rather than reporting a number."""


def rust_code_view(text: str, string_spans: list[tuple[int, int]] | None = None) -> str:
    """Mask strings/chars/comments without moving source positions or line breaks.

    This is a scope lexer, not a Rust parser. Lifetimes remain code. Nested block
    comments, escaped quotes and raw/byte strings cannot contribute fake braces.
    """
    result = list(text)
    length, i = len(text), 0
    raw_pattern = re.compile(r'(?:b|c)?r(\#{0,255})"')
    char_pattern = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'")

    def mask(start: int, end: int) -> None:
        for j in range(start, end):
            if result[j] != "\n":
                result[j] = " "

    while i < length:
        start = i
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = length if end < 0 else end
        elif text.startswith("/*", i):
            depth = 1
            i += 2
            while i < length and depth:
                if text.startswith("/*", i):
                    depth += 1; i += 2
                elif text.startswith("*/", i):
                    depth -= 1; i += 2
                else:
                    i += 1
            if depth:
                raise CensusError("unterminated Rust block comment")
        else:
            raw = raw_pattern.match(text, i) if text[i] in 'brc' else None
            if raw and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
                terminator = '"' + raw.group(1)
                end = text.find(terminator, raw.end())
                if end < 0:
                    raise CensusError("unterminated Rust raw string")
                i = end + len(terminator)
                if string_spans is not None:
                    string_spans.append((start, i))
            elif text[i] == '"':
                i += 1
                while i < length:
                    if text[i] == '\\':
                        i += 2
                    elif text[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                else:
                    raise CensusError("unterminated Rust string")
                if string_spans is not None:
                    string_spans.append((start, i))
            elif text[i] == "'":
                char = char_pattern.match(text, i)
                if char:
                    i = char.end()
                else:
                    i += 1
                    continue  # lifetime/label, not a quoted character
            else:
                i += 1
                continue
        mask(start, i)
    return ''.join(result)


def rust_unescape(literal: str) -> str:
    """The subset of Rust string escapes a skip message uses."""
    # A backslash at the end of a line continues the literal and eats the next line's indent.
    literal = re.sub(r"\\\n\s*", "", literal)
    return re.sub(r'\\(["\\nt])', lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), literal)


def template_regex(message: str) -> re.Pattern[str]:
    """A full-match regex for the printed form of a source message.

    `{path}` / `{x:?}` placeholders match any non-empty text, `{{` and `}}` are literal braces,
    everything else is literal. A full match, not a prefix: a prefix match lets
    `SKIP[{path}]: ...` accept any line that starts with `SKIP[`.
    """
    out = []
    i = 0
    while i < len(message):
        if message.startswith("{{", i):
            out.append(re.escape("{"))
            i += 2
        elif message.startswith("}}", i):
            out.append(re.escape("}"))
            i += 2
        elif message[i] == "{":
            end = message.find("}", i)
            if end < 0:
                raise CensusError(f"unbalanced format placeholder in {message!r}")
            out.append(".+?")
            i = end + 1
        else:
            out.append(re.escape(message[i]))
            i += 1
    return re.compile("".join(out))


def crate_names() -> list[str]:
    root = ROOT / CRATES_DIR
    if not root.is_dir():
        raise CensusError(f"no {root} directory")
    return sorted(p.name for p in root.iterdir() if (p / "src").is_dir())


def crate_roots(crate: str) -> list[tuple[Path, bool]] | None:
    """(root, integration) pairs for the crate, or None when the crate does not exist."""
    src = ROOT / CRATES_DIR / crate / "src"
    if not src.is_dir():
        return None
    roots = [(src, False)]
    tests = ROOT / CRATES_DIR / crate / "tests"
    if tests.is_dir():
        roots.append((tests, True))
    return roots


def file_module_path(rel: Path, integration: bool) -> tuple[str, ...]:
    """The module prefix libtest prints for a test defined in this file.

    src/: the path under src is the module path (`source/hy3.rs` -> `source::hy3`), with
    `lib.rs`, `main.rs` and `mod.rs` naming their parent. tests/: every top-level entry is its
    own binary and libtest prints paths RELATIVE TO THAT BINARY, so `tests/parity.rs` and
    `tests/contracts/mod.rs` (or `main.rs`) are the root of their binary and print no prefix,
    while `tests/contracts/transfer.rs` prints `transfer::...`.
    """
    parts = list(rel.parts[:-1])
    if integration:
        if not parts:
            return ()
        parts = parts[1:]
    if rel.stem not in ("lib", "main", "mod"):
        parts.append(rel.stem)
    return tuple(parts)


def _attr_block(lines: list[str], fn_line: int) -> list[str]:
    """The attribute/doc lines directly above a fn or mod declaration."""
    block = []
    back = fn_line - 1
    while back >= 0 and ATTR_OR_DOC_RE.match(lines[back]):
        block.append(lines[back])
        back -= 1
    return block


def _scan_file(crate: str, root: Path, path: Path, integration: bool, helpers: dict[str, str]):
    """-> (rows, unstructured, helper_defs) for one file.

    rows: skip sites in #[test] fns, in the protocol form.
    unstructured: skip prints that break the protocol or cannot be attributed.
    helper_defs: registered helper name -> its protocol message, found in this file.
    """
    rel_root = path.relative_to(root)
    file_mods = file_module_path(rel_root, integration)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    code = rust_code_view(text)
    code_lines = code.splitlines()
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line) + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Module path and cfg(test) scope by brace depth, so rows carry the name libtest prints.
    mod_at_line: list[tuple[str, ...]] = []
    test_scope_at_line: list[bool] = []
    stack: list[tuple[str, int, bool]] = []
    depth = 0
    for index, line in enumerate(code_lines):
        while stack and stack[-1][1] >= depth:
            stack.pop()
        mod_at_line.append(tuple(name for name, _, _ in stack))
        test_scope_at_line.append(any(flag for _, _, flag in stack))
        mod_match = MOD_RE.match(line)
        if mod_match:
            is_test = any(CFG_TEST_RE.search(a) for a in _attr_block(code_lines, index))
            stack.append((mod_match.group(1), depth, is_test))
        depth += line.count("{") - line.count("}")

    where_file = path.relative_to(ROOT).as_posix()
    file_is_test = integration or rel_root.stem == "tests" or rel_root.stem.endswith("_tests")

    def enclosing_fn(index: int) -> tuple[str, int] | None:
        for back in range(index, -1, -1):
            fn_match = FN_RE.match(code_lines[back])
            if fn_match:
                return fn_match.group(1), back
        return None

    def is_test_fn(fn_line: int) -> bool:
        return any(TEST_ATTR_RE.search(a) for a in _attr_block(code_lines, fn_line))

    rows: list[dict[str, str]] = []
    unstructured: list[dict[str, str]] = []
    helper_defs: dict[str, str] = {}

    for match in PRINT_RE.finditer(text):
        if not re.match(r'(?:e?print(?:ln)?)!', code[match.start():]):
            continue
        literal = match.group(1)
        if not SKIP_WORD_RE.search(literal):
            continue
        index = line_of(match.start())
        if lines[index].lstrip().startswith("//"):
            continue
        message = rust_unescape(literal)
        found = enclosing_fn(index)
        in_test_code = file_is_test or test_scope_at_line[index]
        if found is None:
            if in_test_code:
                raise CensusError(
                    f"{where_file}:{index + 1}: a skip print with no enclosing fn; the census "
                    "cannot attribute it, and an unattributable skip is exactly what this gate "
                    "exists to make impossible"
                )
            continue
        fn_name, fn_line = found
        test_path = "::".join((*file_mods, *mod_at_line[fn_line], fn_name))
        if is_test_fn(fn_line):
            if PROTOCOL_RE.match(message):
                rows.append({"crate": crate, "test": test_path, "where": where_file, "message": message})
            else:
                unstructured.append(
                    {
                        "crate": crate,
                        "test": test_path,
                        "where": f"{where_file}:{index + 1}",
                        "message": message,
                        "why": "not in the protocol form SKIP[<artifact>]: <reason>",
                    }
                )
        elif fn_name in SKIP_HELPERS:
            if not PROTOCOL_RE.match(message):
                unstructured.append(
                    {
                        "crate": crate,
                        "test": test_path,
                        "where": f"{where_file}:{index + 1}",
                        "message": message,
                        "why": "a registered skip helper that is not in the protocol form",
                    }
                )
            helper_defs[fn_name] = message
        elif in_test_code:
            unstructured.append(
                {
                    "crate": crate,
                    "test": test_path,
                    "where": f"{where_file}:{index + 1}",
                    "message": message,
                    "why": (
                        "a skip print in a test helper that is not a registered skip helper "
                        "(SKIP_HELPERS); the census cannot tie it to a test"
                    ),
                }
            )

    # #[test] fns that call a registered helper are rows with the helper's message.
    for name in SKIP_HELPERS:
        call = re.compile(rf"\b{re.escape(name)}\s*\(")
        for index, line in enumerate(code_lines):
            if not call.search(line) or FN_RE.match(line) or line.lstrip().startswith("//"):
                continue
            found = enclosing_fn(index)
            if found is None or not is_test_fn(found[1]):
                continue
            fn_name, fn_line = found
            rows.append(
                {
                    "crate": crate,
                    "test": "::".join((*file_mods, *mod_at_line[fn_line], fn_name)),
                    "where": where_file,
                    "message": f"@helper:{name}",
                }
            )
    return rows, unstructured, helper_defs


def static_census(crate: str) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    """(rows, unstructured) for the crate, or None when the crate has no source directory."""
    roots = crate_roots(crate)
    if roots is None:
        return None
    rows: list[dict[str, str]] = []
    unstructured: list[dict[str, str]] = []
    helpers: dict[str, str] = {}
    for root, integration in roots:
        for path in sorted(root.rglob("*.rs")):
            r, u, h = _scan_file(crate, root, path, integration, helpers)
            rows.extend(r)
            unstructured.extend(u)
            helpers.update(h)
    resolved = []
    for row in rows:
        if row["message"].startswith("@helper:"):
            name = row["message"].split(":", 1)[1]
            if name not in helpers:
                raise CensusError(
                    f"{row['where']}: {row['test']} calls skip helper {name} but no definition "
                    f"printing a skip was found in {crate}"
                )
            row = {**row, "message": helpers[name]}
        resolved.append(row)
    # One row per (test, message): a test that prints the same message at two sites is one row.
    unique = {}
    for row in resolved:
        unique.setdefault((row["test"], row["message"]), row)
    return list(unique.values()), unstructured


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise CensusError(f"missing manifest {MANIFEST}")
    rows = []
    for lineno, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise CensusError(
                f"{MANIFEST}:{lineno}: expected 4 tab-separated columns "
                f"(crate, test, file, message), got {len(cols)}"
            )
        if LINE_ANCHOR_RE.search(cols[2]):
            raise CensusError(
                f"{MANIFEST}:{lineno}: {cols[2]!r} carries a line anchor. Rows name the FILE "
                "only (memra #484: line anchors drifted on every edit and were paired silently). "
                "Regenerate the row from `tools/skip-census.py static`."
            )
        if not PROTOCOL_RE.match(cols[3]):
            raise CensusError(
                f"{MANIFEST}:{lineno}: message {cols[3]!r} is not in the protocol form "
                "SKIP[<artifact>]: <reason>"
            )
        rows.append({"crate": cols[0], "test": cols[1], "where": cols[2], "message": cols[3]})
    return rows


def _selected_crates(requested: list[str] | None) -> list[str]:
    names = crate_names()
    for crate in requested or []:
        if crate not in names:
            raise CensusError(f"no such crate source directory: {ROOT / CRATES_DIR / crate / 'src'}")
    return names


def cmd_static(args: argparse.Namespace) -> int:
    crates = args.crate or _selected_crates(None)
    total = 0
    bad = 0
    for crate in crates:
        census = static_census(crate)
        if census is None:
            raise CensusError(f"no such crate source directory: {ROOT / CRATES_DIR / crate / 'src'}")
        rows, unstructured = census
        for row in sorted(rows, key=lambda r: (r["test"], r["message"])):
            print(f"{row['crate']}\t{row['test']}\t{row['where']}\t{row['message']}")
        for site in unstructured:
            print(f"# UNSTRUCTURED {site['crate']} {site['test']} ({site['where']}): {site['message']!r}", file=sys.stderr)
        total += len(rows)
        bad += len(unstructured)
    print(f"# {total} protocol skip row(s), {bad} unstructured site(s) in {len(crates)} crate(s)", file=sys.stderr)
    if total + bad < STATIC_FLOOR and not args.crate:
        print(
            f"skip-census: ERROR: static census found {total} site(s), floor is {STATIC_FLOOR}; "
            "the scan is broken, not the tree",
            file=sys.stderr,
        )
        return 2
    return 1 if bad else 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Every crate under crates/ against the manifest, both directions.

    `--crate` names crates the caller requires to exist; the census itself always covers the
    whole tree, because a skip is blind wherever it lives and a partial verify is how the
    tokenizer and reference skips stayed uncounted.
    """
    manifest = read_manifest()
    failures = 0
    seen: set[tuple[str, str, str]] = set()
    for row in manifest:
        key = (row["crate"], row["test"], row["message"])
        if key in seen:
            print(f"skip-census: FAIL: {MANIFEST.name} declares {key} twice.")
            failures += 1
        seen.add(key)
    crates = sorted(set(_selected_crates(args.crate)) | {row["crate"] for row in manifest})
    total_found = 0
    for crate in crates:
        declared = {(r["test"], r["message"]): r for r in manifest if r["crate"] == crate}
        census = static_census(crate)
        if census is None:
            for test, message in sorted(declared):
                print(
                    f"skip-census: FAIL: {MANIFEST.name} declares {crate} {test} / {message!r}, "
                    f"which no longer exists in the source (no crate {crate})."
                )
                failures += 1
            continue
        found, unstructured = census
        total_found += len(found)
        for site in unstructured:
            print(
                f"skip-census: FAIL: {crate} {site['test']} ({site['where']}) prints "
                f"{site['message']!r}: {site['why']}."
            )
            print(
                "  One protocol for every skip: eprintln!(\"SKIP[<artifact>]: <reason>\"), one "
                "line per missing case, declared in tools/skip-census.tsv."
            )
            failures += 1
        actual = {(r["test"], r["message"]): r for r in found}
        for test, message in sorted(set(actual) - set(declared)):
            print(
                f"skip-census: FAIL: {crate} {test} ({actual[(test, message)]['where']}) prints "
                f"{message!r} and returns, and is NOT in {MANIFEST.name}."
            )
            print(
                "  A new artifact-gated test must be declared, or it is born invisible: the "
                "suite reports it green on every executor that lacks the artifact."
            )
            failures += 1
        for test, message in sorted(set(declared) - set(actual)):
            print(
                f"skip-census: FAIL: {MANIFEST.name} declares {crate} {test} / {message!r}, "
                "which no longer exists in the source."
            )
            print(
                "  A stale row inflates the budget: the executor is allowed a skip that can no "
                "longer happen, which silently permits a different one."
            )
            failures += 1
        for key in sorted(set(declared) & set(actual)):
            if declared[key]["where"] != actual[key]["where"]:
                print(
                    f"skip-census: FAIL: {crate} {key[0]} moved: {MANIFEST.name} says "
                    f"{declared[key]['where']}, source says {actual[key]['where']}. Update the row."
                )
                failures += 1
        if found or declared:
            print(f"skip-census: {crate}: {len(found)} protocol skip row(s), {len(unstructured)} unstructured")
    if manifest and total_found < STATIC_FLOOR:
        print(
            f"skip-census: FAIL: static census found {total_found} site(s), below the "
            f"non-vacuity floor of {STATIC_FLOOR}, while the manifest declares {len(manifest)}. "
            "Refusing to certify the manifest against an empty scan."
        )
        failures += 1
    if failures:
        print(f"skip-census: VERIFY FAIL ({failures} disagreement(s))")
        return 1
    print(f"skip-census: VERIFY OK ({total_found} declared skip row(s) across {len(crates)} crate(s))")
    return 0


def parse_run_output(text: str):
    """-> (protocol skips as (test, message), unstructured as (test, line), results, orphans)."""
    skips: list[tuple[str, str]] = []
    unstructured: list[tuple[str, str]] = []
    results: list[dict[str, int]] = []
    orphans = 0
    current = "<unattributed>"

    def classify(fragment: str) -> None:
        nonlocal orphans
        fragment = fragment.strip()
        if fragment.startswith("SKIP[") and PROTOCOL_RE.match(fragment):
            skips.append((current, fragment))
        elif RUN_SKIP_START_RE.match(fragment):
            unstructured.append((current, fragment))
        else:
            return
        if current == "<unattributed>":
            orphans += 1

    for line in text.splitlines():
        result = RESULT_RE.match(line)
        if result:
            results.append(
                {
                    "ok": 1 if result.group("verdict").rstrip(".") == "ok" else 0,
                    "passed": int(result.group("passed")),
                    "failed": int(result.group("failed")),
                    "filtered": int(result.group("filtered")),
                }
            )
            current = "<unattributed>"
            continue
        test = TEST_LINE_RE.match(line)
        if test:
            current = test.group(1)
            classify(test.group(2))
            continue
        classify(line)
    return skips, unstructured, results, orphans


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        print("skip-census: ERROR: nothing to run after --", file=sys.stderr)
        return 2
    budget_raw = os.environ.get(args.budget_var, "0")
    try:
        budget = int(budget_raw)
    except ValueError:
        print(f"skip-census: ERROR: {args.budget_var}={budget_raw!r} is not an integer", file=sys.stderr)
        return 2
    manifest = read_manifest()
    command = list(args.command) + ["--", "--test-threads=1", "--nocapture"]
    print(f"skip-census: running: {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if args.log:
        Path(args.log).write_text(proc.stdout, encoding="utf-8")
        print(f"skip-census: output banked at {args.log}")
    skips, unstructured, results, orphans = parse_run_output(proc.stdout)

    # The SUITE's own verdict first. A skip count from a suite that did not run, exited red, or
    # ran a name-filtered slice of itself is a green number about nothing.
    if proc.returncode != 0:
        print(f"skip-census: FAIL: the suite exited {proc.returncode}")
        for line in proc.stdout.splitlines():
            if line.startswith("test ") and "FAILED" in line:
                print(f"    {line}")
        # A fixed 20-line tail loses both the panic message AND the "failures:\n    tests::name"
        # block on any multi-crate/multi-binary run: --nocapture prints a failing test's own
        # "thread '...' panicked at ...: <message>" INLINE, while it runs, well before the
        # harness's "failures:" summary at the end of that binary's own run; a 20-line tail
        # showed neither (memra#543 PR #897 cost real time to a tail that showed only the
        # aggregate "N passed; M failed" line and an unrelated crate's doctest output, never the
        # failing test's own name or panic reason). Print from the FIRST "panicked at" line
        # (the actual cause), or else the LAST "failures:" section (the failing names cargo
        # lists), whichever is found and earliest, to the end. Falls back to a much longer tail
        # if neither is found in this captured output.
        lines = proc.stdout.splitlines()
        first_panic_at = next(
            (index for index, line in enumerate(lines) if "panicked at" in line), None
        )
        last_failures_at = None
        for index, line in enumerate(lines):
            if line.strip() == "failures:":
                last_failures_at = index
        candidates = [at for at in (first_panic_at, last_failures_at) if at is not None]
        if candidates:
            start = min(candidates)
            # Cap the slice: a panic early in a 300+ test combined run should not dump every
            # later crate's entire suite into the CI console. The failing test's own panic and
            # cargo's "failures:" list both land well inside this many lines in practice.
            print("\n".join(lines[start : start + 400]))
        else:
            print("\n".join(lines[-200:]))
        return 1
    if not results:
        print("skip-census: FAIL: no `test result:` line at all; the suite did not run.")
        print("\n".join(proc.stdout.splitlines()[-200:]))
        return 1
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_filtered = sum(r["filtered"] for r in results)
    if total_failed or any(not r["ok"] for r in results):
        print(f"skip-census: FAIL: {total_failed} failed test(s) across {len(results)} binaries")
        return 1
    if total_filtered:
        print(
            f"skip-census: FAIL: {total_filtered} test(s) FILTERED OUT of an unfiltered run. "
            "A name filter prints a green '0 passed; N filtered out' the day the name moves."
        )
        return 1
    if total_passed < args.min_passed:
        print(
            f"skip-census: FAIL: VACUOUS: {total_passed} passed, floor {args.min_passed}. "
            "A suite that ran (almost) nothing is not a green suite."
        )
        return 1

    for test, message in skips:
        print(f"  SKIP {test}: {message}")
    if orphans:
        print(
            f"skip-census: FAIL: {orphans} skip line(s) could not be attributed to a test. "
            "An unattributed skip cannot be checked against the manifest."
        )
        return 1
    if unstructured:
        for test, line in unstructured:
            print(
                f"skip-census: FAIL: {test} printed {line!r}, a skip that is not in the protocol "
                "form SKIP[<artifact>]: <reason>."
            )
        print("  An unstructured skip is counted by nothing. Convert it and declare it.")
        return 1

    templates = [(r["test"], template_regex(r["message"])) for r in manifest]
    undeclared = [
        (test, message)
        for test, message in skips
        if not any(test == d_test and pattern.fullmatch(message) for d_test, pattern in templates)
    ]
    if undeclared:
        for test, message in undeclared:
            print(
                f"skip-census: FAIL: {test} skipped with {message!r}, which is not a declared "
                f"row in {MANIFEST.name}."
            )
        print(
            "  An undeclared skip is a new blind spot. Add it to the manifest (and to the "
            "budget) deliberately, or make the test not need the artifact."
        )
        return 1

    if len(skips) > budget:
        print(f"skip-census: FAIL: {len(skips)} skip(s), budget {budget} ({args.budget_var}).")
        print(
            "  A skipped artifact-backed test (or case) is not a green one: the suite reports "
            f"'{total_passed} passed' whether or not the model-backed assertions ran. Stage the "
            f"artifacts, or set {args.budget_var}={len(skips)} to account for them deliberately; "
            "they will still be printed and still be counted."
        )
        return 1
    print(
        f"skip-census: {total_passed} passed, {len(skips)} skipped (budget {budget}), "
        f"0 failed, 0 filtered out across {len(results)} binaries"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Assert the shell-side census file (MEMRA_SKIP_CENSUS) against an expected count.

    The file must EXIST. An absent file is ambiguous: it means either "nothing skipped" or
    "MEMRA_SKIP_CENSUS was never exported to the child", and the second one is a blind census
    reading as a clean one. `init` creates it, so a missing file is a wiring failure.
    """
    path = Path(args.path)
    if args.init:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# memra skip census v1: ts\tkind\tsubject\treason\n", encoding="utf-8")
        print(f"skip-census: initialised {path}")
        return 0
    if not path.exists():
        print(
            f"skip-census: FAIL: {path} does not exist. The census was never initialised, so "
            "this run cannot tell 'nothing skipped' from 'the census was not wired'. Run "
            "`skip-census.py report --init <path>` before the gates."
        )
        return 1
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for row in rows:
        print(f"  SKIP {row}")
    if args.expect is not None and len(rows) != args.expect:
        print(f"skip-census: FAIL: {len(rows)} censused skip(s), expected exactly {args.expect}.")
        print(
            "  An EQUALITY, not a ceiling: this number is the count of things this executor is "
            "known to be blind to. Fewer means a gate stopped recording; more means a new blind "
            "spot. Both need a human, not a green run."
        )
        return 1
    print(f"skip-census: {len(rows)} censused skip(s) in {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    static = sub.add_parser("static", help="print the static census (manifest rows) for crates")
    static.add_argument("--crate", action="append", help="default: every crate under crates/")
    static.set_defaults(func=cmd_static)

    verify = sub.add_parser("verify", help="static census of every crate vs tools/skip-census.tsv")
    verify.add_argument("--crate", action="append", help="crates that must exist (all are verified)")
    verify.set_defaults(func=cmd_verify)

    run = sub.add_parser("run", help="run a cargo test invocation and gate its skips")
    run.add_argument("--budget-var", default="MEMRA_SKIP_BUDGET")
    run.add_argument("--min-passed", type=int, default=1)
    run.add_argument("--log", help="bank the full output here")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="assert a MEMRA_SKIP_CENSUS file")
    report.add_argument("path")
    report.add_argument("--init", action="store_true")
    report.add_argument("--expect", type=int)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.mode == "run" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return int(args.func(args))
    except CensusError as error:
        print(f"skip-census: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
