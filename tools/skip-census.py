#!/usr/bin/env python3
"""Skip census — a Rust test that skips must be counted, named, and fatal by default.

WHY THIS FILE EXISTS (GATE-INTEGRITY-20260819 section 10, round 2's own "still green-but-blind"
list). `nv27b_twin_parity` reads:

    if !st_dir.join("model.safetensors.index.json").exists() || !twin.exists() {
        eprintln!("SKIP: ckpt/twin absent");
        return;
    }

and the test PASSES. That was defensible while nothing consumed it. Round 2 then wired
`cargo test -p memra-gguf ... --lib` into .github/workflows/ci.yml, and a hosted runner has no
checkpoints at all — so from that commit onward CI reports `90 passed` in perpetuity while the
model-backed parity never ran once. It is A-2's shape in `#[test]` form ("ALL GREEN (N cells,
M skipped)" matching `grep -q "ALL GREEN"`), and it is newly load-bearing because the `n_rot`
rotary-width geometry check lands in exactly that gate.

THE SHAPE, copied deliberately from round 2's kernel-check fix in tools/validate-h100.sh: the
harness parses the run's own output, counts the skips, prints them, and compares the count with
a NAMED budget that defaults to 0. Raising the budget is allowed and is the escape hatch for a
developer without artifacts — but it must name the number, and the number appears in the
verdict. An accounted skip, never an invisible one.

Three assertions, because a count alone can go blind in three different ways:

  run     the suite's own verdict is asserted first (cargo's exit status, every
          `test result:` line says ok., the run is not vacuous or name-filtered), then the
          SKIPs are counted against the budget. A census over a suite that did not run, or ran
          12 of 90 tests, would be a green number about nothing.
  verify  the STATIC census — every `#[test]` in the crate that prints SKIP and returns — is
          compared with tools/skip-census.tsv in BOTH directions. A new artifact-gated test
          cannot be born invisible: it has to be declared, or `verify` reds.
  report  the census file the shell gates append to (MEMRA_SKIP_CENSUS) asserted against an
          expected count, for batteries that mix Rust tests and generated gate scripts.

`run` uses `--test-threads=1 --nocapture`, and that is load-bearing rather than tidy: with
parallel threads libtest interleaves un-attributed output, so a SKIP line cannot be tied to the
test that emitted it and the report degrades to a bare number. Single-threaded, libtest writes
`test <path> ... ` with no newline before the test body runs, so the SKIP text lands on that
same line and attribution is exact. Measured cost on the rig for memra-gguf --lib: 34.7 s
serialized.
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

# `eprintln!("SKIP` ... the convention every skipping test in the tree already follows.
SKIP_PRINT_RE = re.compile(r'eprintln!\(\s*"(SKIP[^"]*)"')
FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*\(")
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*\{")
ATTR_OR_DOC_RE = re.compile(r"^\s*(?:#\[|#!\[|///|//!|//|$)")
TEST_LINE_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s*(.*)$")
RESULT_RE = re.compile(
    r"^test result:\s+(?P<verdict>\S+)\.?\s+(?P<passed>\d+) passed;\s+(?P<failed>\d+) failed;"
    r"\s+(?P<ignored>\d+) ignored;\s+(?P<measured>\d+) measured;\s+(?P<filtered>\d+) filtered"
)
# The tree carries 12 artifact-gated skipping tests in memra-gguf. A static census that finds
# nothing is a broken scan, and it would certify a blind suite as fully covered — the same
# non-vacuity refusal tools/check-flags.sh --list makes.
STATIC_FLOOR = 1


class CensusError(RuntimeError):
    """The census cannot be trusted, so it refuses rather than reporting a number."""


def literal_prefix(message: str) -> str:
    """The part of a SKIP message that survives format-argument expansion.

    The manifest records the SOURCE string (`SKIP: no model at {MINIMAX_DIR}`) because that is
    what `verify` compares against the source, and it must be an exact comparison there. At run
    time the same line reads `SKIP: no model at /data/...`, so the run-side match is on the
    literal prefix. Comparing the two forms directly would report every formatted message as
    undeclared, and the natural "fix" for that would be to loosen the match to a substring —
    which is how a check stops being able to fail.
    """
    return message.split("{", 1)[0]


def crate_src(crate: str) -> Path:
    path = ROOT / "crates" / crate / "src"
    if not path.is_dir():
        raise CensusError(f"no such crate source directory: {path}")
    return path


def static_census(crate: str) -> list[dict[str, str]]:
    """Every #[test] in the crate that prints SKIP and returns, with its message.

    A regex census over Rust is not a parser and says so: it walks backwards from each
    `eprintln!("SKIP...")` to the nearest enclosing `fn`, then checks that a `#[test]` attribute
    sits within the five lines above that fn. Both halves are asserted against the manifest by
    `verify`, so a miss on either side shows up as a disagreement rather than as silence.
    """
    rows: list[dict[str, str]] = []
    src = crate_src(crate)
    for path in sorted(src.rglob("*.rs")):
        # The file's own module path, so the census reports the EXACT string libtest prints
        # (`source::hy3_repack_probe::hy3_manifest_offset_roundtrip`) and the run-side match can
        # be an equality. A suffix match would work today and would quietly accept the wrong
        # test the day two modules share a function name.
        rel = path.relative_to(src)
        parts = list(rel.parts[:-1])
        if rel.stem not in ("lib", "main", "mod"):
            parts.append(rel.stem)
        file_mods = tuple(parts)
        lines = path.read_text(encoding="utf-8").splitlines()
        # Module path by brace depth, so the census reports the same name libtest prints.
        mod_at_line: list[tuple[str, ...]] = []
        stack: list[tuple[str, int]] = []
        depth = 0
        for line in lines:
            while stack and stack[-1][1] >= depth:
                stack.pop()
            mod_at_line.append(tuple(name for name, _ in stack))
            mod_match = MOD_RE.match(line)
            if mod_match:
                stack.append((mod_match.group(1), depth))
            depth += line.count("{") - line.count("}")
        for index, line in enumerate(lines):
            match = SKIP_PRINT_RE.search(line)
            if not match:
                continue
            message = match.group(1)
            fn_name = None
            fn_line = None
            for back in range(index, -1, -1):
                fn_match = FN_RE.match(lines[back])
                if fn_match:
                    fn_name = fn_match.group(1)
                    fn_line = back
                    break
            if fn_name is None or fn_line is None:
                raise CensusError(
                    f"{path}:{index + 1}: a SKIP print with no enclosing fn — the census cannot "
                    "attribute it, and an unattributable skip is exactly what this gate exists "
                    "to make impossible"
                )
            # Walk back over the fn's own attribute/doc block only. A fixed lookback window
            # would pick up the PREVIOUS function's #[test] whenever functions are short, which
            # is a false positive in the direction that hides a real one.
            has_test_attr = False
            back = fn_line - 1
            while back >= 0 and ATTR_OR_DOC_RE.match(lines[back]):
                if "#[test]" in lines[back]:
                    has_test_attr = True
                back -= 1
            if not has_test_attr:
                continue
            mods = mod_at_line[fn_line]
            rows.append(
                {
                    "crate": crate,
                    "test": "::".join((*file_mods, *mods, fn_name)),
                    "where": f"{path.relative_to(ROOT).as_posix()}:{index + 1}",
                    "message": message,
                }
            )
    return rows


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise CensusError(f"missing manifest {MANIFEST}")
    rows = []
    for lineno, raw in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise CensusError(
                f"{MANIFEST}:{lineno}: expected 4 tab-separated columns "
                f"(crate, test, where, message), got {len(cols)}"
            )
        rows.append(
            {
                "crate": cols[0],
                "test": cols[1],
                "where": cols[2],
                "message": cols[3],
            }
        )
    return rows


def cmd_static(args: argparse.Namespace) -> int:
    rows = static_census(args.crate)
    for row in rows:
        print(f"{row['crate']}\t{row['test']}\t{row['where']}\t{row['message']}")
    print(f"# {len(rows)} artifact-gated skipping test(s) in {args.crate}", file=sys.stderr)
    if len(rows) < STATIC_FLOOR:
        print(
            f"skip-census: ERROR: static census found {len(rows)} site(s), floor is "
            f"{STATIC_FLOOR} — the scan is broken, not the crate",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    failures = 0
    for crate in sorted({row["crate"] for row in manifest} | set(args.crate or [])):
        found = static_census(crate)
        declared = {(r["test"], r["message"]) for r in manifest if r["crate"] == crate}
        # The floor applies only where the manifest claims there is something to find. A crate
        # with no artifact-gated tests legitimately scans to zero; a crate the manifest says has
        # twelve and scans to zero means the SCAN broke, and the specific diagnosis is worth
        # more than the pile of "stale row" failures that would otherwise be printed.
        if declared and len(found) < STATIC_FLOOR:
            print(
                f"skip-census: FAIL — static census of {crate} found {len(found)} site(s), "
                f"below the non-vacuity floor of {STATIC_FLOOR}, while the manifest declares "
                f"{len(declared)}. Refusing to certify the manifest against an empty scan."
            )
            failures += 1
            continue
        actual = {(r["test"], r["message"]) for r in found}
        undeclared = sorted(actual - declared)
        stale = sorted(declared - actual)
        for test, message in undeclared:
            where = next(
                r["where"] for r in found if (r["test"], r["message"]) == (test, message)
            )
            print(
                f"skip-census: FAIL — {crate} {test} ({where}) prints {message!r} and returns, "
                f"and is NOT in {MANIFEST.name}."
            )
            print(
                "  A new artifact-gated test must be declared, or it is born invisible: the "
                "suite reports it green on every executor that lacks the artifact."
            )
            failures += 1
        for test, message in stale:
            print(
                f"skip-census: FAIL — {MANIFEST.name} declares {crate} {test} / {message!r}, "
                "which no longer exists in the source."
            )
            print(
                "  A stale row inflates the budget: the executor is allowed a skip that can no "
                "longer happen, which silently permits a different one."
            )
            failures += 1
        # `where` drift is reported but not fatal: a line number moves on every refactor and a
        # fatal line number would train people to stop running this.
        for row in found:
            declared_row = next(
                (
                    r
                    for r in manifest
                    if (r["crate"], r["test"], r["message"])
                    == (crate, row["test"], row["message"])
                ),
                None,
            )
            if declared_row and declared_row["where"] != row["where"]:
                print(
                    f"skip-census: NOTE — {crate} {row['test']} moved: manifest says "
                    f"{declared_row['where']}, source says {row['where']}"
                )
        if not undeclared and not stale:
            print(
                f"skip-census: {crate}: {len(found)} artifact-gated skipping test(s), all "
                "declared"
            )
    if failures:
        print(f"skip-census: VERIFY FAIL ({failures} disagreement(s))")
        return 1
    print("skip-census: VERIFY OK")
    return 0


def parse_run_output(text: str) -> tuple[list[tuple[str, str]], list[dict[str, int]], int]:
    """-> (skips as (test, message), per-binary result rows, count of bare SKIP lines)."""
    skips: list[tuple[str, str]] = []
    results: list[dict[str, int]] = []
    orphans = 0
    current = "<unattributed>"
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
            tail = test.group(2)
            if "SKIP" in tail:
                skips.append((current, tail[tail.index("SKIP") :].strip()))
            continue
        if "SKIP" in line and line.lstrip().startswith("SKIP"):
            skips.append((current, line.strip()))
            if current == "<unattributed>":
                orphans += 1
    return skips, results, orphans


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        print("skip-census: ERROR: nothing to run after --", file=sys.stderr)
        return 2
    budget_raw = os.environ.get(args.budget_var, "0")
    try:
        budget = int(budget_raw)
    except ValueError:
        print(
            f"skip-census: ERROR: {args.budget_var}={budget_raw!r} is not an integer",
            file=sys.stderr,
        )
        return 2
    command = list(args.command) + ["--", "--test-threads=1", "--nocapture"]
    print(f"skip-census: running: {' '.join(command)}")
    proc = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if args.log:
        Path(args.log).write_text(proc.stdout, encoding="utf-8")
        print(f"skip-census: output banked at {args.log}")
    skips, results, orphans = parse_run_output(proc.stdout)

    # The SUITE's own verdict first. A skip count from a suite that did not run, exited red, or
    # ran a name-filtered slice of itself is a green number about nothing (A-1/A-5's shape).
    if proc.returncode != 0:
        print(f"skip-census: FAIL — the suite exited {proc.returncode}")
        for line in proc.stdout.splitlines():
            if line.startswith("test ") and "FAILED" in line:
                print(f"    {line}")
        print("\n".join(proc.stdout.splitlines()[-20:]))
        return 1
    if not results:
        print("skip-census: FAIL — no `test result:` line at all; the suite did not run.")
        print("\n".join(proc.stdout.splitlines()[-20:]))
        return 1
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_filtered = sum(r["filtered"] for r in results)
    if total_failed or any(not r["ok"] for r in results):
        print(f"skip-census: FAIL — {total_failed} failed test(s) across {len(results)} binaries")
        return 1
    if total_filtered:
        print(
            f"skip-census: FAIL — {total_filtered} test(s) FILTERED OUT of an unfiltered run. "
            "A name filter prints a green '0 passed; N filtered out' the day the name moves."
        )
        return 1
    if total_passed < args.min_passed:
        print(
            f"skip-census: FAIL — VACUOUS: {total_passed} passed, floor {args.min_passed}. "
            "A suite that ran (almost) nothing is not a green suite."
        )
        return 1

    for test, message in skips:
        print(f"  SKIP {test}: {message}")
    if orphans:
        print(
            f"skip-census: FAIL — {orphans} SKIP line(s) could not be attributed to a test. "
            "Attribution is the point: an unattributed skip cannot be checked against the "
            "manifest, so it cannot be shown to be one of the known-blind ones."
        )
        return 1

    declared = {(r["test"], r["message"]) for r in read_manifest()}
    undeclared = [
        (test, message)
        for test, message in skips
        if not any(
            test == d_test and message.startswith(literal_prefix(d_msg))
            for d_test, d_msg in declared
        )
    ]
    if undeclared:
        for test, message in undeclared:
            print(
                f"skip-census: FAIL — {test} skipped with {message!r}, which is not a declared "
                f"row in {MANIFEST.name}."
            )
        print(
            "  An undeclared skip is a new blind spot. Add it to the manifest (and to the "
            "budget) deliberately, or make the test not need the artifact."
        )
        return 1

    if len(skips) > budget:
        print(
            f"skip-census: FAIL — {len(skips)} test(s) skipped, budget {budget} "
            f"({args.budget_var})."
        )
        print(
            "  A skipped artifact-backed test is not a green one: the suite reports "
            f"'{total_passed} passed' whether or not the model-backed assertions ran. Stage the "
            f"artifacts, or set {args.budget_var}={len(skips)} to account for them deliberately "
            "— they will still be printed and still be counted."
        )
        return 1
    print(
        f"skip-census: {total_passed} passed, {len(skips)} skipped (budget {budget}), "
        f"0 failed, 0 filtered out across {len(results)} binaries"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Assert the shell-side census file (MEMRA_SKIP_CENSUS) against an expected count.

    The file must EXIST. An absent file is ambiguous — it means either "nothing skipped" or
    "MEMRA_SKIP_CENSUS was never exported to the child", and the second one is a blind census
    reading as a clean one. `init` creates it, so a missing file is a wiring failure.
    """
    path = Path(args.path)
    if args.init:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# memra skip census v1 — ts\tkind\tsubject\treason\n", encoding="utf-8"
        )
        print(f"skip-census: initialised {path}")
        return 0
    if not path.exists():
        print(
            f"skip-census: FAIL — {path} does not exist. The census was never initialised, so "
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
        print(
            f"skip-census: FAIL — {len(rows)} censused skip(s), expected exactly {args.expect}."
        )
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

    static = sub.add_parser("static", help="print the static census for a crate")
    static.add_argument("--crate", default="memra-gguf")
    static.set_defaults(func=cmd_static)

    verify = sub.add_parser(
        "verify", help="static census vs tools/skip-census.tsv, both directions"
    )
    verify.add_argument("--crate", action="append")
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
