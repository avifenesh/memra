#!/usr/bin/env python3
r"""Table-shape census for docs/FLAGS.md: every body row carries exactly its header's cell count.

WHY THIS EXISTS (memra #22). GitHub-flavoured Markdown splits a table cell on every unescaped
`|`, INCLUDING one inside backticks. A row documenting a token such as `route=spec|plain` or a
default such as `q4|q8|mixed` therefore renders with extra columns, and the flag registry, the
one document a reader opens to learn what a door does and what its default is, shows the wrong
default in the wrong column for exactly the interesting rows (tokens, log shapes, mode sets).
A second shape of the same defect: a three-cell row pasted into a two-column table, which
renders its default as a third column with no header.

The check is escape-aware: `\|` is one character of cell content, a bare `|` is a cell
boundary, the same rule the renderer applies. Fenced code blocks are skipped. A file with zero
tables is a broken input, not a green one (exit 2), so gutting the registry cannot pass.

Usage: tools/flags-table-census.py [docs/FLAGS.md]
Exit: 0 every row matches its header; 1 violations listed on stderr; 2 unusable input.
Callers: tools/docs-registry-census.sh (ci.yml + pre-push), tools/test_docs_registry_census.sh
(teeth: proves both violation shapes red, escaped pipes and fenced pipes green).
"""
import re
import sys

DELIM = re.compile(r"^\s*\|[-: |]+\|?\s*$")


def cells(line):
    parts = re.split(r"(?<!\\)\|", line.strip())
    return parts[1:-1]


def census(path):
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
    except OSError as e:
        print(f"flags-table-census: cannot read {path}: {e}", file=sys.stderr)
        return 2
    in_code = False
    header = None
    header_line = 0
    tables = 0
    rows = 0
    violations = []
    for idx, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line.lstrip().startswith("|"):
            header = None
            continue
        if header is None:
            if idx < len(lines) and DELIM.match(lines[idx]):
                header = len(cells(line))
                header_line = idx
                tables += 1
            continue
        if DELIM.match(line):
            continue
        rows += 1
        n = len(cells(line))
        if n != header:
            violations.append(
                f"{path}:{idx}: {n} cells, but the table header at line {header_line} has {header}"
                " (an unescaped | inside a cell? write it as \\| ; a default cell pasted into a"
                " two-column table? fold it into the description)"
            )
    if tables == 0:
        print(f"flags-table-census: {path} carries no markdown table (parser broke, or the registry was gutted)", file=sys.stderr)
        return 2
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        print(f"flags-table-census: {len(violations)} malformed row(s) in {path}", file=sys.stderr)
        return 1
    print(f"flags-table-census: {path} tables={tables} rows={rows}, every row matches its header")
    return 0


if __name__ == "__main__":
    sys.exit(census(sys.argv[1] if len(sys.argv) > 1 else "docs/FLAGS.md"))
