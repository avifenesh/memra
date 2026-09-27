#!/usr/bin/env python3
r"""Row-shape census for docs/FLAGS.md: a flag row states one current contract (memra#127).

WHY THIS EXISTS. A FLAGS.md row is edited one clause at a time by different lanes over months.
Past a few thousand characters a row stops holding one coherent state: an edit updates one
clause and leaves a neighbour asserting the opposite. On memra#119 review found a fresh
self-contradiction in MEMRA_B200_DSA_SELECT's row (then ~16k characters) in three consecutive
rounds: a serving receipt sitting above "speed has no B200 receipt yet", ladder figures citing
the wrong gate log, and one change described as both "still open" and "fixed here". No check
noticed. A stale-fragment grep before each push caught what its author predicted and missed
the rest, and it gets worse exactly as the row grows.

So the row holds only the CURRENT CONTRACT (what the flag does, its default, both arms, the
rollback seam, when it engages, a receipt pointer) and receipt history lives in
docs/FLAGS-HISTORY.md, one section per flag, which the row links to. This census enforces the
parts of that shape a machine can check without guessing at meaning:

  R1 length         every table row is at most MAX_ROW_CHARS characters. It cannot see a
                    contradiction; it keeps the row short enough to re-read whole on every edit,
                    which is what prevents one.
  R2 default cell   in a table with a `default` column, that cell is at most
                    MAX_DEFAULT_CHARS characters: the default is a statement, not a narrative.
  R3 decide-by      a row names at most one distinct `decide-by` date. Two dates in one row is
                    a contradiction by construction (one of them is stale).
  R4 history links  every `FLAGS-HISTORY.md#<anchor>` link in the registry resolves to a section
                    heading in docs/FLAGS-HISTORY.md, and every section there is linked from the
                    registry. A dangling pointer is a row pointing at evidence that is not there;
                    an unlinked section is history no row admits to.

No grandfather list, same reason tools/check-flags.sh has none: an exceptions file absorbs the
regression it was never granted for. A row over the cap is split (history out, pointer in) in
the same commit that grew it.

Usage: tools/flags-row-shape.py [docs/FLAGS.md [docs/FLAGS-HISTORY.md]]
Exit: 0 every row conforms; 1 violations listed on stderr; 2 unusable input.
Callers: tools/check-flags.sh (so the pre-push flags arm and ci.yml's live census both run it).
Teeth: tools/test_check_flags.sh (one red arm per rule) and tools/test_flags_guard.sh (a real
push refused on an over-long row).
"""
import os
import re
import sys

MAX_ROW_CHARS = 4000
MAX_DEFAULT_CHARS = 300

DELIM = re.compile(r"^\s*\|[-: |]+\|?\s*$")
DECIDE_BY = re.compile(r"decide[- ]by\W{0,6}(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
HISTORY_LINK = re.compile(r"FLAGS-HISTORY\.md#([A-Za-z0-9_\-]+)")
HEADING = re.compile(r"^##\s+(.+?)\s*$")


def cells(line):
    parts = re.split(r"(?<!\\)\|", line.strip())
    return parts[1:-1]


def slug(heading):
    # GitHub's heading anchor: lowercase, drop everything but word chars, spaces and hyphens,
    # spaces to hyphens. Flag headings are bare names, so this is the identity lowercased.
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().split("\n")


def census(flag_doc, history_doc):
    try:
        lines = read_lines(flag_doc)
    except OSError as e:
        print(f"flags-row-shape: cannot read {flag_doc}: {e}", file=sys.stderr)
        return 2
    violations = []
    links = {}
    rows = 0
    in_code = False
    header = None
    for idx, line in enumerate(lines, 1):
        for anchor in HISTORY_LINK.findall(line):
            links.setdefault(anchor, idx)
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line.lstrip().startswith("|"):
            header = None
            continue
        if DELIM.match(line):
            continue
        if header is None and idx < len(lines) and DELIM.match(lines[idx]):
            header = [c.strip().lower() for c in cells(line)]
            continue
        rows += 1
        row_cells = cells(line)
        name = row_cells[0].strip() if row_cells else "?"
        if len(line) > MAX_ROW_CHARS:
            violations.append(
                f"{flag_doc}:{idx}: {name}: row is {len(line)} chars, cap {MAX_ROW_CHARS}."
                " Keep the current contract (what it does, default, both arms, rollback seam,"
                " when it engages, receipt pointer) and move the history to a"
                f" `## <flag>` section of {history_doc} that the row links to"
            )
        if header and "default" in header and len(row_cells) == len(header):
            d = row_cells[header.index("default")].strip()
            if len(d) > MAX_DEFAULT_CHARS:
                violations.append(
                    f"{flag_doc}:{idx}: {name}: default cell is {len(d)} chars, cap"
                    f" {MAX_DEFAULT_CHARS}. State the default; the why goes in the description"
                    " cell and the history in the flag's history section"
                )
        dates = sorted(set(DECIDE_BY.findall(line)))
        if len(dates) > 1:
            violations.append(
                f"{flag_doc}:{idx}: {name}: row names {len(dates)} decide-by dates"
                f" ({', '.join(dates)}). One is stale; keep the current one"
            )
    if rows == 0:
        print(f"flags-row-shape: {flag_doc} carries no table rows (parser broke, or the registry was gutted)", file=sys.stderr)
        return 2

    sections = {}
    if os.path.exists(history_doc):
        try:
            hist = read_lines(history_doc)
        except OSError as e:
            print(f"flags-row-shape: cannot read {history_doc}: {e}", file=sys.stderr)
            return 2
        in_code = False
        for idx, line in enumerate(hist, 1):
            if line.startswith("```"):
                in_code = not in_code
                continue
            m = None if in_code else HEADING.match(line)
            if m:
                a = slug(m.group(1))
                if a in sections:
                    violations.append(
                        f"{history_doc}:{idx}: duplicate section `{m.group(1)}` (first at line"
                        f" {sections[a]}); append to the existing section instead"
                    )
                else:
                    sections[a] = idx
    for anchor, idx in sorted(links.items(), key=lambda kv: kv[1]):
        if anchor not in sections:
            violations.append(
                f"{flag_doc}:{idx}: links {os.path.basename(history_doc)}#{anchor} but"
                f" {history_doc} has no `## ` section with that anchor"
            )
    for anchor, idx in sorted(sections.items(), key=lambda kv: kv[1]):
        if anchor not in links:
            violations.append(
                f"{history_doc}:{idx}: section #{anchor} is linked from no row of {flag_doc};"
                " point the flag's row (or its removed-door ledger row) at it"
            )

    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        print(f"flags-row-shape: {len(violations)} row-shape violation(s)", file=sys.stderr)
        return 1
    print(
        f"flags-row-shape: {rows} rows within {MAX_ROW_CHARS} chars, default cells within"
        f" {MAX_DEFAULT_CHARS}, one decide-by each, {len(sections)} history sections all linked"
    )
    return 0


if __name__ == "__main__":
    flag_doc = sys.argv[1] if len(sys.argv) > 1 else "docs/FLAGS.md"
    default_hist = os.path.join(os.path.dirname(flag_doc) or ".", "FLAGS-HISTORY.md")
    history_doc = sys.argv[2] if len(sys.argv) > 2 else default_hist
    sys.exit(census(flag_doc, history_doc))
