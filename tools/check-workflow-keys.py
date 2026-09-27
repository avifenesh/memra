#!/usr/bin/env python3
"""check-workflow-keys.py: refuse a GitHub workflow file with a duplicate mapping key.

WHY. On 2026-09-21 two PRs (#592, #590) each added a ci.yml job named `portable-suites`. Each
PR's own CI was green (its merge ref predated the other), main took both, and GitHub refused
the file: the push run had ZERO jobs and read "This run likely failed because of a workflow
file issue". Every push and PR after it was red at the workflow level, and no job in ci.yml
could catch it, because ci.yml itself was the file that did not parse. PyYAML's `safe_load`
keeps the LAST duplicate silently, so `python3 -c "import yaml; yaml.safe_load(...)"` passed on
that tree.

WHAT. Standard library only (revuto on #601: PyYAML is not guaranteed on every interpreter this
repo's hooks run under, and a gate that fails for a missing dependency gets uninstalled). A
line-based walker over the block-style YAML that workflow files are written in:

  * a mapping key is the text before the first `:` followed by a space or the end of line, at
    the line's indentation (after any `- ` sequence indicators); quoted keys are unquoted;
  * mapping scopes are tracked by indentation column: a key at a shallower column closes the
    deeper scopes, a `- ` item closes the previous item's scope even at equal indentation;
  * a block scalar (`|`, `>` with chomping or indentation indicators, optional comment) swallows
    every following line indented deeper than its key, so `run: |` bodies never yield keys;
  * full-line comments, document markers, directives and blank lines are skipped; flow
    sequences (`[main]`) are values and yield no keys; continuation lines of a multi-line plain
    scalar yield no key unless they contain `: `, in which case they open a scope of their own
    and can only report a duplicate against a twin continuation line (not seen in practice).

Scope, stated: flow mappings (`{a: 1}`), complex keys (`? `), merge keys (`<<`), anchors or
tags in key position and tab indentation are NOT modelled; the walker refuses them with exit 2
("cannot answer"), never green. GitHub's own parser rejects most of them too. Zero workflow
files is a refusal (a moved directory must not read as green); a file without a `jobs:`
mapping is a refusal. Exit 0 prints one line per file with its job names; exit 1 names the
file, the key, its line and column and the line of the first occurrence.

WHERE. tools/hooks/pre-push (no skip switch: a tree whose workflow GitHub cannot parse is
unshippable) and the ci.yml `gates` job, which protects the OTHER workflow files; ci.yml can
only be protected from itself before the push, or by a PR run on a merge ref that is current.
Teeth: tools/test_workflow_keys.sh.
"""

import pathlib
import re
import sys


class Refused(Exception):
    """Exit 1: a duplicate key, a missing jobs mapping, no workflow files."""


class Unsupported(Exception):
    """Exit 2: a construct the walker does not model; the census cannot answer."""


BLOCK_SCALAR = re.compile(r"^[|>][-+0-9]*\s*(#.*)?$")
KEY_LINE = re.compile(
    r"""^(?P<indent>[ ]*)(?P<dashes>(?:-[ ]+)*)
        (?P<key>"(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^\s"'#\[\]{},&*!|>%@`?][^#]*?)
        [ ]*:(?:[ ]+(?P<value>.*)|$)""",
    re.VERBOSE,
)


def unquote(key):
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        return key[1:-1]
    return key.strip()


def walk(text):
    """Yield nothing; raise Refused on a duplicate key. Return the list of job names."""
    scopes = []  # each: {"col": int, "keys": {name: line_no}, "parent": (col, key) or None}
    jobs = None
    block_col = None  # indentation column of the key whose block scalar is being skipped
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if block_col is not None:
            if not stripped:
                continue
            if len(line) - len(line.lstrip(" ")) > block_col:
                continue
            block_col = None
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith(("---", "...", "%")):
            continue
        if line[: len(line) - len(line.lstrip())].find("\t") >= 0:
            raise Unsupported(f"line {number}: tab indentation is not modelled")
        if stripped.startswith("? ") or stripped == "?":
            raise Unsupported(f"line {number}: complex key (`? `) is not modelled")
        match = KEY_LINE.match(line)
        if not match:
            body = re.sub(r"^[ ]*(?:-[ ]+)*", "", line)
            if body and body[0] in "&*!" and re.search(r":( |$)", body):
                # a PLAIN key starting with an indicator; a quoted "*" key is an ordinary string
                raise Unsupported(f"line {number}: anchor, alias or tag in key position is not modelled")
            continue  # a scalar sequence item, a plain-scalar continuation, a flow line
        indent = len(match.group("indent"))
        dashes = match.group("dashes")
        key = unquote(match.group("key"))
        key_col = indent + len(dashes)
        if key == "<<":
            raise Unsupported(f"line {number}: merge key (`<<`) is not modelled")
        value = (match.group("value") or "").strip()
        if value.startswith("{"):
            raise Unsupported(f"line {number}: flow mapping (`{{...}}`) is not modelled; use block style")
        if value.startswith("[") and "{" in value:
            raise Unsupported(f"line {number}: flow mapping inside a flow sequence is not modelled")
        if dashes:
            # a new sequence item closes the previous item's mapping even at equal indentation
            while scopes and scopes[-1]["col"] > indent:
                scopes.pop()
        while scopes and scopes[-1]["col"] > key_col:
            scopes.pop()
        if scopes and scopes[-1]["col"] == key_col and not dashes:
            scope = scopes[-1]
        elif scopes and scopes[-1]["col"] == key_col and dashes:
            # `- key:` at the column of an open mapping: a fresh item mapping replaces it
            scopes.pop()
            scope = None
        else:
            scope = None
        if scope is None:
            parent = None
            if scopes:
                top = scopes[-1]
                last = max(top["keys"], key=top["keys"].get) if top["keys"] else None
                parent = (top["col"], last)
            scope = {"col": key_col, "keys": {}, "parent": parent}
            scopes.append(scope)
            if parent == (0, "jobs") and jobs is None:
                jobs = scope["keys"]
        if key in scope["keys"]:
            raise Refused(
                f"duplicate mapping key {key!r} at line {number} column {key_col + 1} "
                f"(first at line {scope['keys'][key]})"
            )
        scope["keys"][key] = number
        if BLOCK_SCALAR.match(value):
            block_col = key_col
    if not scopes or scopes[0]["col"] != 0 or "jobs" not in scopes[0]["keys"]:
        raise Refused("no top-level `jobs:` mapping")
    if not jobs:
        raise Refused("`jobs:` has no jobs")
    return list(jobs)


def check(path):
    return walk(path.read_text(encoding="utf-8"))


def main(argv):
    root = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path(".github/workflows")
    files = sorted(p for p in root.glob("*.y*ml") if p.suffix in (".yml", ".yaml"))
    if not files:
        print(f"check-workflow-keys: FAIL: no workflow files under {root}", file=sys.stderr)
        return 1
    refused = 0
    unsupported = 0
    for path in files:
        try:
            jobs = check(path)
        except Refused as error:
            print(f"check-workflow-keys: FAIL: {path}: {error}", file=sys.stderr)
            refused += 1
            continue
        except (Unsupported, UnicodeDecodeError, OSError) as error:
            print(f"check-workflow-keys: CANNOT ANSWER: {path}: {error}", file=sys.stderr)
            unsupported += 1
            continue
        print(f"check-workflow-keys: {path}: jobs {', '.join(jobs)}")
    if refused:
        print(f"check-workflow-keys: FAIL: {refused} of {len(files)} workflow files refused", file=sys.stderr)
        return 1
    if unsupported:
        print(
            f"check-workflow-keys: CANNOT ANSWER: {unsupported} of {len(files)} workflow files use a "
            "construct this walker does not model (see above); fails closed",
            file=sys.stderr,
        )
        return 2
    print(f"check-workflow-keys: OK: {len(files)} workflow files, no duplicate mapping keys (stdlib walker)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
