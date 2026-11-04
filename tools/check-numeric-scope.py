#!/usr/bin/env python3
"""Refuse a model entry point that does not install the model's numeric profile first.

Each loaded HybridModel owns a NumericProfile (crates/memra-engine/src/numeric_profile.rs). The
kernel launchers read the profile installed on the calling thread, and outside every model scope
they read NumericProfile::BASE. So an entry point that forgets to install its model's profile
does not fail: it runs another model's program.

This census checks the entry shapes the design relies on. In each, the first statement of the
body must be the install:

  * every non-private method with a self receiver and an Engine parameter in an
    `impl HybridModel` block: `let _numeric = self.enter_numeric();`
  * `advance_chunk` and `finish` of every `impl PrimeWalker for ...` block:
    `let _numeric = self.<model>.enter_numeric();`
  * every thread spawned inside those blocks, or inside any fn that takes a `&HybridModel`:
    `let _numeric = <profile>.enter();` or `let _numeric = <model>.enter_numeric();` as the
    closure's first statement.

A spawned closure that launches nothing (host-side work only) is exempt when a comment carrying
`numeric-scope: host-only` sits on the spawn line, in the three lines above it or on the closure's
first line. Code under `#[cfg(test)]` modules is skipped. Exit 1 lists every site that breaks a
rule; exit 2 means the census found no entry point at all, which is a broken census, not a pass.

Usage: tools/check-numeric-scope.py [src-dir]   (default crates/memra-engine/src)
"""
import pathlib
import re
import sys

EXEMPT = "numeric-scope: host-only"
RAW_STR = re.compile(r'b?r(#*)"')
CHAR = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")


def mask(s):
    """Blank comments, strings and char literals (newlines kept), so brackets match."""
    out = list(s)
    i, n = 0, len(s)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = s[i]
        if s.startswith("//", i):
            j = s.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif s.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if s.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif s.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c in "br" and (m := RAW_STR.match(s, i)) and not (i and (s[i - 1].isalnum() or s[i - 1] == "_")):
            end = s.find('"' + m.group(1), m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            blank(m.end(), j - 1 - len(m.group(1)))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
        elif c == "'":
            # A char literal ('x', '\n', '\u{..}', '\''); anything else is a lifetime or a label.
            m = CHAR.match(s, i)
            if m:
                blank(i + 1, m.end() - 1)
                i = m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def close(t, i, o="{", c="}"):
    """Index just past the bracket that matches the opener at t[i]."""
    depth = 0
    for j in range(i, len(t)):
        if t[j] == o:
            depth += 1
        elif t[j] == c:
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError(f"unbalanced {o}{c} at offset {i}")


def test_ranges(t):
    """Offset ranges of `#[cfg(test)]` modules."""
    head = re.compile(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
    return [(m.start(), close(t, m.end() - 1)) for m in head.finditer(t)]


def blocks(t, header):
    """(header offset, body start, body end) for every block whose header matches."""
    for m in re.finditer(header, t):
        yield m.start(), m.end(), close(t, m.end() - 1)


FN = re.compile(
    r"(?P<vis>\bpub(?:\s*\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern\s+\"[^\"]*\")\s+)*\bfn\s+(?P<name>\w+)"
)


def fns(t, a, b):
    """(name, vis, params, body start, body end) for each fn with a body in t[a:b], outermost
    first; a fn nested inside another fn's body is skipped with it."""
    i = a
    while True:
        m = FN.search(t, i, b)
        if not m:
            return
        j = m.end()
        if t[j] == "<":
            depth = 0
            while True:
                if t[j] == "<":
                    depth += 1
                elif t[j] == ">" and t[j - 1] != "-":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
        p0 = t.index("(", j)
        p1 = close(t, p0, "(", ")")
        k, depth = p1, 0
        while depth or t[k] not in "{;":
            if t[k] in "([":
                depth += 1
            elif t[k] in ")]":
                depth -= 1
            k += 1
        if t[k] == ";":
            i = k + 1
            continue
        end = close(t, k)
        yield m.group("name"), m.group("vis"), t[p0 + 1 : p1 - 1], k + 1, end
        i = end


ENTER_SELF = re.compile(r"let\s+_\w*\s*=\s*self\.enter_numeric\(\)\s*;")
ENTER_FIELD = re.compile(r"let\s+_\w*\s*=\s*self(?:\.\w+)+\.enter_numeric\(\)\s*;")
ENTER_ANY = re.compile(r"let\s+_\w*\s*=\s*[\w.]+\.(?:enter_numeric|enter)\(\)\s*;")
ENGINE = re.compile(r"&\s*(?:'\w+\s+)?(?:mut\s+)?(?:(?:crate|super|memra_engine)::)*Engine\b")
MODEL = re.compile(r"&\s*(?:'\w+\s+)?(?:mut\s+)?(?:(?:crate|super|memra_engine)::)*(?:hybrid::)?HybridModel\b")
SELF = re.compile(r"^\s*(?:&\s*(?:'\w+\s+)?(?:mut\s+)?)?(?:mut\s+)?self\b")
SPAWN = re.compile(r"(?:\bthread::spawn|\.spawn|\.spawn_scoped)\s*\(")
HYBRID_IMPL = r"\bimpl\s+(?:(?:crate::)?hybrid::|super::)?HybridModel\s*\{"
WALKER_IMPL = r"\bimpl\s*(?:<[^>{]*>)?\s*(?:crate::prime_walker::|super::)?PrimeWalker\s+for\s+[^{]+\{"


def line_of(s, at):
    return s.count("\n", 0, at) + 1


def spawn_ok(s, t, at):
    """Whether the spawn whose name starts at t[at] installs a scope first or is exempt."""
    m = SPAWN.match(t, at)
    p1 = close(t, m.end() - 1, "(", ")")
    body = None
    bar = t.find("|", m.end(), p1)
    if bar >= 0:
        k = t.find("|", bar + 1, p1) + 1
        while k < p1 and t[k] != "{":
            k += 1
        body = k + 1 if k < p1 else None
    lo = at
    for _ in range(4):
        lo = max(s.rfind("\n", 0, lo), 0)
    hi = s.find("\n", body if body is not None else m.end())
    hi = len(s) if hi < 0 else s.find("\n", hi + 1)
    if EXEMPT in s[lo : hi if hi >= 0 else len(s)]:
        return True
    return body is not None and ENTER_ANY.match(t[body:].lstrip()) is not None


def check_file(path, rel=None):
    """(violations, {kind: sites checked}) for one Rust source file."""
    rel = rel or str(path)
    s = path.read_text()
    t = mask(s)
    tests = test_ranges(t)

    def live(at):
        return not any(x <= at < y for x, y in tests)

    bad = []
    counts = {"method": 0, "walker": 0, "spawn": 0}
    spawn_scopes = []
    for start, a, b in blocks(t, HYBRID_IMPL):
        if not live(start):
            continue
        spawn_scopes.append((a, b))
        for name, vis, params, at, _end in fns(t, a, b - 1):
            if vis and SELF.match(params) and ENGINE.search(params):
                counts["method"] += 1
                if not ENTER_SELF.match(t[at:].lstrip()):
                    bad.append(f"{rel}:{line_of(s, at)}: HybridModel::{name} takes an Engine and does not open with `let _numeric = self.enter_numeric();`")
    for start, a, b in blocks(t, WALKER_IMPL):
        if not live(start):
            continue
        spawn_scopes.append((a, b))
        for name, _vis, _params, at, _end in fns(t, a, b - 1):
            if name in ("advance_chunk", "finish"):
                counts["walker"] += 1
                if not ENTER_FIELD.match(t[at:].lstrip()):
                    bad.append(f"{rel}:{line_of(s, at)}: PrimeWalker::{name} does not open with `let _numeric = self.<model>.enter_numeric();`")
    for _name, _vis, params, at, end in fns(t, 0, len(t)):
        if live(at) and MODEL.search(params):
            spawn_scopes.append((at, end))
    spawns = sorted({m.start() for a, b in spawn_scopes for m in SPAWN.finditer(t, a, b) if live(m.start())})
    for at in spawns:
        counts["spawn"] += 1
        if not spawn_ok(s, t, at):
            bad.append(f"{rel}:{line_of(s, at)}: a thread spawned for a model does not enter its profile first (or say `{EXEMPT}`)")
    return bad, counts


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else "crates/memra-engine/src")
    files = sorted(root.rglob("*.rs"))
    if not files:
        print(f"::error::check-numeric-scope: no Rust sources under {root}")
        return 2
    bad, counts = [], {"method": 0, "walker": 0, "spawn": 0}
    for f in files:
        b, c = check_file(f, f.as_posix())
        bad += b
        for k, v in c.items():
            counts[k] += v
    if counts["method"] == 0 or counts["walker"] == 0:
        print(f"::error::check-numeric-scope: found {counts} under {root}; a census that sees no "
              "HybridModel method or prime walker is broken")
        return 2
    total = sum(counts.values())
    for line in bad:
        print(f"::error::check-numeric-scope: {line}")
    shape = f"{counts['method']} HybridModel methods, {counts['walker']} walker steps, {counts['spawn']} model threads"
    if bad:
        print(f"check-numeric-scope: {len(bad)} of {total} entry sites do not install the model's numeric profile ({shape})")
        return 1
    print(f"check-numeric-scope: OK: {total} entry sites install the model's numeric profile first ({shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
