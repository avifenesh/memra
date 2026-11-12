#!/usr/bin/env python3
"""Refuse a CUDA entry symbol that docs/KERNELS.md does not name.

AGENTS.md: docs/KERNELS.md maps every `.cu` entry symbol to what it computes, its types, arch
guard, selector, caller and gate. This census extracts the entry symbols of every TU in
crates/memra-engine/cu/*.cu, counting the cu/ headers each TU includes as part of it:

  - every `__global__` kernel: `extern "C"` fatbin kernels looked up by name, and the template
    and TU-local kernels a static-library host launcher starts;
  - every `extern "C"` host function, declared on its own line or inside an `extern "C" { }`
    block (`static` helpers in a block have no C linkage and are skipped);
  - the kernels and host functions a function-like macro stamps out, with `##` resolved per
    invocation.

A symbol is named when some inline code span in the doc is exactly the name, or the name
followed by a template marker (`name<...>`, `name<S>`). A family pattern (`foo_*`, `foo_{a,b}`,
`foo[_vl]`) names nothing, so a new kernel needs its own mention. No grandfather list.

Usage: tools/kernels-symbol-census.py [--cu-dir DIR] [--doc FILE] [--list]
  --list   print `symbol<TAB>kind<TAB>TU` for every entry symbol and exit 0
Exit 0: every symbol named. Exit 1: the unnamed symbols, one line each on stderr.
Exit 2: unusable input (missing or empty cu dir or doc, an unterminated `extern "C"` block,
a macro-stamped name that does not resolve, zero symbols or zero code spans).
Callers: tools/docs-registry-census.sh (ci.yml and pre-push), tools/test_docs_registry_census.sh.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDENT = re.compile(r"[A-Za-z_]\w*")
# Words that can sit between `__global__` / `extern "C"` and the function name.
ATTRS = {"__launch_bounds__", "__cluster_dims__", "__attribute__", "__declspec", "__align__",
         "alignas", "__maxnreg__"}
NOT_A_NAME = {"void", "int", "unsigned", "signed", "long", "short", "char", "float", "double",
              "bool", "const", "volatile", "static", "inline", "__forceinline__", "__host__",
              "__device__", "__global__", "__noinline__", "extern", "template", "typename",
              "struct", "class", "auto", "return", "if", "for", "while", "switch", "sizeof"}
SKIP_STATEMENT = {"typedef", "using", "struct", "class", "enum", "union", "static_assert",
                  "namespace", "template"}


class Unusable(Exception):
    pass


def strip_comments(text):
    """Blank comments, keep string and char literals and the line count."""
    def repl(m):
        s = m.group(0)
        if s[0] in "\"'":
            return s
        return "\n" * s.count("\n") if s.startswith("/*") else " "
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])+\'', repl, text,
                  flags=re.S)


def split_macros(text):
    """Return (text with #define lines blanked, {name: (params, body)} for function-like macros)."""
    lines = text.split("\n")
    out, macros, i = [], {}, 0
    while i < len(lines):
        line = lines[i]
        if not re.match(r"\s*#\s*define\b", line):
            out.append(line)
            i += 1
            continue
        chunk = [line]
        while chunk[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            chunk.append(lines[i])
        out.extend("" for _ in chunk)
        i += 1
        joined = " ".join(c.rstrip().rstrip("\\") for c in chunk)
        m = re.match(r"\s*#\s*define\s+(\w+)\(([^)]*)\)(.*)", joined)
        if m:
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            macros[m.group(1)] = (params, m.group(3))
    return "\n".join(out), macros


def balanced(text, i):
    """Index just past the group that opens at text[i] (one of ( [ { <)."""
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    opener, closer, depth = text[i], pairs[text[i]], 0
    for j in range(i, len(text)):
        c = text[j]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return j + 1
        elif c == ";" and opener == "<":
            break  # a comparison, not a template group
    return -1


def declared_name(text, i):
    """Name of the function declared from text[i]: the identifier before the first top-level `(`,
    skipping attribute groups. None for a variable, a function pointer or a cast."""
    last = None
    while i < len(text):
        c = text[i]
        if c in ";={}":
            return None
        m = IDENT.match(text, i)
        if m:
            word = m.group(0)
            i = m.end()
            if word in ATTRS:
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "(":
                    end = balanced(text, j)
                    if end < 0:
                        return None
                    i = end
                continue
            last = word
            continue
        if c == "(":
            return last if last and last not in NOT_A_NAME else None
        if c == "<":  # template parameters or explicit template arguments
            end = balanced(text, i)
            if end < 0:
                return None
            i = end
            continue
        i += 1
    return None


def extern_c_blocks(text):
    """[(start, end)] of each `extern "C" { ... }` body."""
    spans = []
    for m in re.finditer(r'extern\s+"C"\s*\{', text):
        end = balanced(text, m.end() - 1)
        if end < 0:
            raise Unusable(f"unterminated extern \"C\" block at offset {m.start()}")
        spans.append((m.end(), end - 1))
    return spans


def extract(text):
    """{name: kind} for one comment-stripped, macro-free source text."""
    found = {}
    # Directive lines (#if, #include, #pragma) are not declarations; both arms of a
    # conditional are scanned, so a stub arm and its real arm count once each.
    text = re.sub(r"^[ \t]*#.*$", "", text, flags=re.M)
    blocks = extern_c_blocks(text)
    for m in re.finditer(r"\b__global__\b", text):
        name = declared_name(text, m.end())
        if name:
            found[name] = "kernel"
    for m in re.finditer(r'extern\s+"C"\s+(?!\{)', text):
        if text.startswith("__global__", m.end()):
            continue
        name = declared_name(text, m.end())
        if name and name not in found:
            found[name] = "host"
    for start, end in blocks:
        body = text[start:end]
        depth, stmt_start = 0, 0
        for k, c in enumerate(body):
            if c == "{":
                if depth == 0:
                    record_block_statement(body[stmt_start:k], found)
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    stmt_start = k + 1
            elif c == ";" and depth == 0:
                record_block_statement(body[stmt_start:k], found)
                stmt_start = k + 1
    return found


def record_block_statement(stmt, found):
    words = IDENT.findall(stmt)
    if not words or words[0] in SKIP_STATEMENT or "static" in words[:3]:
        return
    if "__global__" in words:
        return  # counted by the kernel scan
    name = declared_name(stmt, 0)
    if name and name not in found:
        found[name] = "host"


def expand_macros(text, macros):
    """{name: kind} stamped out by function-like macros that define kernels or C functions."""
    found = {}
    for macro, (params, body) in macros.items():
        if "__global__" not in body and 'extern "C"' not in body:
            continue
        for m in re.finditer(r"(?<![\w#])" + re.escape(macro) + r"\s*\(", text):
            end = balanced(text, m.end() - 1)
            if end < 0:
                raise Unusable(f"unbalanced invocation of macro {macro}")
            args = [a.strip() for a in text[m.end():end - 1].split(",")]
            if len(args) != len(params):
                raise Unusable(f"macro {macro} invoked with {len(args)} args, defined with {len(params)}")
            sub = dict(zip(params, args))
            expanded = re.sub(r"\b\w+\b", lambda w: sub.get(w.group(0), w.group(0)), body)
            expanded = re.sub(r"\s*##\s*", "", expanded)
            if "#" in re.sub(r'"[^"]*"', "", expanded):
                raise Unusable(f"macro {macro} uses a preprocessor operator this census cannot expand")
            for name, kind in extract(expanded).items():
                found.setdefault(name, kind)
    for name in found:
        if name in {p for params, _ in macros.values() for p in params}:
            raise Unusable(f"macro-stamped symbol did not resolve: {name}")
    return found


def tu_symbols(cu_dir, path, seen):
    """{name: kind} for one file and the cu/ headers it includes (once per TU)."""
    raw = path.read_text(errors="replace")
    text, macros = split_macros(strip_comments(raw))
    found = extract(text)
    for name, kind in expand_macros(text, macros).items():
        found.setdefault(name, kind)
    for inc in re.findall(r'^\s*#\s*include\s+"([^"/]+\.cuh?)"', text, flags=re.M):
        header = cu_dir / inc
        if header.is_file() and header not in seen:
            seen.add(header)
            for name, kind in tu_symbols(cu_dir, header, seen).items():
                found.setdefault(name, kind)
    return found


def doc_spans(doc_text):
    """Every inline code span outside fenced blocks, stripped."""
    spans, fenced = set(), False
    for line in doc_text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            spans.update(s.strip() for s in re.findall(r"`([^`\n]+)`", line))
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cu-dir", default=str(ROOT / "crates/memra-engine/cu"))
    ap.add_argument("--doc", default=str(ROOT / "docs/KERNELS.md"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cu_dir, doc = pathlib.Path(a.cu_dir), pathlib.Path(a.doc)
    try:
        # Real TUs first, so a symbol a stub mirrors is attributed to the TU that implements it.
        tus = sorted(cu_dir.glob("*.cu"), key=lambda p: (p.name.endswith("_stub.cu"), p.name))
        if not tus:
            raise Unusable(f"no .cu files in {cu_dir}")
        symbols = {}
        for tu in tus:
            for name, kind in sorted(tu_symbols(cu_dir, tu, set()).items()):
                symbols.setdefault(name, (kind, tu.name))
        if not symbols:
            raise Unusable(f"extracted zero entry symbols from {cu_dir} (parser broke?)")
        if a.list:
            for name, (kind, tu) in sorted(symbols.items()):
                print(f"{name}\t{kind}\t{tu}")
            return 0
        try:
            spans = doc_spans(doc.read_text())
        except OSError as e:
            raise Unusable(f"cannot read {doc}: {e}")
        if not spans:
            raise Unusable(f"{doc} has zero code spans (parser broke, or the doc was gutted)")
    except Unusable as e:
        print(f"kernels-symbol-census: {e}", file=sys.stderr)
        return 2
    named = {re.sub(r"<[^`]*>$", "", s) for s in spans}
    missing = sorted(n for n in symbols if n not in named)
    for n in missing:
        kind, tu = symbols[n]
        print(f"{doc.name}: {kind} entry symbol not named: {n} (cu/{tu})", file=sys.stderr)
    kernels = sum(1 for k, _ in symbols.values() if k == "kernel")
    if missing:
        print(f"kernels-symbol-census: {len(missing)} of {len(symbols)} entry symbols are not "
              f"named in {doc.name}; give each its exact name in a row's Symbol cell",
              file=sys.stderr)
        return 1
    print(f"kernels-symbol-census: {len(symbols)} entry symbols in {len(tus)} TUs "
          f"({kernels} kernels, {len(symbols) - kernels} host entries), all named in {doc.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
