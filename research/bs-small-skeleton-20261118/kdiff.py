#!/usr/bin/env python3
"""Print a unified diff of one kernel's SASS (instructions only, addresses and encodings dropped) between two dumps.
usage: kdiff.py BASE.sass NEW.sass OLD_KEY [context]
"""
import difflib
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sass_cmp  # noqa: E402


def instrs(body):
    out = []
    for l in body:
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;?\s*/\* 0x[0-9a-f]+ \*/", l)
        if m:
            out.append(m.group(1))
    return out


def main():
    fa, fb = sass_cmp.split_sass(sys.argv[1]), sass_cmp.split_sass(sys.argv[2])
    ka, kb = sass_cmp.keyed(fa), sass_cmp.keyed(fb)
    k = sys.argv[3]
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    a, b = instrs(fa[ka[k]]), instrs(fb[kb[k]])
    print(f"{len(a)} vs {len(b)} instructions")
    sa = [re.sub(r"\bU?R\d+\b|\bP\d\b|\bUP\d\b", "R", x) for x in a]
    sb = [re.sub(r"\bU?R\d+\b|\bP\d\b|\bUP\d\b", "R", x) for x in b]
    print("equal modulo register names:", sa == sb)
    for line in difflib.unified_diff(a, b, lineterm="", n=ctx):
        print(line)


if __name__ == "__main__":
    main()
