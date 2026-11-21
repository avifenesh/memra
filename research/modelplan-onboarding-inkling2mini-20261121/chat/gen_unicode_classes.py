#!/usr/bin/env python3
"""Generate crates/memra-tokenizer/src/unicode_gpt4o_data.rs.

The gpt-4o (o200k) Split regex names Unicode general categories the llama.cpp-derived table in
unicode_data.rs does not carry (Lu, Lt, Lm, Lo, Ll). The oracle for token ids is the
tokenizers library, whose Split step runs on Oniguruma, so the classes are read from that
engine itself: every scalar value is probed with a one-class Split regex. The result is also
compared with this interpreter's unicodedata and the script fails if they differ, so the
table's Unicode version is the one printed in the header.

    uv run --with tokenizers==0.23.3 python3 -I gen_unicode_classes.py OUT.rs
"""

import sys
import unicodedata

import tokenizers
from tokenizers import Regex, pre_tokenizers

SEP = "\x00"
SCALARS = [c for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF]


def probe(pattern):
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    hit = set()
    for i in range(0, len(SCALARS), 2048):
        chunk = SCALARS[i : i + 2048]
        text = SEP.join(chr(c) for c in chunk)
        for _, (a, b) in split.pre_tokenize_str(text):
            if b - a == 1 and a % 2 == 0:
                hit.add(chunk[a // 2])
    return hit


def main():
    out_path = sys.argv[1]
    cls = {name: probe("\\p{%s}" % name) for name in ("Lu", "Lt", "Lm", "Lo", "Ll", "L", "M", "N")}
    cls["s"] = probe("\\s")
    assert cls["L"] == cls["Lu"] | cls["Lt"] | cls["Lm"] | cls["Lo"] | cls["Ll"]
    for name in ("Lu", "Lt", "Lm", "Lo", "Ll", "M", "N"):
        py = {c for c in SCALARS if unicodedata.category(chr(c)).startswith(name)}
        assert cls[name] == py, f"{name}: Oniguruma and unicodedata {unicodedata.unidata_version} differ"
    # One class per scalar: the categories are disjoint, and \s holds no letter, mark or number.
    code = {}
    for value, names in ((1, ("Lu", "Lt")), (2, ("Ll",)), (3, ("Lm", "Lo")), (4, ("M",)), (5, ("N",)), (6, ("s",))):
        for name in names:
            for c in cls[name]:
                assert c not in code, hex(c)
                code[c] = value
    assert code[0x0D] == 6 and code[0x0A] == 6 and 0 not in cls["s"]
    ranges = []
    for c in sorted(code):
        if ranges and ranges[-1][1] == c - 1 and ranges[-1][2] == code[c]:
            ranges[-1][1] = c
        else:
            ranges.append([c, c, code[c]])
    lines = [
        "//! Unicode classes of the gpt-4o (o200k) pre-tokenizer split. GENERATED, do not edit:",
        "//! research/modelplan-onboarding-inkling2mini-20261121/chat/gen_unicode_classes.py",
        f"//! read them from Oniguruma through tokenizers {tokenizers.__version__} (Unicode {unicodedata.unidata_version}; equal to",
        "//! that version's general categories, checked by the script).",
        "//!",
        "//! (first, last, class), sorted and disjoint; a scalar in no range is class 0 (other).",
        "//! 1 = Lu|Lt, 2 = Ll, 3 = Lm|Lo, 4 = M, 5 = N, 6 = regex whitespace.",
        "",
        "#[rustfmt::skip]",
        f"pub static GPT4O_CLASS_RANGES: [(u32, u32, u8); {len(ranges)}] = [",
    ]
    row = []
    for a, b, v in ranges:
        row.append(f"(0x{a:X}, 0x{b:X}, {v}),")
        if len(row) == 6:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("];")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(ranges)} ranges, whitespace {sorted(cls['s'])}")


if __name__ == "__main__":
    main()
