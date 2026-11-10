#!/usr/bin/env python3
"""cut_prompts.py TOKENIZER_JSON OUTDIR TARGET...: raw-text prompts of exactly TARGET tokens for the
deep gate cells (the argmax-margin probe and run-spec encode the file raw, no chat template).

The text is the served cells' synthetic record document (pf_long_memra.py's `Record i:` lines,
no needles, no question), long enough for the largest target, cut on a character prefix by
binary search so the tokenizer's count equals the target. The gates print their own count; the
manifest keeps this one so the two can be compared.
Run: uv run --with tokenizers python3 cut_prompts.py ..."""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()


def doc(n_lines):
    return "\n".join(f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}."
                     for i in range(n_lines))


tok = Tokenizer.from_file(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
targets = sorted(int(t) for t in sys.argv[3:])


def ntok(s):
    return len(tok.encode(s, add_special_tokens=True).ids)


n = 1000
while ntok(doc(n)) < targets[-1] + 64:
    n *= 2
text = doc(n)
man = open(out / "prompts-manifest.jsonl", "a")
for target in targets:
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok(text[:mid]) <= target:
            lo = mid
        else:
            hi = mid - 1
    got = ntok(text[:lo])
    path = out / f"deep-{target}.txt"
    path.write_text(text[:lo])
    row = {"target": target, "chars": lo, "tokens": got, "file": path.name, "tokenizer": Path(sys.argv[1]).parent.name}
    man.write(json.dumps(row) + "\n")
    print(json.dumps(row))
