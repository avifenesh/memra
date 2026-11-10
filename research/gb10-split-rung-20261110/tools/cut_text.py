#!/usr/bin/env python3
"""cut_text.py TOKENIZER_JSON SOURCE.txt TARGET OUT.txt: the longest character prefix of SOURCE whose
token count is TARGET (binary search; the gates encode the file raw). Prints the count.
Run: uv run --with tokenizers python3 cut_text.py ..."""
import sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2], encoding="utf-8").read()
target = int(sys.argv[3])
ntok = lambda s: len(tok.encode(s, add_special_tokens=True).ids)
lo, hi = 1, len(text)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if ntok(text[:mid]) <= target:
        lo = mid
    else:
        hi = mid - 1
open(sys.argv[4], "w", encoding="utf-8").write(text[:lo])
print({"source": sys.argv[2].rsplit("/", 1)[-1], "chars": lo, "tokens": ntok(text[:lo]), "out": sys.argv[4].rsplit("/", 1)[-1]})
