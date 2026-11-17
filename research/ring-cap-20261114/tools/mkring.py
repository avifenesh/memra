"""memra#1299 ring cell: a prefix of research/e2e/prompts/p6-64k.txt that the Step-3.7-Flash
tokenizer (the pinned tokenizer.json; the GGUF tokenizer prepends BOS 0) encodes as exactly
64 + 32,768 tokens, so concat-prime-probe's `tickinv --splits 64` primes 64 tokens and then a
32,768-token continuation. usage: mkring.py TOKENIZER_JSON P6_64K_TXT OUT_TXT"""
import hashlib
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2]).read()
WANT = 64 + 32768 - 1  # without BOS
enc = tok.encode(text, add_special_tokens=False)
print("source tokens (no BOS)", len(enc.ids))
assert len(enc.ids) > WANT + 8
best = None
for k in range(WANT - 4, WANT + 5):
    cut = enc.offsets[k][0]
    n = len(tok.encode(text[:cut], add_special_tokens=False).ids)
    if n == WANT:
        best = cut
        break
assert best is not None, "no cut gives the wanted count"
out = text[:best]
open(sys.argv[3], "w").write(out)
ids = tok.encode(out, add_special_tokens=False).ids
assert ids == enc.ids[: len(ids)], "the prefix does not tokenize as the source's prefix"
print("chars", len(out), "tokens with BOS", len(ids) + 1, "sha256", hashlib.sha256(out.encode()).hexdigest())
