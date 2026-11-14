import hashlib, json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
text = open(sys.argv[2]).read()
ids = [0] + tok.encode(text, add_special_tokens=False).ids  # BOS 0, as the GGUF tokenizer adds (checked on board-2048)
print("source tokens", len(ids))
for n in (4100, 8200):
    p = ids[:n]
    assert len(p) == n
    s = " ".join(map(str, p))
    open(f"p{n}.ids", "w").write(s + "\n")
    print(n, "sha256", hashlib.sha256(s.encode()).hexdigest()[:16], "last", p[-3:])
