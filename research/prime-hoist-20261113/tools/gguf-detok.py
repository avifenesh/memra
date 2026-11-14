#!/usr/bin/env python3
"""memra#1258: write prompt texts whose byte-level BPE tokenization is a given prefix of a token
list. Reads the vocabulary (tokenizer.ggml.tokens) from a GGUF's metadata, detokenizes
ids[:n] for each requested n (GPT-2 byte-to-unicode mapping), and writes one .txt per n.

usage: gguf-detok.py GGUF IDS_FILE OUT_DIR N...
IDS_FILE holds a run-gen log; its 'prompt tokens: [...]' line is the token list."""
import json
import re
import struct
import sys
from pathlib import Path


def read_tokens(path: str) -> list[str]:
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack("<" + fmt, f.read(n))

        def rstr():
            (n,) = rd("Q")
            return f.read(n).decode("utf-8", "surrogateescape")

        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = rd("I")
                (cnt,) = rd("Q")
                return [rval(et) for _ in range(cnt)]
            return rd(sizes[t])[0]

        assert f.read(4) == b"GGUF"
        (ver,) = rd("I")
        (_nt,) = rd("Q")
        (nkv,) = rd("Q")
        for _ in range(nkv):
            key = rstr()
            (t,) = rd("I")
            if key == "tokenizer.ggml.tokens":
                return rval(t)
            rval(t)
    raise SystemExit("no tokenizer.ggml.tokens")


def byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    gguf, ids_file, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    ns = [int(x) for x in sys.argv[4:]]
    m = re.search(r"prompt tokens: (\[[0-9, ]*\])", Path(ids_file).read_text(errors="replace"))
    ids = json.loads(m.group(1))
    vocab = read_tokens(gguf)
    dec = byte_decoder()
    out.mkdir(parents=True, exist_ok=True)
    for n in ns:
        assert n <= len(ids), (n, len(ids))
        raw = bytearray()
        for i in ids[:n]:
            raw.extend(dec[c] for c in vocab[i])
        (out / f"p{n}.txt").write_bytes(bytes(raw))
        (out / f"p{n}.ids").write_text(" ".join(str(i) for i in ids[:n]) + "\n")
        print(f"p{n}: {len(raw)} bytes, last ids {ids[n-3:n]}")


if __name__ == "__main__":
    main()
