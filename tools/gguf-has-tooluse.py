#!/usr/bin/env python3
"""Exit 0 iff the GGUF's tokenizer.chat_template carries the gemma4 tooluse markers
(<|turn> AND <|tool>), i.e. the gemma4 tools renderer arm will engage. Header-only read,
no tensor data. Usage: gguf-has-tooluse.py MODEL.gguf"""
import struct
import sys


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def read_val(f, t):
    if t == 0:
        return struct.unpack("<B", f.read(1))[0]
    if t == 1:
        return struct.unpack("<b", f.read(1))[0]
    if t == 2:
        return struct.unpack("<H", f.read(2))[0]
    if t == 3:
        return struct.unpack("<h", f.read(2))[0]
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return bool(struct.unpack("<B", f.read(1))[0])
    if t == 8:
        return read_str(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_val(f, et) for _ in range(n)]
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"bad gguf value type {t}")


def chat_template(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        _ver, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            v = read_val(f, t)
            if k == "tokenizer.chat_template":
                return v
    return None


tmpl = chat_template(sys.argv[1]) or ""
sys.exit(0 if ("<|turn>" in tmpl and "<|tool>" in tmpl) else 1)
