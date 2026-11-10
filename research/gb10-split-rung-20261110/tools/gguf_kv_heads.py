#!/usr/bin/env python3
"""Print a GGUF's attention head counts the way the split ladder sees them: the scalar or
per-layer `*.attention.head_count_kv` key, `*.attention.head_count`, the context length, and for
every layer with an attention K projection the KV head count its shape implies (out features /
key length). Reads the header and tensor index only (no weights).
usage: gguf_kv_heads.py MODEL.gguf"""
import struct, sys

f = open(sys.argv[1], "rb")
def rd(fmt):
    n = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(n))
def rstr():
    (n,) = rd("Q"); return f.read(n).decode("utf-8", "replace")
SCAL = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(t):
    if t in SCAL: return rd(SCAL[t])[0]
    if t == 8: return rstr()
    if t == 9:
        (et,) = rd("I"); (n,) = rd("Q"); return [rval(et) for _ in range(n)]
    raise ValueError(t)
magic, = rd("I"); ver, = rd("I"); nt, = rd("Q"); nkv, = rd("Q")
meta = {}
for _ in range(nkv):
    k = rstr(); (t,) = rd("I"); meta[k] = rval(t)
arch = meta.get("general.architecture")
want = [f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv", f"{arch}.attention.key_length", f"{arch}.context_length", f"{arch}.block_count", f"{arch}.full_attention_interval"]
print("arch", arch)
for k in want:
    if k in meta:
        v = meta[k]
        if isinstance(v, list):
            nz = sorted(set(v))
            print(k, f"per layer, {len(v)} layers, values {nz}")
        else:
            print(k, v)
klen = meta.get(f"{arch}.attention.key_length")
kv_by_shape = {}
for _ in range(nt):
    name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd); (tt,) = rd("I"); (off,) = rd("Q")
    if name.endswith("attn_k.weight") and klen:
        kv_by_shape.setdefault(dims[1] // klen if len(dims) > 1 else None, []).append(name.split(".")[1])
for kv, layers in sorted(kv_by_shape.items(), key=lambda x: str(x[0])):
    print(f"attn_k shape -> {kv} KV heads in {len(layers)} layers ({','.join(layers[:6])}{'...' if len(layers) > 6 else ''})")
