#!/usr/bin/env python3
"""gguf_bytes.py FILE: the tensor table of a GGUF file (header only): per tensor its name, type id,
shape and stored bytes (the distance to the next tensor's offset, so padding counts with it), then
sums by group: trunk blocks, the MTP block(s) past the trunk, the output head, the embedding."""
import collections, os, struct, sys

f = open(sys.argv[1], "rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s(): return f.read(u64()).decode("utf-8", "replace")
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(t):
    if t == 8: return s()
    if t == 9:
        et, n = u32(), u64()
        return [val(et) for _ in range(n)] if et in (8, 9) else f.read(SCALAR[et] * n) and None
    raw = f.read(SCALAR[t])
    return struct.unpack({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t], raw)[0]
assert f.read(4) == b"GGUF"
ver, nt, nkv = u32(), u64(), u64()
kv = {}
for _ in range(nkv):
    k = s(); kv[k] = val(u32())
align = kv.get("general.alignment", 32)
ts = []
for _ in range(nt):
    name = s(); nd = u32(); dims = [u64() for _ in range(nd)]; typ = u32(); off = u64()
    ts.append([name, typ, dims, off])
data0 = (f.tell() + align - 1) // align * align
size = os.path.getsize(sys.argv[1])
ts.sort(key=lambda t: t[3])
for i, t in enumerate(ts):
    t.append((ts[i + 1][3] if i + 1 < len(ts) else size - data0) - t[3])
nl = kv.get(next((k for k in kv if k.endswith(".block_count")), ""), None)
print("arch", kv.get("general.architecture"), "block_count", nl, {k: v for k, v in kv.items() if k.endswith(("head_count", "head_count_kv", "key_length", "value_length", "embedding_length", "nextn_predict_layers", "full_attention_interval", "ssm.state_size", "ssm.group_count", "ssm.inner_size", "ssm.time_step_rank", "feed_forward_length", "vocab_size"))})
grp = collections.Counter(); bytype = collections.Counter()
for name, typ, dims, off, nb in ts:
    if name.startswith("blk."):
        b = int(name.split(".")[1])
        g = "trunk" if nl is None or b < nl - kv.get(next((k for k in kv if k.endswith("nextn_predict_layers")), ""), 0) else "mtp"
    else:
        g = name
    grp[g] += nb; bytype[(g if g in ("trunk", "mtp") else "other", typ)] += nb
for g, v in grp.most_common():
    print(f"{v / 1e9:9.4f} GB  {g}")
print("by group and type id:", {f"{g}/{t}": round(v / 1e9, 4) for (g, t), v in sorted(bytype.items())})
for name, typ, dims, off, nb in ts:
    if not name.startswith("blk.") or name.startswith(("blk.0.", "blk.3.", "blk.64.", "blk.63.")):
        print(f"  {name:48s} type {typ:3d} dims {dims} {nb / 1e6:.3f} MB")
