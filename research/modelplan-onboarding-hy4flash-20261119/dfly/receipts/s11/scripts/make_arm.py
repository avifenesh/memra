#!/usr/bin/env python3
"""Build a drafter arm of the Hy4-flash NVFP4 artifact: the same target, its DFly2 drafter's
tensors replaced by the BF16 source's (tencent/Hy4-flash@d88c9dfd, shard 148, every
model.dfly_layers.* tensor).

usage: make_arm.py ARTIFACT_DIR SOURCE_SHARD OUT_DIR MODE
  MODE bf16      every drafter tensor from the source (MLP, hidden correction and attention)
  MODE bf16-mlp  only the tensors the artifact holds in NVFP4 (the five MLPs and the hidden
                 correction); the FP8 attention linears stay the artifact's

The target shards are hard links (the loader refuses symlinked shards). The drafter's shard
(the one holding model.dfly_layers.*) is rewritten: its other tensors byte for byte, the replaced
drafter tensors from the source byte for byte, their quant sidecars dropped. The index and the
three quant maps (config.json quantization_config, hf_quant_config.json, quant_config.json)
drop the replaced modules, so the arm describes itself: those modules are unquantized BF16.
"""
import hashlib
import json
import os
import struct
import sys

art, src_path, out, mode = sys.argv[1:5]
assert mode in ("bf16", "bf16-mlp"), mode


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, h


def replaced_module(name):
    """the drafter module a tensor belongs to when this arm replaces it, else None"""
    if not name.startswith("model.dfly_layers."):
        return None
    if mode == "bf16-mlp" and not (".mlp." in name or ".hidden_correction." in name):
        return None
    for suf in (".weight_scale_inv", ".weight_scale_2", ".weight_scale", ".weight"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


os.makedirs(out, exist_ok=True)
idx = json.load(open(os.path.join(art, "model.safetensors.index.json")))
wm = idx["weight_map"]
dshards = sorted({f for n, f in wm.items() if n.startswith("model.dfly_layers.")})
assert len(dshards) == 1, dshards
dshard = dshards[0]
sn, sh = header(src_path)
src_names = {k for k in sh if k != "__metadata__"}
# the source's drafter tensors this arm takes
take = sorted(k for k in src_names if replaced_module(k) is not None)
mods = sorted({replaced_module(k) for k in take})
an, ah = header(os.path.join(art, dshard))
# artifact tensors of the drafter shard that the arm drops (the replaced modules' tensors)
drop = sorted(k for k in ah if k != "__metadata__" and replaced_module(k) is not None
              and replaced_module(k) in mods)
print(f"mode {mode}: {len(take)} source tensors, {len(mods)} modules, drop {len(drop)} artifact tensors")
for k in take:
    assert k in ah, f"source tensor {k} not in the artifact"
    assert sh[k]["dtype"] == "BF16", (k, sh[k]["dtype"])

# hard links for every other file of the artifact
for f in sorted(os.listdir(art)):
    p = os.path.join(art, f)
    if f in (dshard, "model.safetensors.index.json", "config.json", "hf_quant_config.json",
             "quant_config.json") or os.path.isdir(p):
        continue
    q = os.path.join(out, f)
    if not os.path.exists(q):
        os.link(p, q)

# the rewritten drafter shard
entries = []  # (name, dtype, shape, file, begin, end)
for k, v in ah.items():
    if k == "__metadata__" or k in drop:
        continue
    entries.append((k, v["dtype"], v["shape"], os.path.join(art, dshard), 8 + an + v["data_offsets"][0],
                    8 + an + v["data_offsets"][1]))
for k in take:
    v = sh[k]
    entries.append((k, v["dtype"], v["shape"], src_path, 8 + sn + v["data_offsets"][0], 8 + sn + v["data_offsets"][1]))
entries.sort(key=lambda e: e[0])
hdr, off = {}, 0
meta = dict(ah.get("__metadata__") or {})
meta["note"] = f"modified, see README; drafter arm {mode}: model.dfly_layers modules from tencent/Hy4-flash@d88c9dfd BF16"
hdr["__metadata__"] = meta
for name, dt, shape, _, b, e in entries:
    hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + (e - b)]}
    off += e - b
hb = json.dumps(hdr, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
tmp = os.path.join(out, dshard + ".tmp")
skip_shard = os.environ.get("JSON_ONLY") == "1" and os.path.exists(os.path.join(out, dshard))
if skip_shard:
    tmp = None
with open(tmp or os.devnull, "wb") as w:
    w.write(struct.pack("<Q", len(hb)))
    w.write(hb)
    for name, dt, shape, path, b, e in entries:
        with open(path, "rb") as r:
            r.seek(b)
            left = e - b
            while left:
                chunk = r.read(min(left, 64 << 20))
                assert chunk, (name, path)
                w.write(chunk)
                left -= len(chunk)
if tmp:
    os.replace(tmp, os.path.join(out, dshard))

# byte check: every kept tensor equals its source bytes
_, nh = header(os.path.join(out, dshard))
nn = struct.unpack("<Q", open(os.path.join(out, dshard), "rb").read(8))[0]
with open(os.path.join(out, dshard), "rb") as o:
    for name, dt, shape, path, b, e in entries:
        v = nh[name]
        o.seek(8 + nn + v["data_offsets"][0])
        got = hashlib.sha256(o.read(e - b)).digest()
        with open(path, "rb") as r:
            r.seek(b)
            want = hashlib.sha256(r.read(e - b)).digest()
        assert got == want, name

# index
wm2 = {k: f for k, f in wm.items() if k not in drop}
for k in take:
    wm2[k] = dshard
idx2 = dict(idx)
idx2["weight_map"] = dict(sorted(wm2.items()))
json.dump(idx2, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)


dropped = set()


def strip(obj):
    """drop the replaced modules from every quantized_layers map"""
    n = 0
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "quantized_layers" and isinstance(v, dict):
                for m in mods:
                    if m in v:
                        del v[m]
                        dropped.add(m)
                        n += 1
            else:
                n += strip(v)
    elif isinstance(obj, list):
        for v in obj:
            n += strip(v)
    return n


def exclude(obj):
    """list the dropped modules in every exclusion list (`ignore`, `exclude_modules`): the pack
    reads a module neither quantized nor excluded as an incomplete map"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("ignore", "exclude_modules") and isinstance(v, list):
                for m in sorted(dropped):
                    if m not in v:
                        v.append(m)
            else:
                exclude(v)
    elif isinstance(obj, list):
        for v in obj:
            exclude(v)


for f in ("config.json", "hf_quant_config.json", "quant_config.json"):
    j = json.load(open(os.path.join(art, f)))
    n = strip(j)
    exclude(j)
    json.dump(j, open(os.path.join(out, f), "w"), indent=2)
    print(f"{f}: {n} module entries dropped, listed as excluded")
print(f"ARM-OK {out} {len(entries)} tensors in {dshard}, {off / 1e9:.2f} GB data")
