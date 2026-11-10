#!/usr/bin/env python3
"""memra #1158 step 1: write a text-trunk copy of an HF safetensors checkpoint dir.

Every tensor whose name starts with `model.visual.` is dropped; every other tensor is copied
byte for byte, in the source order, into shards of the same names. config.json, the tokenizer
and the chat template are copied unchanged, and model.safetensors.index.json loses the dropped
names. A manifest (JSON) records, for both dirs, each shard's sha256 and, per kept tensor, the
sha256 of its bytes in the source and in the copy, which must be equal.

usage: strip_tower.py SRC_DIR DST_DIR MANIFEST.json
"""
import glob
import hashlib
import json
import os
import shutil
import struct
import sys

SRC, DST, MANIFEST = sys.argv[1:4]
DROP = "model.visual."
os.makedirs(DST, exist_ok=False)


def read_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


manifest = {"source": SRC, "copy": DST, "dropped_prefix": DROP, "shards": [], "tensors": {}}
dropped_total = kept_total = 0
for src in sorted(glob.glob(os.path.join(SRC, "*.safetensors"))):
    name = os.path.basename(src)
    header, base = read_header(src)
    meta = header.pop("__metadata__", None)
    # data order = source offset order, so the copy is a contiguous re-pack of the kept tensors
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    kept = [(k, v) for k, v in items if not k.startswith(DROP)]
    dropped = [k for k, _ in items if k.startswith(DROP)]
    new_header = {}
    cursor = 0
    for k, v in kept:
        size = v["data_offsets"][1] - v["data_offsets"][0]
        new_header[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [cursor, cursor + size]}
        cursor += size
    out_header = dict(new_header)
    if meta is not None:
        out_header = {"__metadata__": meta, **new_header}
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    dst = os.path.join(DST, name)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for k, v in kept:
            a, b = v["data_offsets"]
            fin.seek(base + a)
            h = hashlib.sha256()
            left = b - a
            while left:
                chunk = fin.read(min(left, 1 << 24))
                assert chunk, f"short read in {name}:{k}"
                h.update(chunk)
                fout.write(chunk)
                left -= len(chunk)
            manifest["tensors"][k] = {"shard": name, "bytes": b - a, "sha256_source": h.hexdigest()}
    # re-read the copy and hash every kept tensor from it
    header2, base2 = read_header(dst)
    header2.pop("__metadata__", None)
    with open(dst, "rb") as fh:
        for k, v in header2.items():
            a, b = v["data_offsets"]
            fh.seek(base2 + a)
            h = hashlib.sha256()
            left = b - a
            while left:
                chunk = fh.read(min(left, 1 << 24))
                h.update(chunk)
                left -= len(chunk)
            manifest["tensors"][k]["sha256_copy"] = h.hexdigest()
            assert manifest["tensors"][k]["sha256_copy"] == manifest["tensors"][k]["sha256_source"], k
    manifest["shards"].append(
        {
            "name": name,
            "source_sha256": file_sha(src),
            "copy_sha256": file_sha(dst),
            "kept": len(kept),
            "dropped": len(dropped),
        }
    )
    dropped_total += len(dropped)
    kept_total += len(kept)

for src in sorted(glob.glob(os.path.join(SRC, "*"))):
    name = os.path.basename(src)
    if name.endswith(".safetensors") or os.path.isdir(src):
        continue
    dst = os.path.join(DST, name)
    if name == "model.safetensors.index.json":
        idx = json.load(open(src))
        idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if not k.startswith(DROP)}
        json.dump(idx, open(dst, "w"), indent=2)
    else:
        shutil.copyfile(src, dst)
manifest["kept_tensors"] = kept_total
manifest["dropped_tensors"] = dropped_total
json.dump(manifest, open(MANIFEST, "w"), indent=1, sort_keys=True)
print(f"kept {kept_total} tensors, dropped {dropped_total} ({DROP}*), every kept tensor byte-identical")
