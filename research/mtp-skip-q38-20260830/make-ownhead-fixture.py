#!/usr/bin/env python3
"""Refusal-tooth fixture for MEMRA_MTP_SKIP x MEMRA_FRSPEC_TRIM (mtp-skip lane, 2026-08-30).

Builds a metadata-faithful GGUF that copies the ENTIRE KV section of a real donor model
byte-for-byte (so ModelConfig and the ModelPlan compile exactly as the donor does) but whose
tensor list contains exactly ONE tensor: `blk.{n_trunk}.nextn.shared_head_head.weight`, the
per-block own lm_head marker of a step35-class artifact. Loading it with MEMRA_MTP_SKIP=1 +
MEMRA_FRSPEC_TRIM must hit the loader's own-head refusal, which fires BEFORE any tensor
upload, so the fixture never needs real weights.

usage: make-ownhead-fixture.py <donor.gguf> <out.gguf>
"""

import struct
import sys

TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def walk_kv_value(buf, pos, vtype):
    if vtype in TYPE_SIZES:
        return pos + TYPE_SIZES[vtype]
    if vtype == 8:  # string
        (n,) = struct.unpack_from("<Q", buf, pos)
        return pos + 8 + n
    if vtype == 9:  # array
        etype, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if etype in TYPE_SIZES:
            return pos + TYPE_SIZES[etype] * count
        if etype == 8:
            for _ in range(count):
                (n,) = struct.unpack_from("<Q", buf, pos)
                pos += 8 + n
            return pos
        raise ValueError(f"array elem type {etype}")
    raise ValueError(f"kv type {vtype}")


def main(donor_path, out_path):
    with open(donor_path, "rb") as f:
        head = f.read(64 << 20)  # metadata lives at the head; 64 MiB is plenty
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", head, 0)
    assert magic == 0x46554747, hex(magic)
    pos = 24
    kv_start = pos
    arch = None
    kvs = {}
    for _ in range(n_kv):
        (klen,) = struct.unpack_from("<Q", head, pos)
        key = head[pos + 8 : pos + 8 + klen].decode()
        pos += 8 + klen
        (vtype,) = struct.unpack_from("<I", head, pos)
        vpos = pos + 4
        pos = walk_kv_value(head, vpos, vtype)
        if vtype == 4:
            kvs[key] = struct.unpack_from("<I", head, vpos)[0]
        elif vtype == 8:
            (n,) = struct.unpack_from("<Q", head, vpos)
            kvs[key] = head[vpos + 8 : vpos + 8 + n].decode()
    kv_bytes = head[kv_start:pos]
    arch = kvs["general.architecture"]
    n_layer = kvs[f"{arch}.block_count"]
    nextn = kvs.get(f"{arch}.nextn_predict_layers", 0)
    n_trunk = n_layer - nextn
    align = kvs.get("general.alignment", 32)
    print(f"donor: arch={arch} n_layer={n_layer} nextn={nextn} -> n_trunk={n_trunk}")
    assert nextn > 0, "donor must declare nextn > 0 for the refusal arm to be reachable"

    name = f"blk.{n_trunk}.nextn.shared_head_head.weight".encode()
    ne = [64, 64]
    data = b"\x00" * (ne[0] * ne[1] * 4)  # F32 zeros; never read (refusal fires pre-upload)
    tinfo = struct.pack("<Q", len(name)) + name
    tinfo += struct.pack("<I", len(ne))
    for d in ne:
        tinfo += struct.pack("<Q", d)
    tinfo += struct.pack("<I", 0)  # GGML_TYPE_F32
    tinfo += struct.pack("<Q", 0)  # offset within data section

    out = struct.pack("<IIQQ", magic, version, 1, n_kv) + kv_bytes + tinfo
    pad = (-len(out)) % align
    out += b"\x00" * pad + data
    with open(out_path, "wb") as f:
        f.write(out)
    print(f"wrote {out_path}: {len(out)} bytes, tensor {name.decode()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
