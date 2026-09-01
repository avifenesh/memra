#!/usr/bin/env python3
"""Independent (numpy-only, no safetensors lib) decode oracle for the DeepSeek-V4-Flash
NVFP4 artifact. Two jobs:

1. `probe`  — measure the quant geometry from bytes: NVFP4 expert scale structure
   (power-of-two check, 16-group pair sharing from the lossless MXFP4->NVFP4 cast),
   MXFP4 (MTP expert) E8M0 byte range, FP8 128x128 block-scale decode, stats.
2. `dump`   — decode ONE tensor to little-endian F32 bytes + print sha256, so the Rust
   path (dsv4-census --dump) can be cross-checked bit-for-bit (deliverable E).

Decode semantics implemented from hf_quant_config.json + config.json quantization_config
(NVFP4 group_size 16, scale_fmt ue8m0, weight_block_size [128,128]), independently of the
memra Rust implementation:
  NVFP4 expert:  w[r,c] = e2m1(code) * e4m3(weight_scale[r, c//16]) * weight_scale_2
  MXFP4 expert:  w[r,c] = e2m1(code) * 2^(scale[r, c//32] - 127)         (mtp.*)
  FP8 linear:    w[r,c] = e4m3(byte) * 2^(scale[r//128, c//128] - 127)
  packing:       element 2i -> low nibble, 2i+1 -> high nibble (modelopt order)
"""

import hashlib
import json
import mmap
import os
import struct
import sys

import numpy as np

MODEL_DIR = "/home/ubuntu/models/dsv4-flash-nvfp4"

# ---- minimal safetensors reader (stdlib only) ------------------------------------------------


def load_index(d):
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


class Shard:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        (hlen,) = struct.unpack("<Q", self.mm[:8])
        self.header = json.loads(self.mm[8 : 8 + hlen])
        self.base = 8 + hlen

    def raw(self, name):
        info = self.header[name]
        b, e = info["data_offsets"]
        return info, self.mm[self.base + b : self.base + e]


_shards = {}


def tensor(d, wm, name):
    sh = wm[name]
    if sh not in _shards:
        _shards[sh] = Shard(os.path.join(d, sh))
    return _shards[sh].raw(name)


# ---- element decoders --------------------------------------------------------------------------

# e2m1 (FP4) code -> value, sign bit at 0x8. Standard OCP FP4 table.
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e4m3_to_f32(b):
    """FP8 E4M3 (fn variant: 0x7f/0xff = NaN -> 0.0, no inf) byte array -> f32 array."""
    b = np.asarray(b, dtype=np.uint8)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    mag = b & 0x7F
    exp = (mag >> 3).astype(np.int32)
    man = (mag & 0x7).astype(np.float32)
    normal = (1.0 + man / 8.0) * np.exp2(exp - 7).astype(np.float32)
    sub = (man / 8.0) * np.float32(2.0**-6)
    v = np.where(exp == 0, sub, normal).astype(np.float32) * sign
    return np.where(mag == 0x7F, np.float32(0.0), v)


def e8m0_to_f32(b):
    """FP8 E8M0 (OCP MX scale): value = 2^(byte-127); 0xff = NaN -> propagate NaN."""
    b = np.asarray(b, dtype=np.uint8)
    v = np.exp2(b.astype(np.float32) - 127.0)
    return np.where(b == 0xFF, np.float32(np.nan), v).astype(np.float32)


def unpack_fp4(packed, rows, cols):
    """(rows, cols/2) packed bytes -> (rows, cols) e2m1 values. elem 2i = low nibble."""
    p = np.frombuffer(packed, dtype=np.uint8).reshape(rows, cols // 2)
    lo = p & 0x0F
    hi = p >> 4
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    return E2M1[codes]


def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---- full-tensor decoders ------------------------------------------------------------------------


def decode_nvfp4_expert(d, wm, stem):
    wi, wb = tensor(d, wm, stem + ".weight")
    si, sb = tensor(d, wm, stem + ".weight_scale")
    s2i, s2b = tensor(d, wm, stem + ".weight_scale_2")
    assert wi["dtype"] == "U8" and si["dtype"] == "F8_E4M3" and s2i["dtype"] == "F32"
    rows, half = wi["shape"]
    cols = half * 2
    assert si["shape"] == [rows, cols // 16], (si["shape"], rows, cols)
    codes = unpack_fp4(wb, rows, cols)
    scales = e4m3_to_f32(np.frombuffer(sb, dtype=np.uint8)).reshape(rows, cols // 16)
    (s2,) = struct.unpack("<f", s2b)
    w = codes * np.repeat(scales, 16, axis=1) * np.float32(s2)
    return w.astype(np.float32), scales, s2


def decode_mxfp4_expert(d, wm, stem):
    wi, wb = tensor(d, wm, stem + ".weight")
    si, sb = tensor(d, wm, stem + ".scale")
    assert wi["dtype"] == "I8" and si["dtype"] == "F8_E8M0"
    rows, half = wi["shape"]
    cols = half * 2
    assert si["shape"] == [rows, cols // 32], (si["shape"], rows, cols)
    codes = unpack_fp4(wb, rows, cols)
    scales = e8m0_to_f32(np.frombuffer(sb, dtype=np.uint8)).reshape(rows, cols // 32)
    return (codes * np.repeat(scales, 32, axis=1)).astype(np.float32), sb


def decode_fp8_linear(d, wm, stem):
    wi, wb = tensor(d, wm, stem + ".weight")
    si, sb = tensor(d, wm, stem + ".scale")
    assert wi["dtype"] == "F8_E4M3" and si["dtype"] == "F8_E8M0"
    rows, cols = wi["shape"]
    assert si["shape"] == [rows // 128, cols // 128], (si["shape"], rows, cols)
    w = e4m3_to_f32(np.frombuffer(wb, dtype=np.uint8)).reshape(rows, cols)
    scales = e8m0_to_f32(np.frombuffer(sb, dtype=np.uint8)).reshape(rows // 128, cols // 128)
    return (w * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)).astype(np.float32), sb


def stats(name, w):
    print(
        f"  {name}: shape={w.shape} min={w.min():.6g} max={w.max():.6g} "
        f"mean={w.mean():.6g} absmean={np.abs(w).mean():.6g} "
        f"zerofrac={(w == 0).mean():.4f} nan={np.isnan(w).sum()} inf={np.isinf(w).sum()}"
    )


def probe(d):
    wm = load_index(d)
    print(f"index: {len(wm)} tensors, {len(set(wm.values()))} shards")

    stem = "layers.20.ffn.experts.7.w1"
    w, scales, s2 = decode_nvfp4_expert(d, wm, stem)
    print(f"\n== NVFP4 expert {stem} (weight_scale_2 = {s2:.9g}) ==")
    stats("decoded", w)
    eff = scales.astype(np.float64) * s2
    nz = eff[eff > 0]
    l2 = np.log2(nz)
    frac = np.abs(l2 - np.round(l2))
    print(
        f"  effective scale (e4m3*scale_2): {nz.min():.4g}..{nz.max():.4g}; "
        f"power-of-two fraction-err max={frac.max():.3g} (0 => cast kept 2^m scales)"
    )
    pairs_equal = np.array_equal(scales[:, 0::2], scales[:, 1::2])
    print(f"  adjacent 16-group scale pairs identical (32-group ancestry): {pairs_equal}")
    zero_scale_frac = (scales == 0).mean()
    print(f"  zero e4m3 scale bytes: {zero_scale_frac:.6f}")

    stem = "mtp.0.ffn.experts.7.w1"
    w, sb = decode_mxfp4_expert(d, wm, stem)
    print(f"\n== MXFP4 (MTP) expert {stem} ==")
    stats("decoded", w)
    sbytes = np.frombuffer(sb, dtype=np.uint8)
    print(
        f"  e8m0 byte range: {sbytes.min()}..{sbytes.max()} "
        f"(= 2^{int(sbytes.min()) - 127}..2^{int(sbytes.max()) - 127}), 0xff count={int((sbytes == 255).sum())}"
    )

    stem = "layers.20.attn.wq_a"
    w, sb = decode_fp8_linear(d, wm, stem)
    print(f"\n== FP8+E8M0(128x128) linear {stem} ==")
    stats("decoded", w)
    sbytes = np.frombuffer(sb, dtype=np.uint8)
    print(f"  e8m0 byte range: {sbytes.min()}..{sbytes.max()}, 0xff count={int((sbytes == 255).sum())}")

    ei, eb = tensor(d, wm, "embed.weight")
    rows = bf16_to_f32(eb[: 4096 * 2 * 8]).reshape(8, 4096)
    print("\n== embed.weight rows 0..8 (BF16) ==")
    stats("rows0..8", rows)

    ti, tb = tensor(d, wm, "layers.0.ffn.gate.tid2eid")
    t = np.frombuffer(tb, dtype=np.int64).reshape(ti["shape"])
    print(f"\n== tid2eid layer0: shape={t.shape} min={t.min()} max={t.max()} (expect [0,256)) ==")


def dump(d, name, out):
    """Decode one tensor to LE f32 bytes; print sha256. Stem-quant is inferred from siblings."""
    wm = load_index(d)
    if name + ".weight_scale_2" in wm:
        w, _, _ = decode_nvfp4_expert(d, wm, name)
    elif name + ".scale" in wm and tensor(d, wm, name + ".weight")[0]["dtype"] == "I8":
        w, _ = decode_mxfp4_expert(d, wm, name)
    elif name + ".scale" in wm:
        w, _ = decode_fp8_linear(d, wm, name)
    else:
        raise SystemExit(f"no quant siblings for {name}")
    raw = np.ascontiguousarray(w, dtype="<f4").tobytes()
    with open(out, "wb") as f:
        f.write(raw)
    print(f"{name}: {w.shape} -> {out}")
    print(f"sha256 {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    d = os.environ.get("DSV4_DIR", MODEL_DIR)
    if mode == "probe":
        probe(d)
    elif mode == "dump":
        dump(d, sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: dsv4_decode_oracle.py [probe | dump <tensor-stem> <out.bin>]")
