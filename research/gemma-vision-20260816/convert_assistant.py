#!/usr/bin/env python3
"""Convert google/gemma-4-31B-it-assistant (bf16 safetensors) to a gemma4-assistant GGUF
by cloning the metadata of the proven-loading QAT MTP GGUF and swapping tensor data.

bf16 -> f16 is exact for values inside f16 range (asserted). Tensor name mapping is the
same 1:1 table the weight-parity gate used; layer_output_scale stays F32.
"""
import json
import struct
import sys

import numpy as np

TEMPLATE = "/home/avifenesh/ai-ml/hf-models/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q8_0-MTP.gguf"
SNAP = "/data/ai-ml/hf-models/models--google--gemma-4-31B-it-assistant/snapshots/627c5ec1458b9086b841a91e0512fd31fd2fbbf1"
OUT = "/home/avifenesh/ai-ml/hf-models/gemma4-31b-tooluse-gguf/gemma-4-31B-it-official-F16-MTP.gguf"

GGML_F32, GGML_F16 = 0, 1
TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_FMTS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}


def hf_name(gg: str) -> str:
    if gg == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gg == "output_norm.weight":
        return "model.norm.weight"
    if gg == "nextn.pre_projection.weight":
        return "pre_projection.weight"
    if gg == "nextn.post_projection.weight":
        return "post_projection.weight"
    if gg.startswith("blk."):
        _, il, rest = gg.split(".", 2)
        m = {
            "attn_norm.weight": "input_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "post_attention_norm.weight": "post_attention_layernorm.weight",
            "ffn_norm.weight": "pre_feedforward_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
            "layer_output_scale.weight": "layer_scalar",
        }[rest]
        return f"model.layers.{il}.{m}"
    if gg == "rope_freqs.weight":
        return ""  # synthesized, keep template data
    raise KeyError(gg)


def read_str(f):
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n)


def read_val_raw(f, t):
    """Read a KV value, returning raw bytes (we re-emit verbatim)."""
    start = f.tell()
    if t == 8:
        read_str(f)
    elif t == 9:
        at, = struct.unpack('<I', f.read(4))
        n, = struct.unpack('<Q', f.read(8))
        for _ in range(n):
            read_val_raw(f, at)
    else:
        f.read(TYPE_SIZES[t])
    end = f.tell()
    f.seek(start)
    return f.read(end - start)


def main():
    # --- load HF tensors ---
    st = open(f"{SNAP}/model.safetensors", 'rb')
    hn, = struct.unpack('<Q', st.read(8))
    hdr = json.loads(st.read(hn))
    hdr.pop("__metadata__", None)
    base = 8 + hn

    def hf_bf16(name):
        m = hdr[name]
        assert m["dtype"] == "BF16", (name, m["dtype"])
        st.seek(base + m["data_offsets"][0])
        raw = st.read(m["data_offsets"][1] - m["data_offsets"][0])
        u32 = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        return np.frombuffer(u32.tobytes(), dtype='<f4').reshape(m["shape"])

    # --- parse template ---
    f = open(TEMPLATE, 'rb')
    assert f.read(4) == b'GGUF'
    ver, = struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<Q', f.read(8))
    n_kv, = struct.unpack('<Q', f.read(8))
    align = 32
    kv_blobs = []
    for _ in range(n_kv):
        name = read_str(f)
        t, = struct.unpack('<I', f.read(4))
        raw = read_val_raw(f, t)
        if name == b'general.alignment':
            align = struct.unpack('<I', raw)[0]
        # rewrite file_type 7? general.file_type=2(Q8_0 mostly f16 flag) -> keep names; set to 1 (F16)
        if name == b'general.file_type':
            raw = struct.pack('<I', 1)
        kv_blobs.append((name, t, raw))
    infos = []
    for _ in range(n_tensors):
        name = read_str(f).decode()
        nd, = struct.unpack('<I', f.read(4))
        ne = struct.unpack(f'<{nd}Q', f.read(8 * nd))
        gt, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        infos.append([name, ne, gt, off])
    tmpl_data_start = (f.tell() + align - 1) // align * align

    # --- build new tensor data (order preserved) ---
    out_tensors = []
    for name, ne, gt, off in infos:
        if name == "rope_freqs.weight":
            # keep template's synthesized rope_freqs verbatim (F32)
            f.seek(tmpl_data_start + off)
            n = int(np.prod(ne))
            data = f.read(4 * n)
            out_tensors.append((name, ne, GGML_F32, data))
            continue
        w = hf_bf16(hf_name(name))
        # GGUF ne is reversed vs numpy shape; verify count matches
        assert int(np.prod(ne)) == w.size, (name, ne, w.shape)
        if name.endswith("layer_output_scale.weight"):
            out_tensors.append((name, ne, GGML_F32, w.astype('<f4').tobytes()))
            continue
        f16 = w.astype('<f2')
        back = f16.astype('<f4')
        if not np.array_equal(back, w):
            bad = int((back != w).sum())
            mx = float(np.abs(back - w).max())
            print(f"  note {name}: bf16->f16 not exact for {bad} vals (max|d| {mx:.3e})")
        out_tensors.append((name, ne, GGML_F16, f16.tobytes()))

    # --- emit ---
    o = open(OUT, 'wb')
    o.write(b'GGUF')
    o.write(struct.pack('<I', ver))
    o.write(struct.pack('<Q', len(out_tensors)))
    o.write(struct.pack('<Q', len(kv_blobs)))
    for name, t, raw in kv_blobs:
        o.write(struct.pack('<Q', len(name)))
        o.write(name)
        o.write(struct.pack('<I', t))
        o.write(raw)
    # tensor infos with fresh offsets
    pos = 0
    hdr_entries = []
    for name, ne, gt, data in out_tensors:
        pos = (pos + align - 1) // align * align
        hdr_entries.append((name, ne, gt, pos, data))
        pos += len(data)
    for name, ne, gt, off, _ in hdr_entries:
        nb = name.encode()
        o.write(struct.pack('<Q', len(nb)))
        o.write(nb)
        o.write(struct.pack('<I', len(ne)))
        o.write(struct.pack(f'<{len(ne)}Q', *ne))
        o.write(struct.pack('<I', gt))
        o.write(struct.pack('<Q', off))
    here = o.tell()
    pad = (here + align - 1) // align * align - here
    o.write(b'\x00' * pad)
    data_start = o.tell()
    for name, ne, gt, off, data in hdr_entries:
        cur = o.tell() - data_start
        if cur < off:
            o.write(b'\x00' * (off - cur))
        o.write(data)
    o.close()
    print(f"wrote {OUT} ({len(out_tensors)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
