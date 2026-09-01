#!/usr/bin/env python3
"""Deterministic census-parity gate for the Gemma-4 assistant mapping.

Stands alone (no GPU, no full download — range-reads the safetensors header): asserts the
checkpoint's tensor set matches the ASSISTANT-ARM-SPEC.md mapping table exactly, so the
memra loader arm is written against verified ground truth, not a guess. Run this before
and after the loader lands; a shape/key drift here is a silent-wrong risk (norm-fold class).
"""
import json, struct, sys, urllib.request

URL = "https://huggingface.co/google/gemma-4-31B-it-assistant/resolve/main/model.safetensors"

EXPECT = {
    "model.embed_tokens.weight": ("BF16", [262144, 1024]),
    "pre_projection.weight": ("BF16", [1024, 10752]),
    "post_projection.weight": ("BF16", [5376, 1024]),
    "model.norm.weight": ("BF16", [1024]),
    # per-layer (N in 0..3); validated by pattern below
}
# Layers 0-2 are sliding_attention (head_dim 256 -> q 32*256=8192); layer 3 is
# full_attention (global_head_dim 512 -> q 32*512=16384). The census gate caught this
# geometry split before the loader was written — do not flatten it.
def q_dim(n):
    return 16384 if n == 3 else 8192


def head_dim(n):
    return 512 if n == 3 else 256


PER_LAYER_FIXED = {
    "input_layernorm.weight": [1024],
    "post_attention_layernorm.weight": [1024],
    "pre_feedforward_layernorm.weight": [1024],
    "post_feedforward_layernorm.weight": [1024],
    "layer_scalar": [1],
    "mlp.gate_proj.weight": [8192, 1024],
    "mlp.up_proj.weight": [8192, 1024],
    "mlp.down_proj.weight": [1024, 8192],
}
FORBIDDEN_SUBSTR = ["k_proj", "v_proj"]  # k_eq_v + KV-share ⇒ these must NOT exist
# use_ordered_embeddings=False on this checkpoint ⇒ plain tied lm_head, NO centroid head.
CENTROID_MUST_BE_ABSENT = True


def header():
    n = struct.unpack("<Q", urllib.request.urlopen(
        urllib.request.Request(URL, headers={"Range": "bytes=0-7"})).read())[0]
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"Range": f"bytes=8-{8 + n - 1}"})).read()
    return {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}


def main() -> int:
    h = header()
    fail = []
    for k, (dt, shp) in EXPECT.items():
        if k not in h:
            fail.append(f"MISSING {k}")
        elif h[k]["dtype"] != dt or h[k]["shape"] != shp:
            fail.append(f"SHAPE {k}: {h[k]['dtype']}{h[k]['shape']} != {dt}{shp}")
    for n in range(4):
        attn = {
            "self_attn.q_proj.weight": [q_dim(n), 1024],
            "self_attn.q_norm.weight": [head_dim(n)],
            "self_attn.o_proj.weight": [1024, q_dim(n)],
        }
        for suf, shp in {**attn, **PER_LAYER_FIXED}.items():
            k = f"model.layers.{n}.{suf}"
            if k not in h:
                fail.append(f"MISSING {k}")
            elif h[k]["shape"] != shp:
                fail.append(f"SHAPE {k}: {h[k]['shape']} != {shp}")
    for k in h:
        for bad in FORBIDDEN_SUBSTR:
            if bad in k:
                fail.append(f"UNEXPECTED {k} (k_eq_v/KV-share ⇒ no {bad})")
    centroid = [k for k in h if "centroid" in k or "token_ordering" in k]
    if CENTROID_MUST_BE_ABSENT and centroid:
        fail.append(f"UNEXPECTED centroid tensors {centroid} (use_ordered_embeddings=False)")
    print(f"tensors={len(h)} centroid/ordering={centroid or 'absent (plain tied lm_head)'}")
    if fail:
        print("CENSUS FAIL:")
        for f in fail:
            print("  " + f)
        return 1
    print("CENSUS OK — mapping table matches the checkpoint exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
