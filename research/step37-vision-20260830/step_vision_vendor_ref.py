#!/usr/bin/env python3
"""Vendor-code reference for the step37 vision tower: the checkpoint's OWN
vision_encoder.py (StepRoboticsVisionEncoder) + the modeling file's
_process_image_features law (downsampler1/2 + vit_large_projector), run offline on
the exact pixels the memra oracle dumped. This is the authoritative arm of the parity
oracle; step_vision_ref.py is the independent NumPy arm. External implementations run
ONLY offline to create pinned oracle evidence, never as a serving path.

Usage: step_vision_vendor_ref.py <model_dir> <dump_dir>

Loads the encoder class through transformers' dynamic-module machinery
(trust_remote_code on the LOCAL pinned files only), the 667 BF16 vision tensors from
the shards, reconstructs the [1,3,H,W] pixel tensor from patches.bin (the patchify is
a lossless permutation), and prints per-token cosine vs the memra stage dumps.
"""
import json
import os
import struct
import sys

import numpy as np
import torch

model_dir, dump = sys.argv[1], sys.argv[2]
sys.path.insert(0, model_dir)

# The vendor files use package-relative imports; load them as a package.
import importlib.util


def load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(
        f"step37_vendor.{name}", os.path.join(model_dir, fname)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"step37_vendor.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


import types

pkg = types.ModuleType("step37_vendor")
pkg.__path__ = [model_dir]
sys.modules["step37_vendor"] = pkg
cfg_mod = load_mod("configuration_step3p7", "configuration_step3p7.py")
enc_mod = load_mod("vision_encoder", "vision_encoder.py")

config = json.load(open(os.path.join(model_dir, "config.json")))
vcfg = cfg_mod.StepRoboticsVisionEncoderConfig(**config["vision_config"])
encoder = enc_mod.StepRoboticsVisionEncoder(vcfg).eval()

# ---- load the vision weights from the shards ----
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
weight_map = idx["weight_map"]
_shards = {}


def raw_tensor(name):
    fname = weight_map[name]
    if fname not in _shards:
        f = open(os.path.join(model_dir, fname), "rb")
        (hlen,) = struct.unpack("<Q", f.read(8))
        _shards[fname] = (f, 8 + hlen, json.loads(f.read(hlen)))
    f, base, header = _shards[fname]
    info = header[name]
    lo, hi = info["data_offsets"]
    f.seek(base + lo)
    raw = f.read(hi - lo)
    assert info["dtype"] == "BF16", (name, info["dtype"])
    u16 = np.frombuffer(raw, dtype=np.uint16)
    arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    return torch.from_numpy(arr.copy())


prefix = "model.vision_model."
state = {}
for name in weight_map:
    if name.startswith(prefix):
        state[name[len(prefix) :]] = raw_tensor(name)
missing, unexpected = encoder.load_state_dict(state, strict=False)
# ls_1/ls_2 register as EncoderLayerScale.gamma; everything must land.
assert not missing, f"missing: {missing[:5]}"
assert not unexpected, f"unexpected: {unexpected[:5]}"
encoder = encoder.float()

proj_w = raw_tensor("model.vit_large_projector.weight").float()

# ---- rebuild pixels from the dumped patch rows (lossless permutation) ----
g = int(open(f"{dump}/grid.txt").read().split()[0])
n = g * g
patch = 14
rows = np.fromfile(f"{dump}/patches.bin", dtype=np.float32).reshape(n, 588)
side = g * patch
pix = np.zeros((3, side, side), dtype=np.float32)
for py in range(g):
    for px in range(g):
        r = rows[py * g + px].reshape(3, patch, patch)
        pix[:, py * patch : (py + 1) * patch, px * patch : (px + 1) * patch] = r
pixels = torch.from_numpy(pix).unsqueeze(0)

with torch.no_grad():
    feats = encoder(pixels)  # [1, n, 1536] (no ln_post, no cls)
    post_blocks = feats[0].numpy().astype(np.float64)
    # _process_image_features: [B,P,D] -> [B,D,HW,HW] -> down1 -> down2 -> flatten -> proj
    hw = int(feats.shape[1] ** 0.5)
    f = feats.permute(0, 2, 1).reshape(1, -1, hw, hw)
    f = encoder.vit_downsampler1(f)
    f = encoder.vit_downsampler2(f)
    b, c, oh, ow = f.shape
    f = f.reshape(b, c, oh * ow).permute(0, 2, 1)
    downsampled = f[0].numpy().astype(np.float64)
    projected = (f @ proj_w.T)[0].numpy().astype(np.float64)


def cos_report(stage, ours):
    ref = np.fromfile(f"{dump}/rust_{stage}.bin", dtype=np.float32).astype(np.float64)
    ref = ref.reshape(ours.shape)
    dot = (ours * ref).sum(-1)
    denom = np.sqrt((ours * ours).sum(-1) * (ref * ref).sum(-1))
    cos = dot / np.maximum(denom, 1e-30)
    print(f"vendor {stage:14s} min_cos {cos.min():.6f}  mean_cos {cos.mean():.6f}")
    return cos.min()


cos_report("post_blocks", post_blocks)
cos_report("downsampled", downsampled)
mc = cos_report("projected", projected)
bar = 0.9997
verdict = "PASS" if mc >= bar else "FAIL"
print(f"vendor projected min_cos {mc:.6f} vs bar {bar} -> {verdict}")
sys.exit(0 if verdict == "PASS" else 1)
