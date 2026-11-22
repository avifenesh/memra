#!/usr/bin/env python3
"""TinyParity oracle of the inkling2 nvfp4-rig class (lane M).

The declared activation-quant program simulated on an f32 stream (lane E's ink2_oracle.py, arm actq, through
reference/tiny_oracle.py's schedule runner), on the tiny checkpoint merged with its minted overlay, and with every
cache slot's keys and values stored as unit-scale e4m3:

- minted dense tensors (attention outside the loop core, the output projection): dequantized HERE from the
  overlay's bytes, fl32(code * fl32(e4m3(scale) * scale_2)), and loaded as F32 weights;
- minted expert banks (the tiny config's MXFP8 layers): the overlay's NVFP4 planes replace the MXFP8 banks, and
  the oracle's expert store reads them as NVFP4 (it keys on the stored dtype);
- K/V: a forward hook on every k_proj / v_proj (their output, after the key clip and the value norm, is what the
  cache stores) rounds each f32 element to the nearest e4m3, ties to even, saturating at 448.

Input: the image `memra model verify tiny --against inkling2 --out DIR` writes (DIR/tiny-checkpoint-nvfp4-rig/).
  python -I tiny_rig_oracle.py --image DIR/tiny-checkpoint-nvfp4-rig --work SCRATCH --out inkling2-nvfp4-rig.tsv
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "reference"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import ink2_oracle as O  # noqa: E402
import tiny_oracle as T  # noqa: E402

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e4m3_magnitudes():
    v = np.zeros(127, dtype=np.float64)
    for b in range(127):
        e, m = b >> 3, b & 7
        v[b] = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return v


E4M3_POS = e4m3_magnitudes()


def dequant_dense(codes, scales, t):
    t = np.float32(np.asarray(t).reshape(-1)[0])
    lo, hi = E2M1[codes & 15], E2M1[codes >> 4]
    vals = np.stack([lo, hi], axis=-1).reshape(codes.shape[0], -1)  # even element in the low nibble
    st = (E4M3_POS[scales].astype(np.float32) * t).astype(np.float32)
    return (vals.reshape(vals.shape[0], -1, 16) * st[..., None]).astype(np.float32).reshape(vals.shape)


def e4m3_unit(x):
    """f32 tensor -> nearest e4m3 (ties to the even byte), saturating at 448, sign kept."""
    a = x.abs().clamp(max=448.0).double()
    P = torch.from_numpy(E4M3_POS)
    hi = torch.searchsorted(P, a.contiguous()).clamp(1, 126)
    lo = hi - 1
    da, db = a - P[lo], P[hi] - a
    pick = (db < da) | ((db == da) & (hi % 2 == 0))
    return torch.copysign(P[torch.where(pick, hi, lo)].to(x.dtype), x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    merged = os.path.join(args.work, "merged")
    os.makedirs(merged, exist_ok=True)
    base = load_file(os.path.join(args.image, "model.safetensors"))
    over = load_file(os.path.join(args.image, "overlay.safetensors"))
    dense, banks = [], []
    for name in sorted(over):
        if not name.endswith(".weight") and not name.endswith("_weight"):
            continue
        if ".mlp.experts." in name:
            for plane in ("", "_scale", "_amax", "_outer_scale_denominator"):
                base[name + plane] = over[name + plane]
            banks.append(name)
        else:
            raw = lambda p: over[name + p].contiguous().view(torch.uint8).numpy() if over[name + p].dtype != torch.float32 else over[name + p].numpy()
            base[name] = torch.from_numpy(dequant_dense(raw(""), raw("_scale"), raw("_scale_2")))
            dense.append(name)
    save_file(base, os.path.join(merged, "model.safetensors"))
    for f in ("config.json", "hf_quant_config.json", "schedule.json"):
        shutil.copy(os.path.join(args.image, f), merged)
    work = os.path.join(args.work, "filtered")
    O.cmd_filter(argparse.Namespace(ckpt=merged, work=work))
    _, model, cfg = O.load_model(merged, work, "actq", "f32", {})
    hooks = 0
    for layer in model.model.layers:
        for nm in ("k_proj", "v_proj"):
            getattr(layer.self_attn, nm).register_forward_hook(lambda mod, inp, out: e4m3_unit(out))
            hooks += 1
    schedule = json.load(open(os.path.join(merged, "schedule.json")))
    ids = schedule["input_ids"]
    logits = T.run(model, ids, T.schedule_ends(schedule))
    assert logits.shape == (len(ids), cfg.unpadded_vocab_size), logits.shape
    bits = logits.to(torch.float32).numpy().view(np.uint32)
    with open(args.out, "w") as f:
        f.write("stream\tposition\ttoken\tlogit_f32_bits\n")
        for position in range(bits.shape[0]):
            for token in range(bits.shape[1]):
                f.write(f"main\t{position}\t{token}\t{int(bits[position, token]):08x}\n")
    print(json.dumps({"arm": "actq+nvfp4-rig", "dtype": "f32", "positions": len(ids), "vocab": int(bits.shape[1]),
                      "dense_minted": dense, "banks_minted": banks, "kv_hooks": hooks,
                      "overlay_sha256": O.sha256_file(os.path.join(args.image, "overlay.safetensors")),
                      "weights_sha256": O.sha256_file(os.path.join(args.image, "model.safetensors")),
                      "pr_head": O.PR_HEAD}))


if __name__ == "__main__":
    main()
