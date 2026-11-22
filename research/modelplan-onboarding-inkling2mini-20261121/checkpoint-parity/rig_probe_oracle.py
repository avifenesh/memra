#!/usr/bin/env python3
"""The nvfp4-rig class through lane E's CPU oracle, on the real checkpoint: one probe's last-token logits.

Lane M's tiny_rig_oracle.py applied to the pinned checkpoint and its minted overlay, without writing a merged
copy of the 17.6 GB file:
- minted dense tensors (attention projections of every layer, the output projection) are dequantized from the
  overlay's bytes, fl32(code * fl32(e4m3(scale) * scale_2)), and replace the BF16 tensors in the oracle's
  non-expert work file;
- minted expert banks (layers 20..23, MXFP8 converted to NVFP4) are read from the overlay: the oracle's raw
  reader looks a name up in the overlay first, so its expert store reads them as NVFP4 (it keys on the dtype);
- every k_proj / v_proj output (what the cache stores) is rounded to the nearest e4m3, ties to even,
  saturating at 448 (a forward hook);
- the experts run lane E's actq arm (the declared NVFP4 W4A4 program) on an f32 stream.

usage: python -I rig_probe_oracle.py --ckpt CKPT --overlay OVERLAY_DIR --oracle-dir LANE_E_DIR --probe TSV
       --work SCRATCH --out OUT.tsv   (TSV: any memra-checkpoint-oracle-v1 tape whose tokens are the probe)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CLASS = "inkling2-nvfp4-rig"


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ckpt", "--overlay", "--oracle-dir", "--probe", "--work", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, args.oracle_dir)
    import ink2_oracle as O
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(args.threads)
    E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
    P = np.zeros(127)
    for b in range(127):
        e, m = b >> 3, b & 7
        P[b] = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    E4M3_POS = np.concatenate([P, np.zeros(129)]).astype(np.float64)

    def dequant_dense(codes, scales, t):
        t = np.float32(np.asarray(t).reshape(-1)[0])
        lo, hi = E2M1[codes & 15], E2M1[codes >> 4]
        vals = np.stack([lo, hi], axis=-1).reshape(codes.shape[0], -1)
        st = (E4M3_POS[scales].astype(np.float32) * t).astype(np.float32)
        return (vals.reshape(vals.shape[0], -1, 16) * st[..., None]).astype(np.float32).reshape(vals.shape)

    Pt = torch.from_numpy(P)

    def e4m3_unit(x):
        a = x.abs().clamp(max=448.0).double()
        hi = torch.searchsorted(Pt, a.contiguous()).clamp(1, 126)
        lo = hi - 1
        da, db = a - Pt[lo], Pt[hi] - a
        pick = (db < da) | ((db == da) & (hi % 2 == 0))
        return torch.copysign(Pt[torch.where(pick, hi, lo)].to(x.dtype), x)

    over_path = os.path.join(args.overlay, "overlay.safetensors")
    over = O.STFile(over_path)
    names = over.keys()
    dense = sorted(n for n in names if n.endswith(".weight") and ".mlp.experts." not in n)
    banks = sorted(n for n in names if n.endswith("_weight") and ".mlp.experts." in n)
    # the work file: the base non-expert tensors, minted dense ones dequantized from the overlay
    work = os.path.join(args.work, "filtered")
    os.makedirs(work, exist_ok=True)
    src = O.weights_file(args.ckpt)
    out = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.llm.") and not O.is_expert_key(k):
                out[k] = f.get_tensor(k)
    for n in dense:
        if n not in out:
            raise SystemExit(f"minted {n} is not a base non-expert tensor")
        out[n] = torch.from_numpy(dequant_dense(np.asarray(over.raw(n)), np.asarray(over.raw(n + "_scale")), np.asarray(over.raw(n + "_scale_2"))))
    save_file(out, os.path.join(work, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(work, "generation_config.json"), "w") as g:
        g.write("{}")
    del out
    # the raw reader: overlay first, then the base file
    Base = O.STFile

    class Chain:
        def __init__(self, path):
            self.b, self.o = Base(path), over

        def keys(self):
            return sorted(set(self.b.keys()) | set(self.o.keys()))

        def info(self, k):
            return (self.o if k in self.o.h else self.b).info(k)

        def raw(self, k):
            return (self.o if k in self.o.h else self.b).raw(k)

    O.STFile = Chain
    _, model, cfg = O.load_model(args.ckpt, work, "actq", "f32", {})
    hooks = 0
    for layer in model.model.layers:
        for nm in ("k_proj", "v_proj"):
            getattr(layer.self_attn, nm).register_forward_hook(lambda mod, inp, o: e4m3_unit(o))
            hooks += 1
    ids = None
    for line in open(args.probe):
        f = line.rstrip("\n").split("\t")
        if f[0] == "tokens":
            ids = [int(t) for t in f[1].split(",")]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(ids, dtype=torch.long)[None], cache=None, use_cache=True).logits[0, -1]
    lg = logits.float()[: cfg.unpadded_vocab_size].numpy()
    bits = np.ascontiguousarray(lg, dtype="<f4").view("<u4")
    with open(args.out, "w") as f:
        f.write(f"format\tmemra-checkpoint-oracle-v1\nengine\tlane-e-cpu-actq-nvfp4-rig\nnumeric_class\t{CLASS}\n")
        f.write("tokens\t" + ",".join(map(str, ids)) + "\n")
        f.write(f"vocab\t{len(bits)}\n")
        for i, b in enumerate(bits):
            f.write(f"logit\t{i}\t{int(b):08x}\n")
    print(json.dumps({"arm": "actq+nvfp4-rig", "positions": len(ids), "argmax": int(lg.argmax()), "dense_minted": len(dense),
                      "banks_minted": len(banks), "kv_hooks": hooks, "overlay_sha256": O.sha256_file(over_path), "pr_head": O.PR_HEAD}))


if __name__ == "__main__":
    main()
