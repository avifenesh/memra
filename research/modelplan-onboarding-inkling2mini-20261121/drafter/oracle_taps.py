#!/usr/bin/env python3
"""Lane S: the oracle target's drafter taps and argmax along a given sequence (teacher-forced),
with lane E's oracle model (ink2_oracle.py, transformers PR #51050 code, experts as stored).

Taps as the vLLM drafter reads them (NOTES.md 1.1): the input of the layer visit at cache slots
5, 29 and 47 (the running partial block on entry) and the final norm output.

usage: oracle_taps.py ORACLE_DIR CKPT WORK PP_DIR --arm exact|actq [--chunk 512] [--threads 32]
writes PP_DIR/<id>.oracle.taps.f32 ([positions, 4, hidden] f32) and <id>.oracle.json (argmax)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("oracle_dir")
ap.add_argument("ckpt")
ap.add_argument("work")
ap.add_argument("pp_dir")
ap.add_argument("--arm", required=True)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--threads", type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, a.oracle_dir)
import ink2_oracle as O  # noqa: E402

torch.set_num_threads(a.threads)
m, model, cfg = O.load_model(a.ckpt, a.work, a.arm, "f32", {})
SLOTS = [5, 29, 47]
cur = {}


def pre(mod, args, kwargs):
    slot = mod.self_attn.layer_idx + kwargs.get("cache_offset", 0)
    if slot in SLOTS:
        cur[slot] = args[0][0].detach().float().clone()


def norm_post(mod, args, out):
    cur["final"] = out[0].detach().float().clone()


for layer in model.model.layers:
    layer.register_forward_pre_hook(pre, with_kwargs=True)
model.model.norm.register_forward_hook(norm_post)

for jf in sorted(Path(a.pp_dir).glob("*.json")):
    if jf.name.endswith(".oracle.json"):
        continue
    meta = json.load(open(jf))
    pid, seq = meta["id"], meta["seq"]
    if (Path(a.pp_dir) / f"{pid}.oracle.json").exists():
        continue
    t0 = time.time()
    cache = None
    taps, argmax = [], []
    with torch.no_grad():
        for s in range(0, len(seq), a.chunk):
            cur.clear()
            x = torch.tensor(seq[s:s + a.chunk], dtype=torch.long)[None]
            out = model(input_ids=x, cache=cache, use_cache=True)
            cache = out.cache
            taps.append(torch.stack([cur[5], cur[29], cur[47], cur["final"]], 1).numpy())
            argmax.extend(out.logits[0].argmax(-1).tolist())
    np.concatenate(taps).astype(np.float32).tofile(Path(a.pp_dir) / f"{pid}.oracle.taps.f32")
    json.dump({"id": pid, "arm": a.arm, "dtype": "f32", "argmax": argmax, "seconds": time.time() - t0},
              open(Path(a.pp_dir) / f"{pid}.oracle.json", "w"))
    print(pid, len(seq), f"{time.time() - t0:.0f}s", flush=True)
print("ORACLE_TAPS_DONE", flush=True)
