#!/usr/bin/env python3
"""TinyParity oracle for the inkling2 pack: the float-activation implementation in f64.

Runs the transformers PR 51050 model code (head 6173a2f679ff4ed393afc499aefc569da7e68bc1), source-transformed to
f64 by ink2_oracle.py, on the checkpoint image that `memra model verify tiny --against inkling2 --out DIR` writes
(DIR/tiny-checkpoint*/: model.safetensors, config.json, hf_quant_config.json, schedule.json). The experts stay in
their stored bytes and each routed expert is dequantized exactly (ink2_oracle.py arm "exact").

The request runs under the image's schedule with the PR's cache: one call per prefill chunk, then one call per
decode token. Output: the pinned-oracle TSV (stream, position, token, logit_f32_bits), every position, the
unpadded vocabulary.

  python -I tiny_oracle.py --image DIR/tiny-checkpoint --work SCRATCH --out inkling2.tsv
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ink2_oracle as O


def schedule_ends(schedule):
    ids = schedule["input_ids"]
    prefill = schedule["decode_from"] if schedule["decode_from"] is not None else len(ids)
    return list(schedule["chunk_ends"]) + [prefill] + list(range(prefill + 1, len(ids) + 1))


def run(model, ids, ends):
    cache, rows, start = None, [], 0
    with torch.no_grad():
        for end in ends:
            out = model(input_ids=torch.tensor([ids[start:end]]), cache=cache, use_cache=True)
            cache = out.cache
            rows.append(out.logits[0].double())
            start = end
    return torch.cat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["exact", "actq"], default="exact",
                    help="exact: float activations in f64 (the gate's oracle). actq: the declared class, the "
                         "activation-quant expert program simulated on an f32 stream (ink2_oracle.py forward_actq)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    O.cmd_filter(argparse.Namespace(ckpt=args.image, work=args.work))
    dtype = "f64" if args.arm == "exact" else "f32"
    _, model, cfg = O.load_model(args.image, args.work, args.arm, dtype, {})
    schedule = json.load(open(os.path.join(args.image, "schedule.json")))
    ids = schedule["input_ids"]
    logits = run(model, ids, schedule_ends(schedule))
    assert logits.shape == (len(ids), cfg.unpadded_vocab_size), logits.shape
    # self-check: the scheduled run against one uncached call over the whole request
    with torch.no_grad():
        mono = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].double()
    spread = float((mono - logits).abs().max())
    # f64 float activations are smooth: any schedule agrees with the uncached call to rounding. The f32
    # activation-quant arm is reported only, since a quantizer can turn one ulp into a code.
    assert args.arm != "exact" or spread < 1e-9, spread
    bits = logits.to(torch.float32).numpy().view(np.uint32)
    with open(args.out, "w") as f:
        f.write("stream\tposition\ttoken\tlogit_f32_bits\n")
        for position in range(bits.shape[0]):
            for token in range(bits.shape[1]):
                f.write(f"main\t{position}\t{token}\t{int(bits[position, token]):08x}\n")
    print(json.dumps({"arm": args.arm, "dtype": dtype, "positions": len(ids), "vocab": int(bits.shape[1]), "ends": schedule_ends(schedule),
                      "scheduled_vs_uncached_max_abs": spread,
                      "weights_sha256": O.sha256_file(os.path.join(args.image, "model.safetensors")),
                      "pr_head": O.PR_HEAD}))


if __name__ == "__main__":
    main()
