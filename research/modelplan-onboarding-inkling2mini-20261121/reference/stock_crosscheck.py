#!/usr/bin/env python3
"""How far the tiny gate's oracle is from the unmodified PR program.

The pinned oracle (tiny_oracle.py, arm "exact") is the PR's trunk, cache, masks and loader ops in f64 with the expert
module REPLACED by ink2_oracle.py's own (stored bytes, each routed expert dequantized exactly, f64 GEMM). This
script runs the PR program with nothing replaced (its own loader, Interleave, dequantizer op and expert module
with the eager expert loop, ink2_oracle.py --stock) on the same checkpoint image in f64, f32 and bf16, and reports each one's distance from the
pinned TSV. f64 and f32 differ from the oracle only by the PR's f32 dequant arithmetic; bf16 also rounds the
dequantized weights to BF16, which is another weight.

  python -I stock_crosscheck.py --image out/tiny-checkpoint --pinned inkling2.tsv > stock-crosscheck.json
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ink2_oracle as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pinned", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = open(args.pinned).read().splitlines()[1:]
    ids = json.load(open(os.path.join(args.image, "schedule.json")))["input_ids"]
    pinned = np.array([int(r.rsplit("\t", 1)[1], 16) for r in rows], dtype=np.uint32).view(np.float32)
    pinned = pinned.reshape(len(ids), -1).astype(np.float64)
    out = {"pinned_sha256": O.sha256_file(args.pinned), "weights_sha256": O.sha256_file(os.path.join(args.image, "model.safetensors")),
           "pr_head": O.PR_HEAD, "torch": torch.__version__, "positions": len(ids), "runs": {}}
    for dtype in ("f64", "f32", "bf16"):
        with contextlib.redirect_stdout(sys.stderr):
            _, model, _ = O.load_model(args.image, None, None, dtype, {}, stock=True)
        # The PR's eager expert loop: its grouped-GEMM path does not take f64.
        model.config._experts_implementation = "eager"
        for module in model.modules():
            if hasattr(module, "config") and module.config is not model.config:
                module.config._experts_implementation = "eager"
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].double().numpy()
        d = np.abs(logits - pinned)
        out["runs"]["stock_" + dtype] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                         "argmax_equal_positions": int((logits.argmax(1) == pinned.argmax(1)).sum())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
