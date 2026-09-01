#!/usr/bin/env python3
"""Compare per-source-tensor and Transformers-fused gate/up NVFP4 macro scales."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modelopt-repo", type=Path, default=Path("/root/modelopt"))
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.modelopt_repo))

    import torch
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    from safetensors import safe_open

    index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]

    def load(name: str):
        with safe_open(str(args.source / index[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to(args.device)

    def qdq(weight):
        qtensor, scale, scale2 = NVFP4QTensor.quantize(weight, 16)
        restored = qtensor.dequantize(
            torch.float32,
            scale=scale,
            double_scale=scale2.float(),
            block_sizes={-1: 16},
        ).reshape(weight.shape)
        return restored, float(scale2.float())

    rows = []
    totals = {"separate": [0.0, 0], "fused": [0.0, 0]}
    better = equal = worse = 0
    for layer in range(1, 80):
        prefix = f"model.layers.{layer}.mlp.experts.{args.expert}"
        names = [f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"]
        weights = [load(name) for name in names]
        separate = [qdq(weight) for weight in weights]
        fused_weight = torch.cat(weights, dim=0)
        fused_restored, fused_scale2 = qdq(fused_weight)
        fused_parts = fused_restored.split([weight.shape[0] for weight in weights], dim=0)
        layer_sep = layer_fused = 0.0
        for name, weight, (separate_restored, separate_scale2), fused_part in zip(
            names, weights, separate, fused_parts
        ):
            sep_mse = float((separate_restored - weight.float()).square().mean())
            fused_mse = float((fused_part - weight.float()).square().mean())
            layer_sep += sep_mse
            layer_fused += fused_mse
            elements = weight.numel()
            totals["separate"][0] += sep_mse * elements
            totals["separate"][1] += elements
            totals["fused"][0] += fused_mse * elements
            totals["fused"][1] += elements
            rows.append((layer, name.rsplit(".", 2)[-2], separate_scale2, fused_scale2, sep_mse, fused_mse))
        if layer_sep < layer_fused:
            better += 1
        elif layer_sep == layer_fused:
            equal += 1
        else:
            worse += 1
        print(f"layer={layer} separate_mse={layer_sep:.12g} fused_mse={layer_fused:.12g}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write("layer\tprojection\tseparate_scale2\tfused_scale2\tseparate_mse\tfused_mse\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    separate_mse = totals["separate"][0] / totals["separate"][1]
    fused_mse = totals["fused"][0] / totals["fused"][1]
    summary = {
        "layers": 79,
        "expert": args.expert,
        "separate_mse": separate_mse,
        "fused_mse": fused_mse,
        "relative_mse_change": separate_mse / fused_mse - 1.0,
        "separate_better_layers": better,
        "equal_layers": equal,
        "separate_worse_layers": worse,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
