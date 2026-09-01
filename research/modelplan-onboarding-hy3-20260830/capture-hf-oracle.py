#!/usr/bin/env python3
"""Capture deterministic last-token logits from a plain HF-compatible HY3 artifact.

Unlike the generic onboarding template, this runner never expands the 597.6 GB BF16
checkpoint to FP32. It streams with Accelerate onto the explicitly selected GPUs and
writes a memra-checkpoint-oracle-v1 file whose numeric class is supplied by the caller.
That keeps BF16 and plain-HF official-FP8 controls honest and distinguishable.

NVIDIA unified ModelOpt exports are not plain Transformers checkpoints.  This script
refuses them before allocation; use ``capture-vllm-oracle.py`` with vLLM's
``modelopt_fp4`` deployment loader instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", default="1,2,3,4")
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--max-memory", default="250GiB")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32", "auto"), default="bf16")
    parser.add_argument("--numeric-class", required=True)
    parser.add_argument("--engine", default="hf-transformers")
    parser.add_argument("--modelopt-repo", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain unique CUDA ordinals")
    tokens = [int(part) for part in args.tokens.split(",") if part]
    if not tokens:
        raise SystemExit("--tokens cannot be empty")
    for required in ("config.json", "model.safetensors.index.json"):
        if not (args.model / required).is_file():
            raise SystemExit(f"missing {args.model / required}")
    config = json.loads((args.model / "config.json").read_text())
    if (config.get("quantization_config") or {}).get("quant_method") == "modelopt":
        raise SystemExit(
            "unified ModelOpt checkpoints require capture-vllm-oracle.py; "
            "plain Transformers is not a ModelOpt deployment loader"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.modelopt_repo:
        sys.path.insert(0, str(args.modelopt_repo))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if args.modelopt_repo:
        import modelopt.torch.quantization  # noqa: F401 - registers the HF quantizer

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }[args.dtype]
    max_memory = {index: args.max_memory for index in range(len(devices))}
    device_map = {"": 0} if len(devices) == 1 else "balanced"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([tokens], device=input_device)).logits[0, -1].float().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError("oracle logits contain non-finite values")

    config_sha = sha256_file(args.model / "config.json")
    index_sha = sha256_file(args.model / "model.safetensors.index.json")
    lines = [
        "format\tmemra-checkpoint-oracle-v1",
        f"engine\t{args.engine}",
        f"numeric_class\t{args.numeric_class}",
        f"transformers_version\t{transformers.__version__}",
        f"torch_version\t{torch.__version__}",
        f"config_sha256\t{config_sha}",
        f"index_sha256\t{index_sha}",
        f"tokens\t{','.join(map(str, tokens))}",
        f"vocab\t{logits.numel()}",
    ]
    # Serialize through IEEE-754 explicitly so the oracle is round-trip exact.
    import struct

    lines.extend(
        f"logit\t{index}\t{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"
        for index, value in enumerate(logits.tolist())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    placement = getattr(model, "hf_device_map", {})
    print(
        json.dumps(
            {
                "status": "passed",
                "out": str(args.out),
                "tokens": tokens,
                "vocab": logits.numel(),
                "argmax": int(logits.argmax()),
                "max_logit": float(logits.max()),
                "device_map_entries": len(placement),
                "config_sha256": config_sha,
                "index_sha256": index_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
