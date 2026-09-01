#!/usr/bin/env python3
"""Capture raw next-token logits from a ModelOpt unified HF checkpoint with vLLM.

Plain ``transformers.AutoModel`` is not a deployment loader for NVIDIA's unified
ModelOpt format.  vLLM is: it consumes ``hf_quant_config.json`` and executes NVFP4
through the ``modelopt_fp4`` backend.  This runner asks vLLM for all-vocabulary
``raw_logits`` for one generated position and serializes the same
``memra-checkpoint-oracle-v1`` format emitted by Memra's ``run-safetensors``.

The external engine is an offline qualification oracle only.  It is never a
Memra serving fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
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
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--parallel-mode",
        choices=("pipeline", "tensor"),
        default="pipeline",
        help="pipeline matches Memra PP whole-layer placement; tensor is a retained diagnostic",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--moe-backend",
        default="auto",
        help="vLLM MoE backend (for example auto or emulation); recorded in the oracle",
    )
    parser.add_argument("--numeric-class", default="ModelOpt-NVFP4-W4A16")
    parser.add_argument("--engine", default="vllm-modelopt-fp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.version_info < (3, 11):
        raise SystemExit(
            "vLLM 0.28.0's Hy3/FlashInfer worker requires Python >=3.11; "
            "the pinned qualification environment uses Python 3.12"
        )
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain unique CUDA ordinals")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1)")
    tokens = [int(part) for part in args.tokens.split(",") if part]
    if not tokens:
        raise SystemExit("--tokens cannot be empty")
    for required in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        if not (args.model / required).is_file():
            raise SystemExit(f"missing {args.model / required}")

    config = json.loads((args.model / "config.json").read_text())
    vocab = int(config["vocab_size"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    plugin_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = plugin_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    capture_path = args.out.with_suffix(args.out.suffix + ".raw-f32")
    capture_path.unlink(missing_ok=True)
    os.environ["HY3_VLLM_RAW_LOGITS_FILE"] = str(capture_path.resolve())
    os.environ["HY3_VLLM_RAW_LOGITS_VOCAB"] = str(vocab)
    # The FlashInfer top-k/top-p JIT is unrelated to this raw-logit oracle and
    # currently mis-detects SM120 when the container toolkit is CUDA 12.8 even
    # though the pinned torch/vLLM wheels carry CUDA 13. Use vLLM's documented
    # native sampler fallback; this does not alter the model forward or raw logits.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import vllm
    from vllm import LLM, SamplingParams

    pipeline_parallel_size = len(devices) if args.parallel_mode == "pipeline" else 1
    tensor_parallel_size = len(devices) if args.parallel_mode == "tensor" else 1
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        quantization="modelopt_fp4",
        dtype="bfloat16",
        trust_remote_code=False,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp",
        skip_tokenizer_init=True,
        max_model_len=max(16, len(tokens) + 1),
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kernel_config={"moe_backend": args.moe_backend},
        logits_processors=["vllm_raw_logits_processor:RawLogitsCapture"],
        enable_prefix_caching=False,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        ignore_eos=True,
        seed=0,
    )
    result = llm.generate(
        [{"prompt_token_ids": tokens}],
        sampling_params=sampling,
        use_tqdm=False,
    )[0]
    completion = result.outputs[0]
    payload = capture_path.read_bytes()
    if len(payload) != vocab * 4:
        raise RuntimeError(f"captured {len(payload)} raw-logit bytes, expected {vocab * 4}")
    values = struct.unpack(f"<{vocab}f", payload)
    scores = dict(enumerate(values))
    if not all(float("-inf") < value < float("inf") for value in scores.values()):
        raise RuntimeError("oracle logits contain non-finite values")

    argmax = max(scores, key=lambda token_id: (scores[token_id], -token_id))
    generated = int(completion.token_ids[0])
    if generated != argmax:
        raise RuntimeError(f"greedy token {generated} differs from raw-logit argmax {argmax}")

    config_sha = sha256_file(args.model / "config.json")
    index_sha = sha256_file(args.model / "model.safetensors.index.json")
    quant_sha = sha256_file(args.model / "hf_quant_config.json")
    lines = [
        "format\tmemra-checkpoint-oracle-v1",
        f"engine\t{args.engine}",
        f"numeric_class\t{args.numeric_class}",
        "score_kind\traw_logits",
        "capture_method\tpassive-v1-logits-processor",
        "vllm_flashinfer_sampler\t0",
        f"vllm_version\t{vllm.__version__}",
        f"parallel_mode\t{args.parallel_mode}",
        f"pipeline_parallel_size\t{pipeline_parallel_size}",
        f"tensor_parallel_size\t{tensor_parallel_size}",
        f"moe_backend\t{args.moe_backend}",
        f"config_sha256\t{config_sha}",
        f"index_sha256\t{index_sha}",
        f"hf_quant_config_sha256\t{quant_sha}",
        f"tokens\t{','.join(map(str, tokens))}",
        f"vocab\t{vocab}",
    ]
    lines.extend(
        f"logit\t{token_id}\t{struct.unpack('<I', struct.pack('<f', scores[token_id]))[0]:08x}"
        for token_id in range(vocab)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "out": str(args.out),
                "tokens": tokens,
                "vocab": vocab,
                "argmax": argmax,
                "max_logit": scores[argmax],
                "vllm_version": vllm.__version__,
                "quantization": "modelopt_fp4",
                "parallel_mode": args.parallel_mode,
                "pipeline_parallel_size": pipeline_parallel_size,
                "tensor_parallel_size": tensor_parallel_size,
                "moe_backend": args.moe_backend,
                "vllm_flashinfer_sampler": 0,
                "capture_method": "passive-v1-logits-processor",
                "config_sha256": config_sha,
                "index_sha256": index_sha,
                "hf_quant_config_sha256": quant_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
