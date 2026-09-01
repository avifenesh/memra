#!/usr/bin/env python3
"""Mint the pinned HY3 BF16 checkpoint into the Memra NVFP4 safetensors profile.

The artifact contract is deliberately narrower than a whole-model PTQ export:

* NVIDIA ModelOpt 0.46.0's ``NVFP4QTensor.quantize`` produces every routed-expert
  weight, E4M3 per-16 scale plane, and F32 macro scale.
* Every routed expert (layers 1..80), including the appended MTP block, is quantized in its
  official safetensors representation. Gate/up share the larger per-tensor ``scale_2`` exactly as
  ModelOpt's fused-MoE recipe requires; down projections retain their own per-tensor scale.
* Attention, routers, shared MLPs, dense layer 0, embeddings, head, norms, biases,
  and non-expert MTP tensors stay byte-identical to the pinned source.
* Source shards are processed independently across all selected GPUs. This bounds
  host/HBM use and avoids Transformers' temporary fused-expert representation, which
  cannot load the 597.6 GB checkpoint inside the mint pod's host-memory cgroup.

This is packaging around NVIDIA's quantizer, not a replacement quantization formula.
The independent Memra-math spot gate below checks nibble order and scale semantics.
Every source tensor must classify exactly; unknown tensors abort the mint.

``MINT_METADATA_ONLY=1`` is the fail-closed pre-publication reseal path for an
already-generated payload. It re-censuses the pinned source and every output header,
renders the deterministic metadata, verifies its locked hashes, and atomically replaces
only ``config.json`` and ``hf_quant_config.json``. Add ``MINT_METADATA_DRY_RUN=1`` to
print the candidate hashes without mutating the artifact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import shutil
import struct
import sys
import time
import traceback
from pathlib import Path


SRC_DIR = Path(os.environ.get("MINT_SRC", "/workspace/hy3-modelopt/source-bf16"))
OUT_DIR = Path(os.environ.get("MINT_OUT", "/workspace/hy3-modelopt/output-experts"))
MODELOPT_REPO = Path(os.environ.get("MINT_MODELOPT_REPO", "/root/modelopt"))
DEVICES = tuple(int(x) for x in os.environ.get("MINT_DEVICES", "0,1,2").split(",") if x)
SPOT_CHECK_EVERY = int(os.environ.get("MINT_SPOT_EVERY", "500"))

PINNED_SOURCE_REVISION = "a960ebc3da325ba167f069f76c41eb62c9280d22"
PINNED_MODELOPT_SHA = "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a"
PINNED_CONFIG_SHA256 = "0c9daab42bff9cce1b6f058b10d7b730f76d583e583e28ad56e92b36373246f0"
PINNED_INDEX_SHA256 = "9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b"
EXPECTED_MODELOPT_VERSION = "0.46.0"
EXPECTED_CONFIG_SHA256 = "3cb16aa29d0046ffddd2f8a4866e4c7511e4018c6fced8dd913d1a788d787af9"
EXPECTED_HF_QUANT_CONFIG_SHA256 = "38e5689cd6847427cc28c26c3cd3ca30568822bf311f479f11d21cf8ab632d2e"
EXPECTED_INDEX_SHA256 = "0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd"

BLOCK = 16
MEMRA_QK = 64
EXPECTED_SOURCE_TENSORS = 47_138
EXPECTED_QUANT_TENSORS = 46_080
EXPECTED_KEEP_TENSORS = 1_058
EXPECTED_OUTPUT_TENSORS = EXPECTED_KEEP_TENSORS + 3 * EXPECTED_QUANT_TENSORS
EXPECTED_SOURCE_PAYLOAD_BYTES = 597_572_342_272
EXPECTED_OUTPUT_PAYLOAD_BYTES = 180_826_481_152
EXPECTED_SOURCE_SHARDS = 99
ROUTED_LAYERS = range(1, 81)
EXPECTED_FUSED_GU_PAIRS = len(ROUTED_LAYERS) * 192
MTP_LAYER = 80

LAYER_PREFIX = "model.layers."
TOPLEVEL_KEEP = {
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
}
COMMON_LAYER_KEEP = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
}
DENSE_LAYER_ZERO_KEEP = {
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}
MOE_KEEP = {
    "mlp.expert_bias",
    "mlp.router.gate.weight",
    "mlp.shared_mlp.gate_proj.weight",
    "mlp.shared_mlp.up_proj.weight",
    "mlp.shared_mlp.down_proj.weight",
}
MTP_KEEP = {
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
}
DTYPE_BYTES = {"BF16": 2, "F32": 4}


class MintError(RuntimeError):
    pass


def die(message: str) -> None:
    raise MintError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_st_header(path: Path) -> tuple[int, dict]:
    """Return safetensors header length and name -> metadata without tensor data."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            die(f"truncated safetensors prefix: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header_len, header


def raw_tensor_sha256(path: Path, header_len: int, meta: dict) -> str:
    """Hash only a tensor's serialized payload, without materializing it."""
    start, end = meta["data_offsets"]
    remaining = end - start
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(8 + header_len + start)
        while remaining:
            chunk = f.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                die(f"truncated tensor payload in {path}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def classify(name: str) -> str:
    if name in TOPLEVEL_KEEP:
        return "keep"
    if not name.startswith(LAYER_PREFIX):
        die(f"unclassifiable tensor (unknown prefix): {name}")
    rest = name[len(LAYER_PREFIX) :]
    layer_text, sep, suffix = rest.partition(".")
    if not sep or not layer_text.isdigit():
        die(f"unclassifiable tensor (missing layer index): {name}")
    layer = int(layer_text)
    if layer < 0 or layer > MTP_LAYER:
        die(f"tensor has out-of-contract layer {layer}: {name}")

    if suffix in COMMON_LAYER_KEEP:
        return "keep"
    if layer == 0 and suffix in DENSE_LAYER_ZERO_KEEP:
        return "keep"
    if layer >= 1 and suffix in MOE_KEEP:
        return "keep"
    if layer == MTP_LAYER and suffix in MTP_KEEP:
        return "keep"

    if suffix.startswith("mlp.experts."):
        parts = suffix.split(".")
        if not (
            len(parts) == 5
            and parts[2].isdigit()
            and 0 <= int(parts[2]) < 192
            and parts[3] in {"gate_proj", "up_proj", "down_proj"}
            and parts[4] == "weight"
        ):
            die(f"malformed expert tensor: {name}")
        return "quant"

    die(f"UNCLASSIFIED tensor (extend deliberately; never default): {name}")
    return "unreachable"


def validate_config() -> None:
    cfg_path = SRC_DIR / "config.json"
    idx_path = SRC_DIR / "model.safetensors.index.json"
    if sha256_file(cfg_path) != PINNED_CONFIG_SHA256:
        die(f"source config hash differs from {PINNED_SOURCE_REVISION}")
    if sha256_file(idx_path) != PINNED_INDEX_SHA256:
        die(f"source index hash differs from {PINNED_SOURCE_REVISION}")
    cfg = json.loads(cfg_path.read_text())
    expected = {
        "model_type": "hy_v3",
        "num_hidden_layers": 80,
        "num_nextn_predict_layers": 1,
        "num_experts": 192,
        "num_experts_per_tok": 8,
        "hidden_size": 4096,
        "moe_intermediate_size": 1536,
        "qk_norm": True,
        "moe_router_use_sigmoid": True,
        "moe_router_enable_expert_bias": True,
    }
    drift = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
    if drift:
        die(f"source semantic config drift: {drift}")


def census() -> tuple[dict, dict, dict]:
    index = json.loads((SRC_DIR / "model.safetensors.index.json").read_text())
    source_map = index["weight_map"]
    names = sorted(source_map)
    shards = sorted(set(source_map.values()))
    if len(names) != EXPECTED_SOURCE_TENSORS:
        die(f"source tensor count {len(names)} != {EXPECTED_SOURCE_TENSORS}")
    if len(shards) != EXPECTED_SOURCE_SHARDS:
        die(f"source shard count {len(shards)} != {EXPECTED_SOURCE_SHARDS}")

    shard_headers = {}
    for shard in shards:
        _n, header = read_st_header(SRC_DIR / shard)
        shard_headers[shard] = header

    plan = {}
    payload = quant_count = keep_count = output_payload = 0
    for name in names:
        shard = source_map[name]
        meta = shard_headers[shard].get(name)
        if meta is None:
            die(f"index tensor missing from shard header: {name} -> {shard}")
        cls = classify(name)
        dtype, shape = meta["dtype"], list(meta["shape"])
        if dtype not in DTYPE_BYTES:
            die(f"unexpected source dtype {dtype}: {name}")
        source_bytes = DTYPE_BYTES[dtype]
        for dim in shape:
            source_bytes *= dim
        payload += source_bytes
        if cls == "quant":
            quant_count += 1
            if dtype != "BF16" or len(shape) != 2:
                die(f"quant tensor must be BF16 2D: {name} {dtype} {shape}")
            out_features, in_features = shape
            if in_features % MEMRA_QK:
                die(f"{name}: in_features {in_features} not divisible by Memra QK={MEMRA_QK}")
            output_payload += out_features * in_features // 2
            output_payload += out_features * in_features // BLOCK
            output_payload += 4
        else:
            keep_count += 1
            output_payload += source_bytes
        plan[name] = (cls, dtype, shape)

    if payload != EXPECTED_SOURCE_PAYLOAD_BYTES:
        die(f"source payload {payload} != {EXPECTED_SOURCE_PAYLOAD_BYTES}")
    if (quant_count, keep_count) != (EXPECTED_QUANT_TENSORS, EXPECTED_KEEP_TENSORS):
        die(
            f"profile census quant={quant_count} keep={keep_count}; expected "
            f"{EXPECTED_QUANT_TENSORS}/{EXPECTED_KEEP_TENSORS}"
        )
    if output_payload != EXPECTED_OUTPUT_PAYLOAD_BYTES:
        die(f"predicted output payload {output_payload} != {EXPECTED_OUTPUT_PAYLOAD_BYTES}")

    # Explicitly prove that all 79 trunk layers plus the MTP layer contribute experts.
    per_layer = {layer: 0 for layer in range(81)}
    for name, (cls, _dtype, _shape) in plan.items():
        if cls == "quant":
            layer = int(name[len(LAYER_PREFIX) :].split(".", 1)[0])
            per_layer[layer] += 1
    expected_per_trunk = 192 * 3
    bad = {
        layer: n
        for layer, n in per_layer.items()
        if n != (expected_per_trunk if layer in ROUTED_LAYERS else 0)
    }
    if bad:
        die(f"routed-expert layer coverage drift: {bad}")

    print(
        f"[census] source={payload:,} bytes, tensors={len(plan)}, shards={len(shards)}, "
        f"quant={quant_count}, keep={keep_count}, output={output_payload:,} bytes",
        flush=True,
    )
    return source_map, plan, shard_headers


# Standard E2M1 codebook. Memra's doubled table and halved UE4M3 decode cancel.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def ue4m3_byte_to_f32(value: int) -> float:
    if value in (0x00, 0x7F):
        return 0.0
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def memra_style_dequant(packed, scale_u8, scale2, out_features: int, in_features: int):
    import torch

    codes = torch.empty(out_features, in_features, dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    values = torch.tensor(E2M1, dtype=torch.float32)[codes.long()]
    lut = torch.tensor([ue4m3_byte_to_f32(x) for x in range(256)], dtype=torch.float32)
    scales = lut[scale_u8.long()].repeat_interleave(BLOCK, dim=1)
    return values * scales * float(scale2)


def spot_check(name, source, qtensor, device_scale, device_scale2, packed, scale, scale2) -> None:
    import torch

    out_features, in_features = source.shape
    scale_u8 = scale.view(torch.uint8)
    if int((scale_u8 & 0x80).ne(0).sum()):
        die(f"{name}: signed E4M3 scale byte; Memra requires unsigned NVFP4 scales")
    if int(scale_u8.eq(0x7F).sum()):
        die(f"{name}: E4M3 NaN scale byte 0x7f")
    consumer = memra_style_dequant(packed, scale_u8, scale2, out_features, in_features)
    producer = qtensor.dequantize(
        torch.float32,
        scale=device_scale,
        double_scale=device_scale2.float(),
        block_sizes={-1: BLOCK},
    ).reshape(out_features, in_features).cpu()
    if not torch.allclose(consumer, producer, rtol=1e-5, atol=1e-30):
        bad = int((~torch.isclose(consumer, producer, rtol=1e-5, atol=1e-30)).sum())
        die(f"{name}: Memra dequant differs from ModelOpt on {bad} elements")
    reference = source.float()
    error = (producer - reference).abs()
    if not torch.isfinite(producer).all() or (float(producer.abs().sum()) == 0 and float(reference.abs().sum()) > 0):
        die(f"{name}: non-finite or all-zero NVFP4 reconstruction")
    print(
        f"[spot] {name}: median_rel={float((error / reference.abs().clamp_min(1e-6)).median()):.6f} "
        f"max_abs={float(error.max()):.6f}",
        flush=True,
    )


def verify_kept_bytes(source_path: Path, output_path: Path, kept_names: list[str]) -> None:
    source_header_len, source_header = read_st_header(source_path)
    output_header_len, output_header = read_st_header(output_path)
    for name in kept_names:
        source_meta = source_header[name]
        output_meta = output_header.get(name)
        if output_meta is None:
            die(f"kept tensor missing after shard write: {name}")
        if source_meta["dtype"] != output_meta["dtype"] or source_meta["shape"] != output_meta["shape"]:
            die(f"kept tensor metadata changed: {name}")
        if raw_tensor_sha256(source_path, source_header_len, source_meta) != raw_tensor_sha256(
            output_path, output_header_len, output_meta
        ):
            die(f"kept tensor bytes changed: {name}")


def mint_worker(worker_id: int, device: int, shards: list[str], source_map: dict, plan: dict) -> dict:
    # The run root may contain a convenience symlink named ``modelopt``. Because
    # multiprocessing spawn puts this script's directory first on sys.path, that
    # symlink otherwise resolves as a namespace package (without __version__) rather
    # than NVIDIA's inner Python package. Bind the verified checkout explicitly.
    sys.path.insert(0, str(MODELOPT_REPO))
    import torch
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.cuda.set_device(device)
    if not torch.cuda.is_available():
        die(f"worker {worker_id}: CUDA unavailable")
    result_map = {}
    payload = quantized = kept = 0
    started = time.time()
    local_quant_ordinal = 0
    for shard_ordinal, shard in enumerate(shards, start=1):
        source_path = SRC_DIR / shard
        output_path = OUT_DIR / shard
        temp_path = OUT_DIR / f".{shard}.worker-{worker_id}.partial"
        output = {}
        kept_names = []
        with safe_open(str(source_path), framework="pt", device="cpu") as source:
            names = sorted(source.keys())
            processed = set()
            for name in names:
                if name in processed:
                    continue
                if source_map.get(name) != shard:
                    die(f"worker {worker_id}: shard/index mismatch for {name}")
                cls, _dtype, shape = plan[name]
                tensor = source.get_tensor(name)
                if cls == "keep":
                    out = tensor.contiguous()
                    output[name] = out
                    kept_names.append(name)
                    payload += out.numel() * out.element_size()
                    result_map[name] = shard
                    kept += 1
                    continue

                quantize_items = [(name, tensor, shape, None)]
                if name.endswith(".gate_proj.weight"):
                    up_name = name.replace(".gate_proj.weight", ".up_proj.weight")
                    if source_map.get(up_name) != shard or plan.get(up_name, (None,))[0] != "quant":
                        die(f"{name}: fused gate/up pair is absent or crosses a shard")
                    up_tensor = source.get_tensor(up_name)
                    up_shape = plan[up_name][2]
                    gate_cuda = tensor.to(f"cuda:{device}")
                    up_cuda = up_tensor.to(f"cuda:{device}")
                    gate_scale2 = NVFP4QTensor.get_weights_scaling_factor_2(gate_cuda)
                    up_scale2 = NVFP4QTensor.get_weights_scaling_factor_2(up_cuda)
                    shared_scale2 = torch.maximum(gate_scale2.reshape(()), up_scale2.reshape(()))
                    quantize_items = [
                        (name, tensor, shape, (gate_cuda, shared_scale2)),
                        (up_name, up_tensor, up_shape, (up_cuda, shared_scale2)),
                    ]
                elif name.endswith(".up_proj.weight"):
                    die(f"{name}: reached up projection before its fused gate/up pair")

                for quant_name, quant_tensor, quant_shape, prepared in quantize_items:
                    out_features, in_features = quant_shape
                    stem = quant_name[: -len(".weight")]
                    if prepared is None:
                        source_cuda = quant_tensor.to(f"cuda:{device}")
                        scale2_override = None
                    else:
                        source_cuda, scale2_override = prepared
                    # Keep try_tensorrt at its default False: that path emits a CUTLASS
                    # scale swizzle, while this artifact contract stores ModelOpt's HF layout.
                    qtensor, device_scale, device_scale2 = NVFP4QTensor.quantize(
                        source_cuda,
                        BLOCK,
                        weights_scaling_factor_2=scale2_override,
                    )
                    packed = qtensor._quantized_data
                    if packed.dtype != torch.uint8 or list(packed.shape) != [
                        out_features,
                        in_features // 2,
                    ]:
                        die(
                            f"{quant_name}: invalid packed output {packed.dtype} "
                            f"{list(packed.shape)}"
                        )
                    if device_scale.dtype != torch.float8_e4m3fn or list(device_scale.shape) != [
                        out_features,
                        in_features // BLOCK,
                    ]:
                        die(
                            f"{quant_name}: invalid scale output {device_scale.dtype} "
                            f"{list(device_scale.shape)}"
                        )
                    if device_scale2.numel() != 1:
                        die(f"{quant_name}: weight_scale_2 must be scalar")
                    packed_cpu = packed.cpu()
                    scale_cpu = device_scale.cpu()
                    scale2_cpu = device_scale2.float().reshape(()).cpu()
                    if local_quant_ordinal % SPOT_CHECK_EVERY == 0:
                        spot_check(
                            quant_name,
                            quant_tensor,
                            qtensor,
                            device_scale,
                            device_scale2,
                            packed_cpu,
                            scale_cpu,
                            float(scale2_cpu),
                        )
                    triples = {
                        f"{stem}.weight": packed_cpu,
                        f"{stem}.weight_scale": scale_cpu,
                        f"{stem}.weight_scale_2": scale2_cpu,
                    }
                    for output_name, out in triples.items():
                        if output_name in output:
                            die(f"worker {worker_id}: duplicate output tensor {output_name}")
                        output[output_name] = out
                        result_map[output_name] = shard
                        payload += out.numel() * out.element_size()
                    processed.add(quant_name)
                    quantized += 1
                    local_quant_ordinal += 1
                    del source_cuda, qtensor, packed, device_scale, device_scale2

        save_file(output, str(temp_path))
        temp_path.replace(output_path)
        verify_kept_bytes(source_path, output_path, kept_names)
        del output
        print(
            f"[worker {worker_id} cuda:{device}] shard {shard_ordinal}/{len(shards)} {shard}: "
            f"quant={quantized} keep={kept} payload={payload / 1e9:.2f}GB "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    return {"worker": worker_id, "weight_map": result_map, "payload": payload, "quant": quantized, "keep": kept}


def exclude_module_list(plan: dict) -> list[str]:
    modules = set()
    for name, (cls, _dtype, shape) in plan.items():
        if cls == "keep" and name.endswith(".weight") and len(shape) >= 2:
            modules.add(name[: -len(".weight")])
    # HYV3FeedForward deliberately reuses the MoE parent's runtime prefix for the
    # shared MLP even though its checkpoint tensors live below ``shared_mlp``.
    # Carry those fused runtime aliases or deployment loaders allocate the preserved
    # BF16 shared projections as packed FP4 parameters.
    runtime_aliases = set()
    for module in modules:
        if module.endswith((".mlp.shared_mlp.gate_proj", ".mlp.shared_mlp.up_proj")):
            runtime_aliases.add(module.rsplit(".", 2)[0] + ".gate_up_proj")
        elif module.endswith(".mlp.shared_mlp.down_proj"):
            runtime_aliases.add(module.replace(".mlp.shared_mlp.down_proj", ".mlp.down_proj"))
        elif module.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            runtime_aliases.add(module.rsplit(".", 1)[0] + ".gate_up_proj")
    modules.update(runtime_aliases)
    # HYV3's deployment class constructs its inner transformer with prefixes like
    # ``layers.N...`` while the checkpoint names are ``model.layers.N...``. NVIDIA's
    # vLLM ModelOpt loader applies exclusions at construction time, before its weight
    # loader strips the outer ``model.`` component. Carry both exact aliases so every
    # preserved BF16 linear stays unquantized in either namespace; unmatched aliases
    # are harmless to consumers that keep the checkpoint prefix.
    modules.update(
        module.removeprefix("model.")
        for module in tuple(modules)
        if module.startswith("model.")
    )
    return sorted(modules)


def render_configs(plan: dict, modelopt_version: str) -> tuple[str, str, list[str]]:
    source_cfg = json.loads((SRC_DIR / "config.json").read_text())
    excluded = exclude_module_list(plan)
    producer = {"name": "modelopt", "version": modelopt_version}
    source_cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": BLOCK},
                "targets": ["Linear"],
            }
        },
        "ignore": excluded,
        "quant_algo": "W4A16_NVFP4",
        "producer": producer,
        "quant_method": "modelopt",
    }
    hf_quant_config = {
        "producer": producer,
        "quantization": {
            "quant_algo": "W4A16_NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": BLOCK,
            "exclude_modules": excluded,
        },
    }
    return (
        json.dumps(source_cfg, indent=4) + "\n",
        json.dumps(hf_quant_config, indent=4) + "\n",
        excluded,
    )


def write_configs(plan: dict, modelopt_version: str) -> None:
    config_text, hf_quant_text, excluded = render_configs(plan, modelopt_version)
    (OUT_DIR / "config.json").write_text(config_text)
    (OUT_DIR / "hf_quant_config.json").write_text(hf_quant_text)

    copied = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_file() or path.suffix == ".safetensors" or path.name in {
            "model.safetensors.index.json",
            "config.json",
        }:
            continue
        if path.suffix in {".json", ".jinja", ".txt", ".model", ".py", ".md"}:
            shutil.copy2(path, OUT_DIR / path.name)
            copied.append(path.name)
    print(f"[config] excluded modules={len(excluded)}; copied sidecars={','.join(copied)}", flush=True)


def verify_output(plan: dict, output_map: dict, payload: int) -> None:
    if len(output_map) != EXPECTED_OUTPUT_TENSORS:
        die(f"output map tensor count {len(output_map)} != {EXPECTED_OUTPUT_TENSORS}")
    if payload != EXPECTED_OUTPUT_PAYLOAD_BYTES:
        die(f"worker payload {payload} != {EXPECTED_OUTPUT_PAYLOAD_BYTES}")
    headers = {}
    for shard in sorted(set(output_map.values())):
        _n, header = read_st_header(OUT_DIR / shard)
        overlap = set(headers) & set(header)
        if overlap:
            die(f"duplicate tensors across output shards: {sorted(overlap)[:3]}")
        headers.update(header)
    if set(headers) != set(output_map):
        die("output index/header tensor sets differ")

    quantized = kept = 0
    for name, (cls, dtype, shape) in plan.items():
        if cls == "keep":
            meta = headers.get(name)
            if meta is None or meta["dtype"] != dtype or meta["shape"] != shape:
                die(f"kept tensor metadata mismatch: {name}")
            stem = name[: -len(".weight")] if name.endswith(".weight") else name
            if f"{stem}.weight_scale" in headers or f"{stem}.weight_scale_2" in headers:
                die(f"kept tensor has a stray NVFP4 scale: {name}")
            kept += 1
            continue
        stem = name[: -len(".weight")]
        out_features, in_features = shape
        expected = {
            f"{stem}.weight": ("U8", [out_features, in_features // 2]),
            f"{stem}.weight_scale": ("F8_E4M3", [out_features, in_features // BLOCK]),
            f"{stem}.weight_scale_2": ("F32", []),
        }
        for output_name, (expected_dtype, expected_shape) in expected.items():
            meta = headers.get(output_name)
            if meta is None or meta["dtype"] != expected_dtype or meta["shape"] != expected_shape:
                die(f"invalid NVFP4 member {output_name}: {meta}")
        quantized += 1
    if (quantized, kept) != (EXPECTED_QUANT_TENSORS, EXPECTED_KEEP_TENSORS):
        die(f"verification counts differ: quant={quantized}, keep={kept}")

    import torch
    from safetensors import safe_open

    pairs_by_shard = {}
    for layer in ROUTED_LAYERS:
        for expert in range(192):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate = f"{prefix}.gate_proj.weight_scale_2"
            up = f"{prefix}.up_proj.weight_scale_2"
            gate_shard = output_map.get(gate)
            up_shard = output_map.get(up)
            if gate_shard is None or gate_shard != up_shard:
                die(f"fused gate/up scale pair is missing or crosses shards: {prefix}")
            pairs_by_shard.setdefault(gate_shard, []).append((gate, up))
    pair_count = 0
    for shard, pairs in sorted(pairs_by_shard.items()):
        with safe_open(str(OUT_DIR / shard), framework="pt", device="cpu") as handle:
            for gate, up in pairs:
                gate_bits = int(handle.get_tensor(gate).view(torch.int32).item())
                up_bits = int(handle.get_tensor(up).view(torch.int32).item())
                if gate_bits != up_bits:
                    die(f"fused gate/up weight_scale_2 differs: {gate} != {up}")
                pair_count += 1
    if pair_count != EXPECTED_FUSED_GU_PAIRS:
        die(f"verified {pair_count} fused gate/up pairs, expected {EXPECTED_FUSED_GU_PAIRS}")
    print(f"[verify] fused gate/up shared weight_scale_2 pairs={pair_count}", flush=True)

    for path, expected in [
        (OUT_DIR / "config.json", EXPECTED_CONFIG_SHA256),
        (OUT_DIR / "hf_quant_config.json", EXPECTED_HF_QUANT_CONFIG_SHA256),
        (OUT_DIR / "model.safetensors.index.json", EXPECTED_INDEX_SHA256),
    ]:
        actual = sha256_file(path)
        if actual != expected:
            die(f"generated metadata hash drift for {path.name}: {actual} != {expected}")


def preflight(metadata_only: bool = False) -> tuple[str, dict, dict]:
    if not DEVICES:
        die("MINT_DEVICES is empty")
    if not (SRC_DIR / "model.safetensors.index.json").is_file() or not (SRC_DIR / "config.json").is_file():
        die(f"pinned source is incomplete: {SRC_DIR}")
    if metadata_only:
        if not (OUT_DIR / "model.safetensors.index.json").is_file():
            die(f"metadata-only reseal requires an existing artifact: {OUT_DIR}")
    elif OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        die(f"refusing to overwrite non-empty output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("nvidia-modelopt")
    if version != EXPECTED_MODELOPT_VERSION:
        die(f"nvidia-modelopt {version} != pinned {EXPECTED_MODELOPT_VERSION}")
    validate_config()
    source_map, plan, _headers = census()
    print(
        f"[preflight] source={SRC_DIR} output={OUT_DIR} modelopt={version} "
        f"modelopt_sha={PINNED_MODELOPT_SHA} devices={DEVICES}",
        flush=True,
    )
    return version, source_map, plan


def main() -> None:
    metadata_only = os.environ.get("MINT_METADATA_ONLY") == "1"
    version, source_map, plan = preflight(metadata_only=metadata_only)
    if metadata_only:
        index = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())
        output_map = index.get("weight_map")
        payload = (index.get("metadata") or {}).get("total_size")
        if not isinstance(output_map, dict) or not isinstance(payload, int):
            die("metadata-only reseal found a malformed output index")
        config_text, hf_quant_text, _excluded = render_configs(plan, version)
        candidate_config = sha256_text(config_text)
        candidate_hf_quant = sha256_text(hf_quant_text)
        print(
            f"[metadata-candidate] config={candidate_config} "
            f"hf_quant_config={candidate_hf_quant}",
            flush=True,
        )
        if os.environ.get("MINT_METADATA_DRY_RUN") == "1":
            return
        if candidate_config != EXPECTED_CONFIG_SHA256:
            die(
                f"candidate config hash {candidate_config} != expected {EXPECTED_CONFIG_SHA256}"
            )
        if candidate_hf_quant != EXPECTED_HF_QUANT_CONFIG_SHA256:
            die(
                "candidate hf_quant_config hash "
                f"{candidate_hf_quant} != expected {EXPECTED_HF_QUANT_CONFIG_SHA256}"
            )
        config_temp = OUT_DIR / ".config.json.metadata-partial"
        hf_quant_temp = OUT_DIR / ".hf_quant_config.json.metadata-partial"
        if config_temp.exists() or hf_quant_temp.exists():
            die("metadata-only reseal found a stale partial file")
        config_temp.write_text(config_text)
        hf_quant_temp.write_text(hf_quant_text)
        config_temp.replace(OUT_DIR / "config.json")
        hf_quant_temp.replace(OUT_DIR / "hf_quant_config.json")
        verify_output(plan, output_map, payload)
        print(
            f"METADATA-RESEAL-DONE tensors={len(output_map)} payload={payload} "
            f"config={candidate_config} hf_quant_config={candidate_hf_quant}",
            flush=True,
        )
        return
    shards = sorted(set(source_map.values()))
    assignments = [shards[i:: len(DEVICES)] for i in range(len(DEVICES))]
    context = mp.get_context("spawn")
    args = [
        (worker_id, device, assignment, source_map, plan)
        for worker_id, (device, assignment) in enumerate(zip(DEVICES, assignments))
    ]
    with context.Pool(processes=len(args)) as pool:
        results = pool.starmap(mint_worker, args)

    output_map = {}
    payload = quantized = kept = 0
    for result in results:
        overlap = set(output_map) & set(result["weight_map"])
        if overlap:
            die(f"workers emitted duplicate tensors: {sorted(overlap)[:3]}")
        output_map.update(result["weight_map"])
        payload += result["payload"]
        quantized += result["quant"]
        kept += result["keep"]
    if (quantized, kept) != (EXPECTED_QUANT_TENSORS, EXPECTED_KEEP_TENSORS):
        die(f"worker totals differ: quant={quantized} keep={kept}")

    index = {"metadata": {"total_size": payload}, "weight_map": dict(sorted(output_map.items()))}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    write_configs(plan, version)
    verify_output(plan, output_map, payload)
    print(
        f"MINT-DONE tensors={len(output_map)} payload={payload} shards={len(set(output_map.values()))} "
        f"quant={quantized} keep={kept}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - one loud task boundary
        traceback.print_exc()
        print(f"MINT-FAILED: {error}", file=sys.stderr)
        sys.exit(1)
