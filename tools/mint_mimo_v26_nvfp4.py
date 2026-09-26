#!/usr/bin/env python3
"""Repack MiMo-V2.6's MXFP4 experts into an audited ModelOpt NVFP4 checkpoint.

Run on the qualification machine. This preserves expert E2M1 payloads and checks
their decoded values; it does not claim that FP4 activation execution is lossless.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time
import urllib.request

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
SOURCE_REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
MODELOPT_REVISION = "051d6adb204f10cd3e78d0f824f31a5a01d54831"
EXPERT = re.compile(
    r"^(model\.layers\.(\d+)\.mlp\.experts\.(\d+))\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, "time": time.time(), **values}), flush=True)


def nonzero_blocks(packed, scales):
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("source experts must contain packed U8 weights and E8M0 scales")
    if packed.ndim != 2 or scales.shape != (
        packed.shape[0], packed.shape[1] // 16
    ) or packed.shape[1] % 16:
        raise ValueError("source expert does not have one E8M0 scale per 32 weights")
    # Either signed zero is zero; both nibbles' magnitude bits are in 0x77.
    return (packed.reshape(*scales.shape, 16).bitwise_and(0x77) != 0).any(-1)


def shared_global_scale(projections):
    """Use NVIDIA's exponent-window selection, sharing gate/up decode scales."""
    from modelopt.torch.quantization.utils.numeric_utils import (
        mxfp4_to_nvfp4_global_amax,
    )

    live = []
    for packed, scales in projections:
        if torch.any(scales == 255):
            raise ValueError("source contains an invalid E8M0 NaN scale")
        live.append(scales[nonzero_blocks(packed, scales)])
    values = torch.cat(live)
    if not values.numel():
        return 1.0
    # E8M0 code zero denotes 2^-127. The helper's GPT-OSS bookkeeping treats
    # it as an empty block; handle an all-zero-exponent, nonzero payload here.
    maximum = int(values.max())
    exponent = maximum - 127 - 8
    if maximum:
        _, info = mxfp4_to_nvfp4_global_amax(values)
        if int(info["m"]) != exponent:
            raise ValueError("ModelOpt exponent-window contract changed")
    scale = torch.tensor(math.ldexp(1.0, exponent), dtype=torch.float32)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("NVFP4 global scale is not representable in FP32")
    return float(scale)


def cast_scales(packed, scales, global_scale):
    """Construct group-16 E4M3 scales, refusing any changed nonzero weight."""
    live = nonzero_blocks(packed, scales)
    if torch.any(scales == 255):
        raise ValueError("source contains an invalid E8M0 NaN scale")
    source = torch.exp2(scales.to(torch.float64) - 127)
    local = source / global_scale
    # Unused scales may be normalized without changing their zero payload.
    local = torch.where(live, local, torch.ones_like(local))
    target = local.to(torch.float8_e4m3fn)
    reconstructed = target.to(torch.float64) * global_scale
    mismatch = live & (source != reconstructed)
    if torch.any(mismatch):
        raise ValueError(
            f"lossless NVFP4 conversion failed for {int(mismatch.sum())} "
            "nonzero MXFP4 blocks; retain the source and select another recipe"
        )
    if not torch.isfinite(target.to(torch.float32)).all():
        raise ValueError("non-finite NVFP4 block scale")
    return target.repeat_interleave(2, dim=-1).contiguous(), {
        "mxfp4_blocks": scales.numel(),
        "nonzero_blocks": int(live.sum()),
        "minimum_source_exponent": int(scales[live].min()) - 127 if live.any() else None,
        "maximum_source_exponent": int(scales[live].max()) - 127 if live.any() else None,
        "weight_scale_2": global_scale,
        "reconstruction": "exact",
    }


def expert_inventory(index):
    groups = defaultdict(dict)
    for key, shard in index["weight_map"].items():
        match = EXPERT.fullmatch(key)
        if match:
            prefix, layer, expert, projection = match.groups()
            groups[prefix][projection] = (key, shard)
            if not 1 <= int(layer) <= 47 or not 0 <= int(expert) < 256:
                raise ValueError("expert identity is outside the MiMo-V2.6 topology")
    expected = {
        f"model.layers.{layer}.mlp.experts.{expert}"
        for layer in range(1, 48) for expert in range(256)
    }
    if set(groups) != expected:
        raise ValueError("the complete 47-layer, 256-expert bank is required")
    for prefix, projections in groups.items():
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"incomplete expert: {prefix}")
        shards = {shard for _, shard in projections.values()}
        if len(shards) != 1:
            raise ValueError(f"expert projections cross shards: {prefix}")
        for weight, shard in projections.values():
            if index["weight_map"].get(weight.removesuffix(".weight") + ".weight_scale") != shard:
                raise ValueError(f"missing or misplaced MXFP4 scale: {weight}")
    return groups


def source_manifest(source):
    url = (
        f"https://huggingface.co/api/models/{MODEL_ID}/revision/"
        f"{SOURCE_REVISION}?blobs=true"
    )
    with urllib.request.urlopen(url, timeout=45) as response:
        value = json.load(response)
    if value["sha"] != SOURCE_REVISION:
        raise ValueError("source revision mismatch")
    files = {}
    for row in value["siblings"]:
        name = row["rfilename"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe checkpoint filename")
        files[name] = row
    return files


def verify_source_file(source, name, files):
    path = source / name
    row = files[name]
    if not path.is_file():
        raise ValueError(f"source download is incomplete: {name}")
    if path.stat().st_size != row["size"]:
        raise ValueError(f"source size mismatch: {name}")
    digest = sha256(path)
    expected = row.get("lfs", {}).get("sha256")
    if expected and digest != expected:
        raise ValueError(f"source SHA256 mismatch: {name}")
    if not expected:
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                blob.update(chunk)
        if blob.hexdigest() != row["blobId"]:
            raise ValueError(f"source Git blob mismatch: {name}")
    return digest


def verify_shard(source, target):
    """Reopen serialized output and independently verify its full tensor map."""
    with safe_open(source, framework="pt", device="cpu") as before, safe_open(
        target, framework="pt", device="cpu"
    ) as after:
        expected = set(before.keys())
        projections = {key.removesuffix(".weight") for key in before.keys() if EXPERT.fullmatch(key)}
        expected.update(prefix + suffix for prefix in projections for suffix in [
            ".weight_scale_2", ".input_scale"
        ])
        if set(after.keys()) != expected:
            raise ValueError(f"output tensor census mismatch: {target.name}")
        for key in before.keys():
            original = before.get_tensor(key)
            minted = after.get_tensor(key)
            if key.endswith(".weight_scale") and key.removesuffix(".weight_scale") in projections:
                prefix = key.removesuffix(".weight_scale")
                packed = before.get_tensor(prefix + ".weight")
                live = nonzero_blocks(packed, original).repeat_interleave(2, -1)
                expected_scales = torch.exp2(original.to(torch.float64) - 127).repeat_interleave(2, -1)
                global_scale = after.get_tensor(prefix + ".weight_scale_2")
                if (
                    global_scale.numel() != 1
                    or global_scale.dtype != torch.float32
                    or not torch.isfinite(global_scale).all()
                    or global_scale.item() <= 0
                ):
                    raise ValueError(f"invalid global scale: {prefix}")
                if minted.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"invalid NVFP4 scale dtype: {prefix}")
                actual = minted.to(torch.float64) * float(global_scale.item())
                if expected_scales.shape != actual.shape or torch.any(live & (expected_scales != actual)):
                    raise ValueError(f"serialized scale reconstruction mismatch: {prefix}")
                activation = after.get_tensor(prefix + ".input_scale")
                if (
                    activation.numel() != 1 or activation.dtype != torch.float32
                    or not torch.isfinite(activation).all() or activation.item() <= 0
                ):
                    raise ValueError(f"invalid activation scale: {prefix}")
            elif original.dtype != minted.dtype or original.shape != minted.shape or not torch.equal(
                original.reshape(-1).view(torch.uint8), minted.reshape(-1).view(torch.uint8)
            ):
                raise ValueError(f"changed source tensor: {key}")
    return len(projections)


def mint(args):
    torch.set_num_threads(args.threads)
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate checkpoint directories")
    source_config = json.loads((source / "config.json").read_text())
    if (
        source_config.get("model_type") != "mimo_v2"
        or source_config.get("num_hidden_layers") != 48
        or source_config.get("quantization_config", {}).get("store_dtype") != "mxfp4"
    ):
        raise ValueError("expected the pinned MiMo-V2.6 MXFP4 checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    groups = expert_inventory(index)
    files = source_manifest(source)
    for name in ("config.json", "model.safetensors.index.json"):
        verify_source_file(source, name, files)
    distribution = importlib.metadata.distribution("nvidia-modelopt")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != MODELOPT_REVISION:
        raise ValueError(f"install ModelOpt from the required revision {MODELOPT_REVISION}")
    scales = json.loads(args.activation_scales.read_text()) if args.activation_scales else {}
    if scales and set(scales) != {f"model.layers.{i}.mlp.experts" for i in range(1, 48)}:
        raise ValueError("activation policy must cover exactly the 47 MoE layers")
    policy = {
        "source_model": MODEL_ID, "source_revision": SOURCE_REVISION,
        "modelopt_revision": MODELOPT_REVISION, "modelopt_version": distribution.version,
        "tool_sha256": sha256(Path(__file__)),
        "default_input_scale": args.input_scale, "activation_scales": scales,
        "weight_recipe": "ModelOpt closed-form exponent window; shared gate/up globals",
        "activation_recipe": "provided calibration" if scales else "constant-scale seed; unqualified",
    }
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "mint-policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("output belongs to another mint policy; use a new output directory")
    write_json(policy_path, policy)
    write_json(output / "DRAFT-NOT-QUALIFIED.json", {"status": "unqualified", "policy_sha256": policy_hash})
    shards = sorted(set(index["weight_map"].values()))
    if args.shard and args.shard not in shards:
        raise ValueError("requested shard is not in the source index")
    records = {}
    for shard in ([args.shard] if args.shard else shards):
        receipt = output / "mint-receipts" / (shard + ".json")
        destination = output / shard
        if receipt.exists() and destination.exists():
            record = json.loads(receipt.read_text())
            if record["policy_sha256"] != policy_hash or sha256(destination) != record["sha256"]:
                raise ValueError(f"invalid resumed output: {shard}")
            records[shard] = record
            emit("shard_resumed", shard=shard)
            continue
        source_hash = verify_source_file(source, shard, files)
        emit("shard_start", shard=shard)
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            proofs = []
            for prefix, projections in groups.items():
                if next(iter(projections.values()))[1] != shard:
                    continue
                matrices = {
                    projection: (
                        tensors[weight],
                        tensors[weight.removesuffix(".weight") + ".weight_scale"],
                    )
                    for projection, (weight, _) in projections.items()
                }
                globals_by_projection = {
                    "gate_proj": shared_global_scale([matrices["gate_proj"], matrices["up_proj"]]),
                    "down_proj": shared_global_scale([matrices["down_proj"]]),
                }
                globals_by_projection["up_proj"] = globals_by_projection["gate_proj"]
                layer = prefix.rsplit(".", 1)[0]
                activation = scales.get(layer, {
                    "input_scale": args.input_scale, "down_input_scale": args.input_scale
                })
                if set(activation) != {"input_scale", "down_input_scale"}:
                    raise ValueError(f"invalid activation policy: {layer}")
                for projection, (packed, block_scales) in matrices.items():
                    key = prefix + "." + projection
                    global_scale = globals_by_projection[projection]
                    converted, proof = cast_scales(packed, block_scales, global_scale)
                    input_scale = float(activation["down_input_scale" if projection == "down_proj" else "input_scale"])
                    if not math.isfinite(input_scale) or input_scale <= 0:
                        raise ValueError(f"invalid input scale: {key}")
                    tensors[key + ".weight_scale"] = converted
                    tensors[key + ".weight_scale_2"] = torch.tensor([global_scale], dtype=torch.float32)
                    tensors[key + ".input_scale"] = torch.tensor([input_scale], dtype=torch.float32)
                    proofs.append({"projection": key, **proof})
            temporary = destination.with_suffix(".safetensors.writing")
            save_file(tensors, temporary, metadata={"format": "pt"})
            total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            del tensors
        count = verify_shard(source / shard, temporary)
        temporary.replace(destination)
        record = {
            "policy_sha256": policy_hash, "source_sha256": source_hash,
            "sha256": sha256(destination), "file_bytes": destination.stat().st_size,
            "tensor_bytes": total_bytes, "expert_projection_count": count,
            "reconstruction": "exact", "projections": proofs,
        }
        write_json(receipt, record)
        records[shard] = record
        emit("shard_complete", shard=shard, projections=count, sha256=record["sha256"])
    if args.shard:
        emit("partial_complete", shard=args.shard, publication_ready=False)
        return
    updated_map = dict(index["weight_map"])
    quantized_layers = {}
    for prefix, projections in groups.items():
        for _, (key, shard) in projections.items():
            base = key.removesuffix(".weight")
            updated_map[base + ".weight_scale_2"] = shard
            updated_map[base + ".input_scale"] = shard
            quantized_layers[base] = {"quant_algo": "NVFP4", "group_size": 16}
    for key in index["weight_map"]:
        if key.endswith(".weight_scale_inv"):
            quantized_layers[key.removesuffix(".weight_scale_inv")] = {"quant_algo": "FP8_BLOCK_SCALES"}
    quantization = {
        "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
        "exclude_modules": [], "quantized_layers": quantized_layers,
    }
    producer = {"name": "modelopt", "version": distribution.version}
    config = dict(source_config)
    config["quantization_config"] = {
        **quantization, "quant_method": "modelopt", "ignore": [], "producer": producer,
    }
    repairs = []
    for name in files:
        if name in shards or name in {"config.json", "model.safetensors.index.json"}:
            continue
        verify_source_file(source, name, files)
        target_name = "README_UPSTREAM.md" if name == "README.md" else name
        target = output / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "dflash/config.json":
            text = (source / name).read_text()
            try:
                json.loads(text)
            except json.JSONDecodeError:
                corrected = re.sub(r",\s*}\s*$", "\n}\n", text, count=1)
                json.loads(corrected)
                target.write_text(corrected)
                repairs.append({"path": name, "change": "remove invalid trailing comma"})
                continue
        shutil.copyfile(source / name, target)
    metadata = dict(index.get("metadata", {}))
    metadata.update(save_format="nvfp4", total_size=sum(r["tensor_bytes"] for r in records.values()))
    write_json(output / "model.safetensors.index.json", {"metadata": metadata, "weight_map": updated_map})
    write_json(output / "config.json", config)
    write_json(output / "hf_quant_config.json", {"producer": producer, "quantization": quantization})
    report = {
        "policy": policy, "policy_sha256": policy_hash,
        "expert_projection_count": sum(r["expert_projection_count"] for r in records.values()),
        "weight_reconstruction": "exact", "quality_qualification": "not run",
        "activation_execution_qualification": "not run",
        "source_metadata_repairs": repairs, "shards": records,
    }
    if report["expert_projection_count"] != 36096:
        raise ValueError("final expert tensor census mismatch")
    write_json(output / "mint-report.json", report)
    emit("mint_complete", projections=36096, output=str(output), publication_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-scale", type=float, required=True)
    parser.add_argument("--activation-scales", type=Path)
    parser.add_argument("--shard", help="Produce one checked shard for an initial component gate")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not math.isfinite(args.input_scale) or args.input_scale <= 0:
        parser.error("--input-scale must be finite and positive")
    mint(args)


if __name__ == "__main__":
    main()
