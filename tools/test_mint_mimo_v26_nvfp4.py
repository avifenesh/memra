"""Numeric and artifact-integrity gates; run on the rented qualification host."""

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location(
    "mimo_mint", Path(__file__).with_name("mint_mimo_v26_nvfp4.py")
)
MINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MINT)


def decode(packed):
    values = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return values[codes.long()]


def source_values(packed, scales):
    return decode(packed) * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)


def target_values(packed, scales, global_scale):
    return decode(packed) * scales.double().repeat_interleave(16, -1) * global_scale


@pytest.mark.parametrize("exponent", [-50, -12, 0, 12, 50])
@pytest.mark.parametrize("code_offset", [0, 3, 8, 13])
def test_decoded_weights_survive_every_sign_and_grid_value(exponent, code_offset):
    packed = ((torch.arange(32, dtype=torch.int64) * 17 + code_offset) % 256).to(torch.uint8).reshape(1, -1)
    scales = torch.tensor([[127 + exponent, 126 + exponent]], dtype=torch.uint8)
    global_scale = MINT.shared_global_scale([(packed, scales)])
    converted, _ = MINT.cast_scales(packed, scales, global_scale)
    assert torch.equal(source_values(packed, scales), target_values(packed, converted, global_scale))


def test_gate_and_up_share_a_global_without_changing_either_projection():
    packed = torch.full((2, 32), 0xF1, dtype=torch.uint8)
    gate = torch.tensor([[100, 110], [108, 107]], dtype=torch.uint8)
    up = torch.tensor([[105, 117], [116, 109]], dtype=torch.uint8)
    shared = MINT.shared_global_scale([(packed, gate), (packed, up)])
    for original in [gate, up]:
        converted, _ = MINT.cast_scales(packed, original, shared)
        assert torch.equal(source_values(packed, original), target_values(packed, converted, shared))


def test_zero_payload_does_not_force_a_lossy_global_scale():
    packed = torch.zeros((1, 32), dtype=torch.uint8)
    packed[0, 16:] = 0x77
    scales = torch.tensor([[1, 127]], dtype=torch.uint8)
    global_scale = MINT.shared_global_scale([(packed, scales)])
    converted, _ = MINT.cast_scales(packed, scales, global_scale)
    assert torch.equal(source_values(packed, scales), target_values(packed, converted, global_scale))


def test_nonzero_underflow_refuses_instead_of_silently_rounding():
    packed = torch.full((1, 32), 0x11, dtype=torch.uint8)
    scales = torch.tensor([[100, 118]], dtype=torch.uint8)
    global_scale = MINT.shared_global_scale([(packed, scales)])
    with pytest.raises(ValueError, match="lossless NVFP4 conversion failed"):
        MINT.cast_scales(packed, scales, global_scale)


def test_nan_scale_refuses_even_for_zero_payload():
    with pytest.raises(ValueError, match="invalid E8M0"):
        MINT.shared_global_scale([
            (torch.zeros((1, 16), dtype=torch.uint8), torch.tensor([[255]], dtype=torch.uint8))
        ])


def test_group_geometry_is_checked():
    with pytest.raises(ValueError, match="one E8M0 scale"):
        MINT.nonzero_blocks(torch.zeros((1, 16), dtype=torch.uint8), torch.ones((1, 2), dtype=torch.uint8))


def test_incomplete_expert_bank_is_not_a_valid_mint():
    with pytest.raises(ValueError, match="complete 47-layer"):
        MINT.expert_inventory({"weight_map": {}})


def shard_pair(tmp_path):
    prefix = "model.layers.1.mlp.experts.0.gate_proj"
    packed = torch.full((2, 16), 0x73, dtype=torch.uint8)
    scales = torch.tensor([[123], [127]], dtype=torch.uint8)
    global_scale = MINT.shared_global_scale([(packed, scales)])
    converted, _ = MINT.cast_scales(packed, scales, global_scale)
    old = {
        prefix + ".weight": packed,
        prefix + ".weight_scale": scales,
        "model.layers.1.mlp.gate.weight": torch.arange(8).reshape(2, 4).to(torch.bfloat16),
    }
    new = {
        **old,
        prefix + ".weight_scale": converted,
        prefix + ".weight_scale_2": torch.tensor([global_scale], dtype=torch.float32),
        prefix + ".input_scale": torch.tensor([1.0], dtype=torch.float32),
    }
    a, b = tmp_path / "source.safetensors", tmp_path / "mint.safetensors"
    save_file(old, a)
    save_file(new, b)
    return prefix, a, b, new


def test_serialized_artifact_reconstruction(tmp_path):
    _, a, b, _ = shard_pair(tmp_path)
    assert MINT.verify_shard(a, b) == 1


@pytest.mark.parametrize("damage", ["payload", "scale", "global", "router", "missing", "extra", "activation"])
def test_serialized_corruption_is_detected(tmp_path, damage):
    prefix, a, b, new = shard_pair(tmp_path)
    if damage == "payload":
        new[prefix + ".weight"] = new[prefix + ".weight"].clone()
        new[prefix + ".weight"][0, 0] ^= 1
    elif damage == "scale":
        new[prefix + ".weight_scale"] = torch.ones_like(new[prefix + ".weight_scale"])
    elif damage == "global":
        new[prefix + ".weight_scale_2"] *= 2
    elif damage == "router":
        new["model.layers.1.mlp.gate.weight"] += 1
    elif damage == "missing":
        del new[prefix + ".input_scale"]
    elif damage == "extra":
        new["unexpected.weight"] = torch.ones(1)
    else:
        new[prefix + ".input_scale"] = torch.tensor([float("inf")])
    save_file(new, b)
    with pytest.raises(ValueError):
        MINT.verify_shard(a, b)
