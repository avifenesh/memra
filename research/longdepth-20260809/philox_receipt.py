#!/usr/bin/env python3
"""Reproduce the two plain-device Gumbel corruption receipts from frozen responses."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/longdepth-20260809/raw/20260808T233600Z/cells"
CELL = BASE / "ctx262144-off-t0p7-p1-gpu-d12288-baseline-rerun"
RECEIPTS = (
    (CELL / "rep1/response.json", 2_026_080_901, 504, 94_712, 4_294_967_240),
    (CELL / "rep2/response.json", 2_026_080_902, 281, 57_066, 4_294_967_187),
)
FIXED_CELL = (
    ROOT
    / "research/longdepth-20260809/raw/20260809T000200Z-fixverify-box1/cells"
    / "ctx262144-off-t0p7-p1-gpu-d12288-gumbel-fix"
)
FIRST_FIXED_DIVERGENCES = (
    (1, 2_026_080_901, 504, 94_712, 28, 4_294_967_240),
    (2, 2_026_080_902, 3, 41_907, 267, 4_294_967_176),
)
MASK32 = (1 << 32) - 1


def f32(value: float | int) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def philox4(seed: int, ctr_lo: int, ctr_hi: int) -> tuple[int, int, int, int]:
    m0, m1 = 0xD2511F53, 0xCD9E8D57
    c0, c1, c2, c3 = ctr_lo, ctr_hi, 0, 0
    k0, k1 = seed & MASK32, (seed >> 32) & MASK32
    for _ in range(10):
        p0, p1 = m0 * c0, m1 * c2
        h0, l0 = (p0 >> 32) & MASK32, p0 & MASK32
        h1, l1 = (p1 >> 32) & MASK32, p1 & MASK32
        c0, c1, c2, c3 = h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0
        k0 = (k0 + 0x9E3779B9) & MASK32
        k1 = (k1 + 0xBB67AE85) & MASK32
    return c0, c1, c2, c3


def old_u01(value: int) -> float:
    return f32(f32(f32(value) + f32(1.0)) * f32(1.0 / 4_294_967_296.0))


def main() -> None:
    below_one = float.fromhex("0x1.fffffep-1")
    rounded_values = sum(
        old_u01(value) == 1.0 for value in range((1 << 32) - 1024, 1 << 32)
    )
    per_vocab_draw = rounded_values / float(1 << 32)
    vocab = 128_896
    per_sample_event = 1.0 - (1.0 - per_vocab_draw) ** vocab
    assert rounded_values == 128
    print(
        f"u32_values_rounding_to_one={rounded_values} per_vocab_draw={per_vocab_draw:.12g} "
        f"vocab={vocab} iid_per_sample_event={per_sample_event:.9f} "
        f"iid_mean_tokens_to_event={1.0 / per_sample_event:.3f}"
    )
    for response_path, seed, stream_pos, token_id, expected_value in RECEIPTS:
        response = json.loads(response_path.read_text())
        observed_id = response["tokens"][stream_pos]
        lanes = philox4(seed, token_id >> 2, stream_pos)
        lane = token_id & 3
        value = lanes[lane]
        old_u = old_u01(value)
        fixed_u = min(old_u, below_one)
        assert observed_id == token_id
        assert value == expected_value
        assert old_u == 1.0
        assert fixed_u < 1.0
        fixed_gumbel = -math.log(-math.log(fixed_u))
        print(
            f"seed={seed} stream_pos={stream_pos} observed_token_id={observed_id} "
            f"lane={lane} philox_u32={value} hex=0x{value:08x} "
            f"old_f32_u={old_u:.1f} old_gumbel=+inf "
            f"fixed_f32_u={fixed_u.hex()} fixed_gumbel={fixed_gumbel:.9f}"
        )
    for rep, seed, expected_pos, old_id, fixed_id, expected_value in FIRST_FIXED_DIVERGENCES:
        old_tokens = json.loads((CELL / f"rep{rep}/response.json").read_text())["tokens"]
        fixed_tokens = json.loads((FIXED_CELL / f"rep{rep}/response.json").read_text())["tokens"]
        shared = min(len(old_tokens), len(fixed_tokens))
        divergence = next(i for i in range(shared) if old_tokens[i] != fixed_tokens[i])
        value = philox4(seed, old_tokens[divergence] >> 2, divergence)[
            old_tokens[divergence] & 3
        ]
        assert (divergence, old_tokens[divergence], fixed_tokens[divergence], value) == (
            expected_pos,
            old_id,
            fixed_id,
            expected_value,
        )
        assert old_u01(value) == 1.0
        print(
            f"postfix_rep={rep} exact_prefix_tokens={divergence} "
            f"old_token_id={old_id} fixed_token_id={fixed_id} "
            f"old_selected_philox_u32={value} old_f32_u=1.0"
        )
    print("receipt_match=PASS")


if __name__ == "__main__":
    main()
