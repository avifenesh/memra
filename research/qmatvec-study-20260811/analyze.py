#!/usr/bin/env python3
"""Reproduce the IQ4_XS qmatvec byte, roofline, and ceiling arithmetic.

This script reads only the committed ncuspike summary.  The semantic launch map
comes from the committed Step-3.7 layer geometry and the Nsys trace order cited
in REPORT.md; assertions keep it tied to the observed 154 + 161 launches/token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "research/ncuspike-20260811/summary.json"

CARD_BW_GBS = 1597.0
CARD_FP32_TFLOPS = 120.0
IQ4_XS_BLOCK_VALUES = 256
IQ4_XS_BLOCK_BYTES = 136


@dataclass(frozen=True)
class LaunchClass:
    count: int
    out_f: int
    in_f: int
    name: str


# Device 0 owns 22 layers (6 SWA, 16 full attention), including three dense
# FFNs and 19 MoE/shared FFNs.  Device 1 owns 23 MoE layers (6 SWA, 17 full).
SEMANTIC_LAUNCHES = {
    0: (
        LaunchClass(6, 8192, 4096, "attention q, SWA"),
        LaunchClass(16, 12288, 4096, "attention q, full"),
        LaunchClass(22, 1024, 4096, "attention k"),
        LaunchClass(6, 64, 4096, "head gate, SWA"),
        LaunchClass(16, 96, 4096, "head gate, full"),
        LaunchClass(6, 4096, 8192, "attention out, SWA"),
        LaunchClass(16, 4096, 12288, "attention out, full"),
        LaunchClass(6, 11264, 4096, "dense gate/up"),
        LaunchClass(3, 4096, 11264, "dense down"),
        LaunchClass(38, 1280, 4096, "shared gate/up"),
        LaunchClass(19, 4096, 1280, "shared down"),
    ),
    1: (
        LaunchClass(6, 8192, 4096, "attention q, SWA"),
        LaunchClass(17, 12288, 4096, "attention q, full"),
        LaunchClass(23, 1024, 4096, "attention k"),
        LaunchClass(6, 64, 4096, "head gate, SWA"),
        LaunchClass(17, 96, 4096, "head gate, full"),
        LaunchClass(6, 4096, 8192, "attention out, SWA"),
        LaunchClass(17, 4096, 12288, "attention out, full"),
        LaunchClass(46, 1280, 4096, "shared gate/up"),
        LaunchClass(23, 4096, 1280, "shared down"),
    ),
}

# NCU used --filter-mode per-launch-config, which keys the 4096-row launches
# together despite their different in_f arguments.  Its first such launch in
# the committed trace is the SWA attention-out projection, in_f=8192.
NCU_CAPTURE_IN_F = {
    64: 4096,
    96: 4096,
    1024: 4096,
    1280: 4096,
    4096: 8192,
    8192: 4096,
    11264: 4096,
    12288: 4096,
}


def weight_bytes(out_f: int, in_f: int) -> int:
    assert in_f % IQ4_XS_BLOCK_VALUES == 0
    return out_f * (in_f // IQ4_XS_BLOCK_VALUES) * IQ4_XS_BLOCK_BYTES


def activation_bytes(in_f: int) -> int:
    # Memra Stage B: one int8 per value plus one f32 scale per 32 values.
    assert in_f % 32 == 0
    return in_f + (in_f // 32) * 4


def logical_bytes(out_f: int, in_f: int) -> int:
    return weight_bytes(out_f, in_f) + activation_bytes(in_f) + out_f * 4


def useful_ops(out_f: int, in_f: int) -> int:
    # Conventional roofline accounting: one multiply and one add per weight.
    return 2 * out_f * in_f


def arithmetic_intensity(out_f: int, in_f: int) -> float:
    return useful_ops(out_f, in_f) / logical_bytes(out_f, in_f)


def kernel_row(summary: dict, device: int) -> dict:
    rows = summary["nsys"]["per_device"][str(device)]["kernels"]
    return next(row for row in rows if row["name"] == "qmatvec_iq4_XS_dp4a")


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    assert summary["nsys"]["n_tokens"] == 32
    assert weight_bytes(1, 4096) == 2176
    assert activation_bytes(4096) == 4608

    print("IQ4_XS + q8_1 layout")
    print(f"  weight: {IQ4_XS_BLOCK_BYTES} B / {IQ4_XS_BLOCK_VALUES} = "
          f"{IQ4_XS_BLOCK_BYTES / IQ4_XS_BLOCK_VALUES:.5f} B/value")
    print(f"  K=4096 row: {weight_bytes(1, 4096)} B")
    print(f"  K=4096 activation: {activation_bytes(4096)} B")
    per_row_ai = useful_ops(1, 4096) / logical_bytes(1, 4096)
    print(f"  pessimistic per-row AI: {per_row_ai:.4f} FLOP/B")
    print()

    ridge = CARD_FP32_TFLOPS * 1000 / CARD_BW_GBS
    print("Roofline")
    print(f"  FP32/BW ridge: {ridge:.2f} FLOP/B")
    print("| out_f | in_f | logical MB | AI FLOP/B | BW roof TFLOP/s |")
    print("|---:|---:|---:|---:|---:|")
    for out_f in (64, 96, 1024, 1280, 8192, 11264, 12288):
        in_f = 4096
        ai = arithmetic_intensity(out_f, in_f)
        roof = ai * CARD_BW_GBS / 1000
        print(f"| {out_f} | {in_f} | {logical_bytes(out_f, in_f) / 1e6:.4f} | "
              f"{ai:.4f} | {roof:.3f} |")
    out_f, in_f = 4096, 8192
    ai = arithmetic_intensity(out_f, in_f)
    print(f"| {out_f} | {in_f} | {logical_bytes(out_f, in_f) / 1e6:.4f} | "
          f"{ai:.4f} | {ai * CARD_BW_GBS / 1000:.3f} |")
    print()

    configs = [
        row for row in summary["ncu"]["launch_configs"]
        if row["kernel"] == "qmatvec_iq4_XS_dp4a"
    ]
    assert {row["grid"][0] for row in configs} == set(NCU_CAPTURE_IN_F)
    print("NCU captured launch configurations")
    print("| out_f | captured in_f | logical MB | DRAM MB | DRAM/logical | "
          "card BW % | occ % | waves/SM | long SB | LG throttle |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    weighted_dram = 0.0
    weighted_logical = 0.0
    for row in sorted(configs, key=lambda item: item["grid"][0]):
        out_f = row["grid"][0]
        in_f = NCU_CAPTURE_IN_F[out_f]
        logical = logical_bytes(out_f, in_f)
        dram = row["dram_gbs"] * row["ncu_duration_us"] * 1000
        # Match ncuspike's Nsys-time weighting for a comparable aggregate ratio.
        weight = row["nsys_per_token_ms"]
        weighted_dram += weight * dram
        weighted_logical += weight * logical
        print(f"| {out_f} | {in_f} | {logical / 1e6:.4f} | {dram / 1e6:.4f} | "
              f"{dram / logical:.4f}x | {row['dram_pct_of_1597']:.2f} | "
              f"{row['occupancy_pct']:.2f} | {row['waves_per_sm']:.2f} | "
              f"{row['stall_long_scoreboard']:.2f} | {row['stall_lg_throttle']:.2f} |")
    print(f"  Nsys-time-weighted NCU DRAM/logical: {weighted_dram / weighted_logical:.4f}x")
    print()

    print("Unperturbed Nsys logical-byte accounting")
    print("| device | launches/token | weight GB | logical GB | Nsys ms/token | "
          "logical GB/s | card BW % |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    total_launches = 0
    total_weight = 0
    total_logical = 0
    total_time_s = 0.0
    for device, classes in SEMANTIC_LAUNCHES.items():
        launches = sum(item.count for item in classes)
        weights = sum(item.count * weight_bytes(item.out_f, item.in_f) for item in classes)
        logical = sum(item.count * logical_bytes(item.out_f, item.in_f) for item in classes)
        row = kernel_row(summary, device)
        assert row["launches_per_token"] == launches
        time_s = row["per_token_ms"] / 1000
        rate = logical / time_s / 1e9
        print(f"| {device} | {launches} | {weights / 1e9:.6f} | {logical / 1e9:.6f} | "
              f"{row['per_token_ms']:.6f} | {rate:.2f} | {100 * rate / CARD_BW_GBS:.2f} |")
        total_launches += launches
        total_weight += weights
        total_logical += logical
        total_time_s += time_s
    assert total_launches == 315
    assert total_weight == 2_872_946_688
    assert total_logical == 2_879_039_040
    rate = total_logical / total_time_s / 1e9
    print(f"| both | {total_launches} | {total_weight / 1e9:.6f} | "
          f"{total_logical / 1e9:.6f} | {total_time_s * 1000:.6f} | "
          f"{rate:.2f} | {100 * rate / CARD_BW_GBS:.2f} |")
    print()

    wall_s = summary["nsys"]["wall_per_token_ms"] / 1000
    floor_s = total_logical / (CARD_BW_GBS * 1e9)
    saving_s = total_time_s - floor_s
    print("Ceilings")
    print(f"  qmatvec full logical-byte floor: {floor_s * 1000:.6f} ms/token")
    print(f"  ideal qmatvec speedup: {total_time_s / floor_s:.4f}x")
    print(f"  ideal wall saving: {saving_s * 1000:.6f} ms/token "
          f"({100 * saving_s / wall_s:.2f}% of wall)")
    print(f"  ideal end-to-end throughput uplift: "
          f"{100 * (wall_s / (wall_s - saving_s) - 1):.2f}%")


if __name__ == "__main__":
    main()
