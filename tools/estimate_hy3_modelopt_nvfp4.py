#!/usr/bin/env python3
"""Estimate HY3 ModelOpt NVFP4 payloads from a Memra tensor census."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


FORMAT = "memra-hy3-modelopt-size-v1"
PROFILES = ("experts", "mlp", "omlp", "full")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}
@dataclass(frozen=True)
class Row:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            shape = ast.literal_eval(raw["shape"])
            if not isinstance(shape, list) or not all(
                isinstance(value, int) and value >= 0 for value in shape
            ):
                raise ValueError(f"invalid shape for {raw['semantic_name']}: {shape!r}")
            if raw["dtype"] not in DTYPE_BYTES:
                raise ValueError(
                    f"unsupported source dtype {raw['dtype']} for {raw['semantic_name']}"
                )
            rows.append(Row(raw["semantic_name"], raw["dtype"], tuple(shape)))
    if not rows:
        raise ValueError("empty tensor census")
    return rows


def quantizes(profile: str, row: Row, mtp_layer: int) -> bool:
    del mtp_layer
    projection = row.name.endswith("_proj.weight")
    expert = ".mlp.experts." in row.name and projection
    mlp = ".mlp." in row.name and projection and ".router." not in row.name
    output_projection = row.name.endswith(".self_attn.o_proj.weight")
    full_linear = (
        len(row.shape) == 2
        and row.name.endswith(".weight")
        and row.name not in {"model.embed_tokens.weight", "lm_head.weight"}
        and ".router." not in row.name
    )
    return {
        "experts": expert,
        "mlp": mlp,
        "omlp": mlp or output_projection,
        "full": full_linear,
    }[profile]


def nvfp4_bytes(elements: int) -> int:
    # ModelOpt unified HF: two E2M1 values per U8, one E4M3 scale per 16 values,
    # and one F32 tensor-level multiplier. Dynamic activation scales are runtime values.
    return (elements + 1) // 2 + (elements + 15) // 16 + 4


def estimate(rows: list[Row], profile: str, mtp_layer: int) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    source_bytes = 0
    output_bytes = 0
    quantized_elements = 0
    quantized_tensors = 0
    for row in rows:
        elements = row.elements
        source_bytes += elements * DTYPE_BYTES[row.dtype]
        if quantizes(profile, row, mtp_layer):
            if len(row.shape) != 2 or not row.shape or row.shape[-1] % 16:
                raise ValueError(
                    f"profile {profile} selected non-NVFP4-compatible tensor "
                    f"{row.name} {row.shape}"
                )
            output_bytes += nvfp4_bytes(elements)
            quantized_elements += elements
            quantized_tensors += 1
        else:
            output_bytes += elements * DTYPE_BYTES[row.dtype]
    return {
        "profile": profile,
        "source_payload_bytes": source_bytes,
        "predicted_payload_bytes": output_bytes,
        "predicted_payload_gib": output_bytes / (1 << 30),
        "reduction_fraction": 1.0 - output_bytes / source_bytes,
        "quantized_tensors": quantized_tensors,
        "quantized_elements": quantized_elements,
        "kept_tensors": len(rows) - quantized_tensors,
    }


def self_test() -> None:
    rows = [
        Row("model.layers.1.mlp.experts.0.gate_proj.weight", "BF16", (16, 32)),
        Row("model.layers.2.mlp.experts.0.gate_proj.weight", "BF16", (16, 32)),
        Row("model.layers.1.self_attn.q_proj.weight", "BF16", (16, 32)),
        Row("lm_head.weight", "F32", (4, 8)),
    ]
    experts = estimate(rows, "experts", 2)
    assert experts["quantized_tensors"] == 2
    full = estimate(rows, "full", 2)
    assert full["quantized_tensors"] == 3
    assert full["predicted_payload_bytes"] < experts["predicted_payload_bytes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", nargs="?", type=Path)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--mtp-layer", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("HY3 ModelOpt size estimator: PASS")
        return 0
    if args.census is None:
        parser.error("census is required unless --self-test is used")
    rows = load_rows(args.census)
    payload = {
        "format": FORMAT,
        "census": str(args.census),
        "census_sha256": hashlib.sha256(args.census.read_bytes()).hexdigest(),
        "mtp_layer": args.mtp_layer,
        "profiles": [
            estimate(rows, profile, args.mtp_layer)
            for profile in (args.profile or list(PROFILES))
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
