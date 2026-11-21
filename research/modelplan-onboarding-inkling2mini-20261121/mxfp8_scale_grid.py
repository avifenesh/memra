#!/usr/bin/env python3
"""Read the MXFP8 expert scale planes of an Inkling 2 Mini checkpoint and report their grid.

`hf_quant_config.json` declares `weight_block_shape [32, 32]` for the MXFP8 expert group. The
stored scale plane of a bank `[experts, out, in]` has shape `[experts, out, in / 32]`: one byte
per row per 32 inputs. If the weights had been scaled per 32x32 block and the plane merely
repeated each block's exponent over its 32 rows, every aligned group of 32 consecutive rows
would hold 32 identical rows. This script counts those groups.

Usage: mxfp8_scale_grid.py CHECKPOINT_DIR OUT_TSV
"""
import json
import struct
import sys

FILE = "model-00001-of-00001.safetensors"
LAYERS = (20, 21, 22, 23)
BANKS = ("w13_weight", "w2_weight")


def main(checkpoint, out_path):
    path = f"{checkpoint}/{FILE}"
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        rows_out = []
        for layer in LAYERS:
            for bank in BANKS:
                name = f"model.llm.layers.{layer}.mlp.experts.{bank}_scale"
                info = header[name]
                assert info["dtype"] == "U8", (name, info["dtype"])
                experts, out, blocks = info["shape"]
                assert out % 32 == 0, (name, out)
                begin, end = info["data_offsets"]
                assert end - begin == experts * out * blocks
                groups = uniform = 0
                low, high = 255, 0
                differing_rows = 0
                for expert in range(experts):
                    f.seek(base + begin + expert * out * blocks)
                    plane = f.read(out * blocks)
                    low = min(low, min(plane))
                    high = max(high, max(plane))
                    for group in range(out // 32):
                        start = group * 32 * blocks
                        first = plane[start : start + blocks]
                        same = sum(
                            plane[start + r * blocks : start + (r + 1) * blocks] == first
                            for r in range(1, 32)
                        )
                        groups += 1
                        uniform += same == 31
                        differing_rows += 31 - same
                rows_out.append(
                    (name, f"{experts},{out},{blocks}", groups, uniform, differing_rows, low, high)
                )
    with open(out_path, "w") as f:
        f.write("# MXFP8 expert scale planes: aligned 32-row groups whose 32 rows are identical.\n")
        f.write("# A 32x32 block grid repeated over rows would make every group uniform.\n")
        f.write("tensor\tshape\trow_groups\tuniform_groups\trows_differing_from_group_first\tmin_exponent\tmax_exponent\n")
        for row in rows_out:
            f.write("\t".join(map(str, row)) + "\n")
    for row in rows_out:
        print(*row)


if __name__ == "__main__":
    main(*sys.argv[1:])
