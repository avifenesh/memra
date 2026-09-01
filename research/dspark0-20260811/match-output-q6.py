#!/usr/bin/env python3
"""Match an exported draft's output.weight to the serving GGUF's Q6_K encoding."""

import argparse
import json
import subprocess
from pathlib import Path

import gguf
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", required=True, type=Path)
    parser.add_argument("--carrier", required=True, type=Path)
    parser.add_argument("--metadata-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--quantizer",
        type=Path,
        default=Path("/data/projects/llama.cpp/build/bin/llama-quantize"),
    )
    args = parser.parse_args()

    index = json.loads((args.hf / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["lm_head.weight"]
    with safe_open(args.hf / shard, framework="pt", device="cpu") as handle:
        head = handle.get_tensor("lm_head.weight").float().half().numpy()

    reader = gguf.GGUFReader(args.carrier)
    metadata_source = gguf.GGUFReader(args.metadata_source)
    arch = reader.fields["general.architecture"].contents()
    intermediate = args.out.with_suffix(".f16.gguf")
    writer = gguf.GGUFWriter(intermediate, arch)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)
    carrier_fields = {field.name for field in reader.fields.values()}
    for field in metadata_source.fields.values():
        if not field.name.startswith(f"{arch}.") or field.name in carrier_fields:
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)

    for tensor in reader.tensors:
        if tensor.name == "output.weight":
            writer.add_tensor(tensor.name, head)
        else:
            writer.add_tensor(
                tensor.name,
                tensor.data,
                raw_shape=tensor.data.shape,
                raw_dtype=tensor.tensor_type,
                tensor_endianess=reader.endianess,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    try:
        subprocess.run(
            [
                args.quantizer,
                "--allow-requantize",
                "--output-tensor-type",
                "q6_k",
                intermediate,
                args.out,
                "Q8_0",
            ],
            check=True,
        )
    finally:
        intermediate.unlink(missing_ok=True)
    print(f"wrote {args.out}: output.weight Q6_K from {args.hf / shard}")


if __name__ == "__main__":
    main()
