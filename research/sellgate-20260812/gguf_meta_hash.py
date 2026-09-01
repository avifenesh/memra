#!/usr/bin/env python3
"""Hash one exact string-valued GGUF v3 metadata field without loading tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO


FIXED_VALUE_BYTES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"truncated GGUF: wanted {size} bytes, got {len(data)}")
    return data


def read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_string_bytes(handle: BinaryIO) -> bytes:
    return read_exact(handle, read_u64(handle))


def skip_value(handle: BinaryIO, value_type: int) -> None:
    if value_type in FIXED_VALUE_BYTES:
        handle.seek(FIXED_VALUE_BYTES[value_type], 1)
    elif value_type == 8:  # STRING
        handle.seek(read_u64(handle), 1)
    elif value_type == 9:  # ARRAY
        element_type = read_u32(handle)
        count = read_u64(handle)
        fixed_size = FIXED_VALUE_BYTES.get(element_type)
        if fixed_size is not None:
            handle.seek(fixed_size * count, 1)
        else:
            for _ in range(count):
                skip_value(handle, element_type)
    else:
        raise ValueError(f"unknown GGUF metadata type {value_type}")


def metadata_string(path: Path, wanted_key: str) -> bytes:
    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = read_u32(handle)
        if version != 3:
            raise ValueError(f"{path}: expected GGUF v3, got v{version}")
        read_u64(handle)  # tensor count
        metadata_count = read_u64(handle)
        for _ in range(metadata_count):
            key = read_string_bytes(handle).decode("utf-8")
            value_type = read_u32(handle)
            if key == wanted_key:
                if value_type != 8:
                    raise ValueError(
                        f"{path}: {wanted_key!r} has type {value_type}, expected STRING"
                    )
                return read_string_bytes(handle)
            skip_value(handle, value_type)
    raise KeyError(f"{path}: missing metadata key {wanted_key!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--key", default="tokenizer.chat_template")
    args = parser.parse_args()

    for model in args.models:
        value = metadata_string(model, args.key)
        print(
            json.dumps(
                {
                    "key": args.key,
                    "model": str(model),
                    "sha256": hashlib.sha256(value).hexdigest(),
                    "size_bytes": len(value),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
