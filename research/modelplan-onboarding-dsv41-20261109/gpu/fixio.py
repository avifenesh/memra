"""Fixture IO for the dsv41 GPU gates: a directory of raw little-endian tensors plus index.json.

index.json: {"tensors": {name: {"dtype": str, "shape": [...], "file": str}}, "meta": {...}}
dtypes: bf16 f32 i32 i64 u8 i8 f8e4m3 e8m0 (raw bytes, row-major)."""
import json
import os

import torch

_DT = {
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.int8: "i8",
    torch.bool: "u8",
    torch.float8_e4m3fn: "f8e4m3",
    torch.float8_e8m0fnu: "e8m0",
}


class Fix:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.idx = {"tensors": {}, "meta": {}}

    def put(self, name, t):
        t = t.detach().to("cpu").contiguous()
        if t.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.int8)
        dt = _DT[t.dtype]
        shape = list(t.shape)
        flat = t.reshape(-1)
        if flat.dtype == torch.bool:
            flat = flat.to(torch.uint8)
        raw = flat.view(torch.uint8) if flat.numel() else torch.empty(0, dtype=torch.uint8)
        fn = name.replace("/", "__") + ".bin"
        raw.numpy().tofile(os.path.join(self.root, fn))
        self.idx["tensors"][name] = {"dtype": dt, "shape": shape, "file": fn}

    def meta(self, k, v):
        self.idx["meta"][k] = v

    def close(self):
        with open(os.path.join(self.root, "index.json"), "w") as f:
            json.dump(self.idx, f, indent=1)
        # the same index as TSV for the Rust gate (no JSON dependency there): name dtype shape file
        with open(os.path.join(self.root, "index.tsv"), "w") as f:
            for k, v in self.idx["tensors"].items():
                f.write(f"{k}\t{v['dtype']}\t{','.join(str(d) for d in v['shape'])}\t{v['file']}\n")
        with open(os.path.join(self.root, "meta.tsv"), "w") as f:
            for k, v in self.idx["meta"].items():
                f.write(f"{k}\t{json.dumps(v)}\n")
