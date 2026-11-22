#!/usr/bin/env python3
"""Lane E's CPU oracle tape -> memra-checkpoint-oracle-v1 at the last prompt position (the probe).
usage: tape_oracle.py TAPE_DIR OUT.tsv ENGINE"""
import json
import struct
import sys

import numpy as np

CLASS = "inkling2-declared-activation-quant"


def tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        dt = {"F32": "<f4", "I64": "<i8", "I32": "<i4"}[v["dtype"]]
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(data[s:e], dtype=dt).reshape(v["shape"])
    return out


tape, out, engine = sys.argv[1:4]
tok = tensors(f"{tape}/tokens.safetensors")
lp = tensors(f"{tape}/logits_post.safetensors")
ids = [int(i) for i in tok["input_ids"]]
row = int(np.nonzero(lp["positions"] == len(ids) - 1)[0][0])
bits = np.ascontiguousarray(lp["logits_post"][row], dtype="<f4").view("<u4")
with open(out, "w") as f:
    f.write(f"format\tmemra-checkpoint-oracle-v1\nengine\t{engine}\nnumeric_class\t{CLASS}\n")
    f.write("tokens\t" + ",".join(map(str, ids)) + "\n")
    f.write(f"vocab\t{len(bits)}\n")
    for i, b in enumerate(bits):
        f.write(f"logit\t{i}\t{int(b):08x}\n")
print(len(ids), "tokens; argmax", int(lp["logits_post"][row].argmax()))
