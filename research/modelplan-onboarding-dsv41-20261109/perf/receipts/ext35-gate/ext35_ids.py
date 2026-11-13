#!/usr/bin/env python3
"""ext35 order gate, step 0: the forced case list dsv41-run --cases reads (ID prompt.i32 gen.i32, the decode lane's
dec/GROUP-forced.txt format) from dec/ext35/ext35-oracle.json, and a check that every case's tape, REF and FLOOR hold
the same tokens and step count as its forced list.
usage: python -I ext35_ids.py   (wrx, torch venv)"""
import array
import json
import os

import torch

E = "/home/access/dsv41/dec/ext35"
cases = json.load(open(os.path.join(E, "ext35-oracle.json")))
os.makedirs(os.path.join(E, "ids"), exist_ok=True)
lines = []
steps = 0
for c in cases:
    i = c["id"]
    p = os.path.join(E, "ids", i + ".prompt.i32")
    g = os.path.join(E, "ids", i + ".gen.i32")
    with open(p, "wb") as f:
        array.array("i", [int(x) for x in c["prompt_ids"]]).tofile(f)
    with open(g, "wb") as f:
        array.array("i", [int(x) for x in c["force_ids"]]).tofile(f)
    T = torch.load(os.path.join(E, "tapes", i + ".pt"), weights_only=False, map_location="cpu")
    R = torch.load(os.path.join(E, "ref", i + ".pt"), weights_only=False, map_location="cpu")
    F = torch.load(os.path.join(E, "floor", i + ".pt"), weights_only=False, map_location="cpu")
    G = len(c["force_ids"])
    assert [int(x) for x in T["gen_ids"]] == [int(x) for x in c["force_ids"]], i
    assert [int(x) for x in T["prompt_ids"]] == [int(x) for x in c["prompt_ids"]], i
    assert len(R["top_ids"]) == G and len(R["lse"]) == G, (i, G, len(R["top_ids"]))
    assert len(F["gen_ids"]) == G and len(F["top_ids"]) == G and len(F["argmax"]) == G, (i, G)
    steps += G
    lines.append(f"{i} {p} {g}")
with open(os.path.join(E, "ext35-forced.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"ext35-forced.txt: {len(lines)} cases, {steps} steps; tapes/ref/floor agree; prompt lengths "
      f"{min(len(c['prompt_ids']) for c in cases)}-{max(len(c['prompt_ids']) for c in cases)}")
