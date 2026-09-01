#!/usr/bin/env python3
"""Build the three arm train files: own x4 oversample + pb mix / pb only / own mix copy."""
import json, pathlib, shutil

out = pathlib.Path("/scratch/corpus/train")
out.mkdir(exist_ok=True)


def rows(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.is_file() else []


own = []
for f in ["own-think-exploded.jsonl", "own-nothink.jsonl",
          "own-mt-think-exploded.jsonl", "own-mt-nothink.jsonl"]:
    own += rows(f"/scratch/corpus/regen/{f}")
pb = rows("/scratch/corpus/regen/pb-think-exploded.jsonl") + rows("/scratch/corpus/regen/pb-nothink.jsonl")
print("own rows", len(own), "pb rows", len(pb))

with (out / "arm-a-own-mix.jsonl").open("w") as f:
    for i in range(4):
        for r in own:
            f.write(json.dumps({**r, "id": "{}-dup{}".format(r.get("id", "x"), i)}) + "\n")
    for r in pb:
        f.write(json.dumps(r) + "\n")
with (out / "arm-b-pb.jsonl").open("w") as f:
    for r in pb:
        f.write(json.dumps(r) + "\n")
shutil.copy(out / "arm-a-own-mix.jsonl", out / "arm-c-own-mix.jsonl")
print("train files rebuilt")
