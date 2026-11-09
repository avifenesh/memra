"""Inputs of the decode gates from the vendor's greedy tapes.

For each tape: the prompt ids and the greedy tokens as raw i32 (OUT/ids/ID.prompt.i32, OUT/ids/ID.gen.i32), a
`dsv41-run --cases` file for teacher-forced decode (OUT/NAME-forced.txt: ID PROMPT GEN) and for greedy generation
(OUT/NAME-greedy.txt: ID PROMPT), and the oracle's forced cases (OUT/NAME-oracle.json: prompt_ids and force_ids, for
make_forced_oracle.py's copies).

usage: prep_decode.py TAPES_DIR OUT_DIR NAME ID [ID ...]"""
import json
import os
import sys

import torch

tapes, out, name, ids = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3], sys.argv[4:]
os.makedirs(os.path.join(out, "ids"), exist_ok=True)
forced, greedy, oracle = [], [], []
for i in ids:
    T = torch.load(os.path.join(tapes, f"{i}.pt"), weights_only=False)
    p, g = list(T["prompt_ids"]), [int(x) for x in T["gen_ids"]]
    pf, gf = os.path.join(out, "ids", f"{i}.prompt.i32"), os.path.join(out, "ids", f"{i}.gen.i32")
    open(pf, "wb").write(torch.tensor(p, dtype=torch.int32).numpy().tobytes())
    open(gf, "wb").write(torch.tensor(g, dtype=torch.int32).numpy().tobytes())
    forced.append(f"{i} {pf} {gf}")
    greedy.append(f"{i} {pf}")
    oracle.append({"id": i, "prompt_ids": p, "force_ids": g})
    print(i, "prompt", len(p), "steps", len(g))
open(os.path.join(out, f"{name}-forced.txt"), "w").write("\n".join(forced) + "\n")
open(os.path.join(out, f"{name}-greedy.txt"), "w").write("\n".join(greedy) + "\n")
json.dump(oracle, open(os.path.join(out, f"{name}-oracle.json"), "w"))
