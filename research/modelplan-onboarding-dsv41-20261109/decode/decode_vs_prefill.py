"""memra's decode steps against memra's own teacher-forced prefill of the same tokens, step by step.

Decode step k of a sequence with a P-token prompt sits at position P - 1 + k of a prefill of the prompt plus the fed
tokens. Below the window (128 positions) the vendor's decode program reads its keys in another order than its prefill
(the ring, oldest slot first, -1 for empty slots), so the two differ there by construction. Above it the ring order is
position order and memra's kernels are per row, so memra's two programs may give the same bits there; this counts the
steps whose 32 largest logits (ids and values) are bitwise equal.

usage: decode_vs_prefill.py DECODE_DIR PREFILL_TF_DIR PROMPT_LEN [--json OUT]"""
import json
import sys

import torch


def raw(p, dt):
    return torch.frombuffer(bytearray(open(p, "rb").read()), dtype=dt)


dec, pre, P = sys.argv[1], sys.argv[2], int(sys.argv[3])
di = raw(f"{dec}/top_ids.i32", torch.int32).view(-1, 32)
dv = raw(f"{dec}/top_vals.f32", torch.float32).view(-1, 32)
pi = raw(f"{pre}/top_ids.i32", torch.int32).view(-1, 32)
pv = raw(f"{pre}/top_vals.f32", torch.float32).view(-1, 32)
G = len(di)
same = [bool(torch.equal(di[k], pi[P - 1 + k]) and torch.equal(dv[k], pv[P - 1 + k])) for k in range(G)]
dmax = [float((dv[k].double() - pv[P - 1 + k].double()).abs().max()) for k in range(G)]
out = {"decode": dec, "prefill": pre, "prompt": P, "steps": G, "bitwise_equal_steps": sum(same),
       "first_unequal": same.index(False) if not all(same) else None,
       "step0_is_the_prefill_row": same[0], "max_abs_top32_diff": max(dmax)}
print(json.dumps(out, indent=1))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
