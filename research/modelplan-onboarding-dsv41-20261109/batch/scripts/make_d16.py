#!/usr/bin/env python3
"""Sixteen distinct forced-token streams of 30 tokens for the wide batch jobs, cut from the lane's generated
streams, and the plan that uses them (16 short prompts, one long one among them).
usage: make_d16.py LANE_DIR"""
import os
import struct
import sys

lane = sys.argv[1].rstrip("/")
ids = os.path.join(lane, "dec/ids")
out = os.path.join(lane, "spec/ids/d16")
os.makedirs(out, exist_ok=True)


def rd(p):
    b = open(p, "rb").read()
    return list(struct.unpack(f"<{len(b) // 4}i", b))


streams = []
for name in ("hebrew_para", "story", "decode32k_he_summary", "decode64k_en_continue", "code_pal", "code",
             "gsm_think", "copy"):
    g = rd(os.path.join(ids, name + ".gen.i32"))
    for o in range(0, len(g) - 29, 30):
        streams.append((name, o, g[o:o + 30]))
x = rd(os.path.join(lane, "spec/ids/cross128.force.i32"))
for o in range(0, len(x) - 29, 30):
    streams.append(("cross128", o, x[o:o + 30]))
# one slice per source first, then the second slices, so the sixteen come from as many sources as there are
streams.sort(key=lambda s: (s[1], s[0]))
streams = streams[:16]
assert len(streams) == 16, len(streams)
prompts = ["gsm", "code", "copy", "story", "hebrew_para", "code_pal", "sys_json", "gsm_think", "capital", "hebrew",
           "multiturn", "needle", "gsm", "code", "copy", "story"]
pairs = []
for i, (name, o, toks) in enumerate(streams):
    f = os.path.join(out, f"f{i:02d}.i32")
    open(f, "wb").write(struct.pack(f"<{len(toks)}i", *toks))
    pairs.append(f"{ids}/{prompts[i]}.prompt.i32 {f}")
    print(i, prompts[i], name, o)
plan = os.path.join(lane, "spec/plans/wide9.plan")
with open(plan, "w") as fh:
    fh.write("batch-rows wide16 " + " ".join(pairs) + "\n")
    fh.write("batch-time distinct16 24 " + " ".join(pairs) + "\n")
print(plan)
# twenty sequences for a call past the small-gather bound of a two-rank group (the sixteen, then four prompts
# again on other streams): a byte gate only, the sequences need not be distinct
more = [f"{ids}/{prompts[i]}.prompt.i32 {os.path.join(out, f'f{i + 8:02d}.i32')}" for i in range(4)]
plan = os.path.join(lane, "spec/plans/wide20.plan")
with open(plan, "w") as fh:
    fh.write("batch-rows wide20 " + " ".join(pairs + more) + "\n")
print(plan)
