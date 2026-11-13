#!/usr/bin/env python3
"""Build spec_oracle.py: the lane's oracle runner up to its case loop (loader, device placement, head patch) with a
--replay argument, then spec_tail.py. The shared oracle is read, never edited.

usage: make_spec_oracle.py RUN_ORACLE.py SPEC_TAIL.py OUT.py"""
import sys

src, tail, out = sys.argv[1:4]
head = open(src).read()
cut = "M.ParallelHead.forward = _head\n"
assert head.count(cut) == 1, "the oracle's head patch line moved"
head = head[: head.index(cut) + len(cut)]
arg = 'ap.add_argument("--index-qchunk", type=int, default=0)'
assert head.count(arg) == 1, "the oracle's argument list moved"
line_end = head.index("\n", head.index(arg)) + 1
head = (head[:line_end]
        + 'ap.add_argument("--replay", default="")  # DSpark draft tapes: replay a tape dir through forward_spec alone (spec tail)\n'
        + head[line_end:])
# the manifest hashes the files this copy runs
head = head.replace('("kernel.py", "run_oracle.py", "long_ctx.py")', '("kernel.py", "spec_oracle.py", "long_ctx.py")')
open(out, "w").write(head + open(tail).read())
print(f"{out}: {len(head.splitlines())} lines of the oracle + the tail")
