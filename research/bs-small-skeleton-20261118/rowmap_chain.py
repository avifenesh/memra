#!/usr/bin/env python3
"""The rebase onto memra#1337's rebased head: the per-kernel digests of three sass_cmp.py runs, chained.

usage: rowmap_chain.py NEW_PAIR.tsv BASES.tsv HEADS.tsv
  NEW_PAIR  the new base (17c6b4fd75) against the rebased head
  BASES     the old base (801243f06e) against the new base
  HEADS     the old head (e658b44cd8) against the rebased head
Checks, kernel for kernel: the old base and the old head have the same body and resource line (the pre-rebase
proof again); the new base and the rebased head likewise (NEW_PAIR); and every kernel that moved between the old
and the new base moved the same way between the old and the new head, to the same body. A kernel that differs
from old to new is then a change of main's, carried exactly.
"""
import csv
import sys


def load(f):
    return {(r["unit"], r["kernel"]): r for r in csv.DictReader(open(f), delimiter="\t")}


A, B, C = (load(f) for f in sys.argv[1:4])
bad = 0
assert set(A) == set(B) == set(C), "the three runs name different kernels"
for k in sorted(A):
    a, b, c = A[k], B[k], C[k]
    old_eq = b["base_sha256"] == c["base_sha256"] and b["base_res"] == c["base_res"]
    new_eq = a["equal"] == "1" and b["new_sha256"] == a["base_sha256"] and c["new_sha256"] == a["new_sha256"]
    same_move = b["equal"] == c["equal"]
    if not (old_eq and new_eq and same_move):
        bad += 1
        print(f"BROKEN {k[0]} {k[1]}: old pair equal {old_eq}, new pair equal {new_eq}, same move {same_move}")
    elif b["equal"] == "0":
        print(f"CHANGED-ON-MAIN {k[0]} {k[1]}: {b['base_sha256'][:16]} -> {b['new_sha256'][:16]}; "
              f"res {b['base_res']} -> {b['new_res']}")
print(f"ROWMAP-CHAIN {'OK' if bad == 0 else 'BROKEN'}: {len(A)} kernels, {bad} broken")
sys.exit(1 if bad else 0)
