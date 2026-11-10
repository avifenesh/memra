#!/usr/bin/env python3
"""red_worker.py BASE_WORKER HEAD_WORKER OUT: the fail-before tree of the spec boundary census.
Writes BASE's crates/memra-server/src/worker.rs with HEAD's spec boundary publish tests added to
its test module: the fixture, the census, the copy-program cell and the budget cell. Production
code is untouched. The failed-copy cell is left out: it drives a test fault only the head has."""
import sys

base, head = (open(p).read() for p in sys.argv[1:3])
out = sys.argv[3]
A = "    /// GPU-only fixture of the spec boundary publication cells"
M = "    /// GPU cell: a plane whose copy failed is never published."
B = "    /// GPU-only: one plane of `len` rows on the engine's owner stream with known bytes."
assert head.count(A) == 1 and head.count(M) == 1 and base.count(B) == 1 and A not in base
block = head[head.index(A):head.index(M)]
open(out, "w").write(base.replace(B, block + B))
