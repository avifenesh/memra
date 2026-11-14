#!/usr/bin/env python3
"""arm_b.py CTX_BOOT_RS: make the red arm of memra#1316's cells. Sets the serving policy's retry
budget to zero in a copy of the tree's ctx_boot.rs, so a build of it creates the context once, as
before the change, and still names the failing call and prints the host's memory. Nothing else
differs between the arms."""
import re
import sys

p = sys.argv[1]
s = open(p).read()
head, sep, tail = s.partition("pub const RECLAIM_RETRY: ReclaimRetry = ReclaimRetry {")
if not sep:
    sys.exit("arm_b: RECLAIM_RETRY not found")
tail, n = re.subn(r"budget: [^\n]*,\n", "budget: Duration::ZERO,\n", tail, count=1)
if n != 1:
    sys.exit("arm_b: budget line not found")
open(p, "w").write(head + sep + tail)
print("arm_b: budget set to zero")
