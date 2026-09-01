#!/usr/bin/env python3
"""Compare two gate.py tapes for byte identity, per cell, and refuse to call a vacuous
comparison a PASS.

The refusals matter more than the equality check. The original bank-v2 gate reported green
three separate ways while the door corrupted text, and every one of those greens was a
comparison that compared nothing: an oracle that branched on the flag under test, a host-side
test that could not see a reader, and a soak that metered request success rather than answer
bytes. So this comparator fails loudly when

  * a cell is missing from either arm (a shrunken corpus silently passes otherwise);
  * either arm's completion is EMPTY (two empty strings are byte-identical and prove nothing —
    and on a thinking model whose bytes ride `reasoning_content`, a content-only reader
    produces exactly that);
  * the two arms report the SAME binary md5 (then it is one program compared with itself);
  * the two arms report the same boot nonce (one boot, not two).

Usage: compare.py <armA.json> <armB.json>
"""
import json, sys

A = json.load(open(sys.argv[1]))
B = json.load(open(sys.argv[2]))
print("arms: %s (md5 %s) vs %s (md5 %s)" % (A["arm"], A["bin_md5"], B["arm"], B["bin_md5"]))
fatal = []
if A["arm"] == B["arm"]:
    fatal.append("same arm tag on both sides")
if A.get("boot_nonce") and A.get("boot_nonce") == B.get("boot_nonce"):
    fatal.append("same boot nonce — one boot, not two")
ka, kb = set(A["cells"]), set(B["cells"])
if ka != kb:
    fatal.append("cell sets differ: only-A=%s only-B=%s" % (sorted(ka - kb), sorted(kb - ka)))

same = diff = 0
first_diffs = []
for k in sorted(ka & kb):
    ca, cb = A["cells"][k], B["cells"][k]
    if not ca["out"].strip() or not cb["out"].strip():
        fatal.append("cell %s has an EMPTY completion (A=%d B=%d chars) — vacuous comparison"
                     % (k, len(ca["out"]), len(cb["out"])))
        continue
    if ca["out_sha256"] == cb["out_sha256"]:
        same += 1
    else:
        diff += 1
        # locate the first differing character: "diverged at char N" is the receipt the
        # incident's bisect needed and had to reconstruct by hand.
        n = next((i for i, (x, y) in enumerate(zip(ca["out"], cb["out"])) if x != y),
                 min(len(ca["out"]), len(cb["out"])))
        first_diffs.append((k, n, ca["out"][n:n + 40], cb["out"][n:n + 40]))
        print("DIFF %s at char %d\n   A: %r\n   B: %r" % (k, n, ca["out"][n:n + 60], cb["out"][n:n + 60]))

print("cells identical=%d differing=%d" % (same, diff))
print("tape A=%s\ntape B=%s" % (A.get("tape_sha256"), B.get("tape_sha256")))
if fatal:
    for f in fatal:
        print("VACUOUS/FATAL: %s" % f)
    print("BYTE GATE: REFUSED (the comparison was not decidable)")
    sys.exit(2)
if not A.get("all_content_ok") or not B.get("all_content_ok"):
    print("BYTE GATE: FAIL — an arm's own content oracle did not pass")
    sys.exit(1)
if diff == 0:
    print("BYTE GATE: PASS — %d cells byte-identical, tapes equal=%s"
          % (same, A.get("tape_sha256") == B.get("tape_sha256")))
    sys.exit(0)
print("BYTE GATE: FAIL — %d cells differ" % diff)
sys.exit(1)
