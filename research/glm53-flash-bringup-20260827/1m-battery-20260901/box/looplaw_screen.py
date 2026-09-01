#!/usr/bin/env python3
"""LOOP-LAW SCREEN (owner law, agent-knowledge greedy-is-the-instrument-not-the-product).

Greedy argmax has no escape randomness, so repetition self-reinforces and locks. A greedy
loop is a KNOWN ARTIFACT, never a finding and never a model bug - but looped output must
NEVER enter a perf row, because loops repeat cheap high-accept tokens and inflate BOTH
tok/s and acceptance. This screen flags any rung whose output degenerated, so it can be
reported SEPARATELY with the exclusion named in the receipt.

Detectors (any one flags the row):
  * a tail n-gram (n=6..12) repeated >= 4 times consecutively
  * the most common line repeated >= 4 times
  * distinct-token ratio over the last 120 whitespace tokens below 0.30
usage: looplaw_screen.py <dir-of-rung-jsons> [more dirs...]
"""
import json
import pathlib
import sys
from collections import Counter


def looped(text: str):
    if not text or len(text.split()) < 24:
        return None
    w = text.split()
    for n in range(6, 13):
        if len(w) < n * 4:
            continue
        tail = w[-n:]
        reps = 1
        i = len(w) - 2 * n
        while i >= 0 and w[i:i + n] == tail:
            reps += 1
            i -= n
        if reps >= 4:
            return f"tail {n}-gram x{reps}"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        ln, c = Counter(lines).most_common(1)[0]
        if c >= 4:
            return f"line x{c}: {ln[:50]!r}"
    last = w[-120:]
    if len(last) >= 60:
        ratio = len(set(last)) / len(last)
        if ratio < 0.30:
            return f"distinct-token ratio {ratio:.2f} over last {len(last)}"
    return None


rows, flagged = 0, []
for d in sys.argv[1:]:
    for p in sorted(pathlib.Path(d).rglob("*.json")):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if "output" not in j or "label" not in j:
            continue
        rows += 1
        # screen the decoded surface the perf number was read off: reasoning + content
        why = looped((j.get("reasoning") or "") + "\n" + (j.get("output") or ""))
        if why:
            u = j.get("usage") or {}
            flagged.append((str(p), j["label"], j.get("mode"), u.get("completion_tokens"), why))

print(f"LOOP-LAW SCREEN: {len(flagged)} flagged of {rows} screened")
for p, label, mode, ct, why in flagged:
    print(f"  FLAGGED {label}/{mode} ct={ct}: {why}")
    print(f"    {p}")
    print("    -> EXCLUDED from aggregates, reported separately (owner loop-law)")
if not flagged:
    print("  no degenerate rows: every rung's decode number is admissible")
