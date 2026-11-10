#!/usr/bin/env python3
"""The end-of-request cell side by side (memra#1186): one line per (mode, class, kind) of
close_cell.py's rows, base against candidate.

Judged: the text sha, the finish reason and the prompt, cached and completion tokens must be
equal in the two arms (a hung-up request has no usage and no judged text: where the client
stops reading is a matter of timing). Reported, not judged: the speculative rounds (the
candidate draws the token after a burst boundary in a round, so its rounds differ) and
close_ms, the wall from the last content chunk to the end of the response.

usage: close_summary.py ROWS.jsonl [ROWS.jsonl ...] [BASE CAND]
exit 0 when every judged field is equal on every row.
"""
import json
import sys

files = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
names = [a for a in sys.argv[1:] if not a.endswith(".jsonl")]
base, cand = names if len(names) == 2 else ("base", "cand")
rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
keys = []
for r in rows:
    k = (r["mode"], r["class"], r["kind"])
    if k not in keys:
        keys.append(k)
JUDGED = ("sha", "finish", "prompt_tokens", "cached", "completion_tokens")
print("| mode | class | request | finish | prompt / cached / completion tokens | text | rounds base | cand | close ms base | cand |")
print("|---|---|---|---|---|---|---:|---:|---:|---:|")
bad = 0
for k in keys:
    b = [r for r in rows if (r["mode"], r["class"], r["kind"]) == k and r["arm"] == base]
    c = [r for r in rows if (r["mode"], r["class"], r["kind"]) == k and r["arm"] == cand]
    if not b or not c:
        print(f"| {k[0]} | {k[1]} | {k[2]} | one arm only | | | | | | |")
        bad += 1
        continue
    b, c = b[-1], c[-1]
    if b["error"] or c["error"]:
        print(f"| {k[0]} | {k[1]} | {k[2]} | error: {b['error'] or c['error']} | | | | | | |")
        bad += 1
        continue
    hung = k[2] == "cancelled"
    same = hung or all(b[f] == c[f] for f in JUDGED)
    bad += not same
    toks = "hung up after 40 chunks" if hung else (
        f"{b['prompt_tokens']} / {b['cached']} / {b['completion_tokens']}" if same else
        f"{b['prompt_tokens']} / {b['cached']} / {b['completion_tokens']} against "
        f"{c['prompt_tokens']} / {c['cached']} / {c['completion_tokens']}")
    text = "not judged" if hung else ("equal" if b["sha"] == c["sha"] else f"DIFFERS {b['sha']} {c['sha']}")
    print(f"| {k[0]} | {k[1]} | {k[2]} | {b['finish'] if b['finish'] == c['finish'] else str(b['finish']) + ' against ' + str(c['finish'])} | "
          f"{toks} | {text} | {b['rounds']} | {c['rounds']} | {b['close_ms']} | {c['close_ms']} |")
print()
print(f"rows: {len(keys)}, with a judged difference: {bad}")
sys.exit(1 if bad else 0)
