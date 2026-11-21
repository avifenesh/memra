"""Drafts on against drafts off: every answer of two chat_gate.py reports must be equal.

usage: python3 draft_identity.py ON_JSON OFF_JSON [OUT_JSON]

chat_gate.py compares answers inside one server, so with drafts on both sides of each of its
cells draft. This compares the report of a server with DFly2 drafts on with the report of the
same binary with `MEMRA_HY4F_DRAFT=0`: every served answer found at the same place in both
(token ids, text, content, reasoning, tool calls, stop and finish reasons) must be equal. That
covers greedy, the seeded sampled request (alone and beside others), the penalized request, the
stop string and the budget cut, the efforts, the tool call round trip and the concurrent sets.
Exits 1 on any difference or when a required cell holds no answer.
"""

import json
import sys

on = json.load(open(sys.argv[1]))
off = json.load(open(sys.argv[2]))
out = sys.argv[3] if len(sys.argv) > 3 else None

FIELDS = ("tokens", "text", "content", "reasoning", "tool_calls", "stop", "finish", "status")
REQUIRED = ("sampled", "penalties", "stops", "concurrent", "tools", "efforts", "stream", "long")

compared = {}
differ = []


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        answer = isinstance(a.get("tokens"), list) or "content" in a
        if answer:
            cell = path[0] if path else ""
            compared[cell] = compared.get(cell, 0) + 1
            for f in FIELDS:
                if a.get(f) != b.get(f):
                    differ.append({"at": "/".join(path), "field": f, "on": a.get(f), "off": b.get(f)})
        for k in a:
            if k in b and not (answer and k in FIELDS):
                walk(a[k], b[k], path + [k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differ.append({"at": "/".join(path), "field": "length", "on": len(a), "off": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [str(i)])


for cell in on:
    if cell in ("gates", "pass"):
        continue
    if cell not in off:
        differ.append({"at": cell, "field": "cell", "on": "present", "off": "missing"})
        continue
    walk(on[cell], off[cell], [cell])

missing = [c for c in REQUIRED if not compared.get(c)]
ok = not differ and not missing and bool(on.get("pass")) and bool(off.get("pass"))
result = {"pass": ok, "answers_compared": sum(compared.values()), "by_cell": compared,
          "missing_cells": missing, "differences": differ[:50], "on_gate_pass": on.get("pass"),
          "off_gate_pass": off.get("pass")}
if out:
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: result[k] for k in ("pass", "answers_compared", "by_cell", "missing_cells")}))
for d in differ[:10]:
    print("DIFFER", json.dumps(d, ensure_ascii=False)[:400])
print("DRAFT-IDENTITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
