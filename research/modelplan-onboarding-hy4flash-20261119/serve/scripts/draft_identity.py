"""Drafts on against drafts off: every answer of two chat_gate.py reports must be equal.

usage: python3 draft_identity.py ON_JSON OFF_JSON [OUT_JSON]

chat_gate.py compares answers inside one server, so with drafts on both sides of each of its
cells draft. This compares the report of a server with DFly2 drafts on with the report of the
same binary with `MEMRA_HY4F_DRAFT=0`: every served answer found at the same place in both
(token ids, text, content, reasoning, tool calls, stop and finish reasons, status) must be equal,
and an answer, a key or a cell on one side only is a difference. That covers greedy, the seeded
sampled request (alone and beside others), the penalized request, the stop string and the budget
cut, the efforts, the tool call round trip and the concurrent sets. Timings, the served context
and the server's log lines are not answers and are not compared. Exits 1 on any difference or
when a required cell holds no answer.
"""

import json
import sys

on = json.load(open(sys.argv[1]))
off = json.load(open(sys.argv[2]))
out = sys.argv[3] if len(sys.argv) > 3 else None

FIELDS = ("tokens", "text", "content", "reasoning", "tool_calls", "stop", "finish", "status")
# the server's log lines kept in the report: not answers
LOG_KEYS = ("admits", "waits", "last_step")
REQUIRED = ("cases", "solo", "sampled", "penalties", "stops", "concurrent", "tools", "efforts", "stream",
            "cancel", "admission", "long")

compared = {}
differ = []


def is_answer(x):
    return isinstance(x, dict) and (isinstance(x.get("tokens"), list) or "content" in x
                                    or "status" in x or "error" in x)


def note(path, field, a, b):
    differ.append({"at": "/".join(path), "field": field, "on": a, "off": b})


def walk(a, b, path):
    """Both sides are walked together: a place that holds an answer on either side is an answer,
    and a key, an item or a kind present on one side only is a difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        answer = is_answer(a) or is_answer(b)
        if answer:
            cell = path[0] if path else ""
            compared[cell] = compared.get(cell, 0) + 1
            for f in FIELDS:
                if a.get(f) != b.get(f):
                    note(path, f, a.get(f), b.get(f))
            if ("error" in a) != ("error" in b) or bool(a.get("error")) != bool(b.get("error")):
                note(path, "error", a.get("error"), b.get("error"))
        for k in sorted(set(a) | set(b)):
            if k in LOG_KEYS:
                continue
            if k not in a or k not in b:
                note(path + [k], "key", "present" if k in a else "missing", "present" if k in b else "missing")
            elif not (answer and k in FIELDS):
                walk(a[k], b[k], path + [k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            note(path, "length", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [str(i)])
    elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)) or (a is None) != (b is None):
        note(path, "kind", type(a).__name__, type(b).__name__)


for cell in sorted(set(on) | set(off)):
    if cell in ("gates", "pass"):
        continue
    if cell not in on or cell not in off:
        note([cell], "cell", "present" if cell in on else "missing", "present" if cell in off else "missing")
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
