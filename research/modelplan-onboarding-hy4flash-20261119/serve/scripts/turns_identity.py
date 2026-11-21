"""Real chat turns, two servers: every answer of two real_turns.py outputs must be equal.

usage: python3 turns_identity.py OFF_JSON OTHER_JSON [OUT_JSON] [--rounds N]

OFF_JSON is the drafts-off server's turns; OTHER_JSON a drafting server's, or a server's whose
drafts were refused by `MEMRA_HY4F_DRAFT_FAULT`. Each turn's status, content, reasoning, finish
reason and token counts must be equal. `--rounds N` also requires every turn of OTHER_JSON to have
verified exactly N draft rounds: the fault door's `draft:N` and `backup:N` both end a request's
drafting after N rounds, so the rest of each answer came from the fallback's plain calls.
Exits 1 on any difference.
"""

import json
import sys

argv = sys.argv[1:]
rounds = None
if "--rounds" in argv:
    at = argv.index("--rounds")
    rounds = int(argv[at + 1])
    del argv[at:at + 2]
off = json.load(open(argv[0]))
other = json.load(open(argv[1]))
out = argv[2] if len(argv) > 2 else None

FIELDS = ("status", "content", "reasoning", "finish", "prompt_tokens", "completion_tokens", "error")
rows = []
ok = len(off) == len(other) and len(off) > 0
for x, y in zip(off, other):
    equal = x["turn"] == y["turn"] and all(x.get(f) == y.get(f) for f in FIELDS)
    served = x.get("status") == 200 and bool(x.get("content") or x.get("reasoning"))
    spec = y.get("spec") or {}
    rounds_ok = rounds is None or spec.get("rounds") == rounds
    rows.append({"turn": x["turn"], "equal": equal, "served": served, "rounds_ok": rounds_ok,
                 "completion_tokens": x.get("completion_tokens"), "wall_s_off": x.get("wall_s"),
                 "wall_s_other": y.get("wall_s"), "spec": y.get("spec")})
    ok &= equal and served and rounds_ok
result = {"pass": ok, "want_rounds": rounds, "turns": rows}
if out:
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
for r in rows:
    print(json.dumps(r))
print("TURNS-IDENTITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
