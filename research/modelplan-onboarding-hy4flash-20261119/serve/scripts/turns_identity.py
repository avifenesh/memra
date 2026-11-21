"""Real chat turns, two servers: every answer of two real_turns.py outputs must be equal.

usage: python3 turns_identity.py OFF_JSON OTHER_JSON [--out OUT_JSON] [--rounds N --log SERVER_LOG]

OFF_JSON is the drafts-off server's turns; OTHER_JSON a drafting server's, or a server's whose
drafts were refused by `MEMRA_HY4F_DRAFT_FAULT`. Each turn's status, content, reasoning, finish
reason and token counts must be equal. `--rounds N` is the fault door's check: `draft:N` and
`backup:N` both end a request's drafting after N verified rounds, so every turn of OTHER_JSON
must report exactly N rounds, must have at least two tokens that no round produced (the
fallback's plain calls), and SERVER_LOG must hold one refusal by the door for every turn.
Exits 1 on any difference.
"""

import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("off")
ap.add_argument("other")
ap.add_argument("--out")
ap.add_argument("--rounds", type=int)
ap.add_argument("--log")
a = ap.parse_args()
if (a.rounds is None) != (a.log is None):
    ap.error("--rounds and --log go together")
off = json.load(open(a.off))
other = json.load(open(a.other))

FIELDS = ("status", "content", "reasoning", "finish", "prompt_tokens", "completion_tokens", "error")
rows = []
ok = len(off) == len(other) and len(off) > 0
for x, y in zip(off, other):
    equal = x["turn"] == y["turn"] and all(x.get(f) == y.get(f) for f in FIELDS)
    served = x.get("status") == 200 and bool(x.get("content") or x.get("reasoning"))
    spec = y.get("spec") or {}
    row = {"turn": x["turn"], "equal": equal, "served": served,
           "completion_tokens": x.get("completion_tokens"), "wall_s_off": x.get("wall_s"),
           "wall_s_other": y.get("wall_s"), "spec": y.get("spec")}
    good = equal and served
    if a.rounds is not None:
        # tokens: one from the prefill's last row, one a round plus its kept drafts, the rest plain
        plain = (y.get("completion_tokens") or 0) - 1 - spec.get("rounds", 0) - spec.get("accepted", 0)
        row["rounds_ok"] = spec.get("rounds", 0) == a.rounds
        row["plain_tokens_after"] = plain
        good = good and row["rounds_ok"] and plain >= 2
    rows.append(row)
    ok = ok and good
result = {"pass": ok, "want_rounds": a.rounds, "turns": rows}
if a.log is not None:
    refused = sum("refused by MEMRA_HY4F_DRAFT_FAULT" in line for line in open(a.log, errors="replace"))
    result["refusals_logged"] = refused
    ok = ok and refused == len(other)
    result["pass"] = ok
if a.out:
    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
for r in rows:
    print(json.dumps(r))
if a.log is not None:
    print("refusals logged:", result["refusals_logged"], "of", len(other), "turns")
print("TURNS-IDENTITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
