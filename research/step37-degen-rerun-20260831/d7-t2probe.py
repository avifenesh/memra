#!/usr/bin/env python3
# ADDENDUM to the re-run (recorded in PLAN-DIFF.md before any generation): reproduce the
# original lane's stop-inside-think denominator at n=8.
#
# The original quoted "turn-2 5/8 attempts across passes": its clean-transcript builder
# accepts and BREAKS on the first qualifying attempt, so a single build pass can never
# emit 8 turn-2 attempts. The 8 came from two separate build passes (PLAN.md deviations 1
# and 2). This probe fires the SAME request shape as the builder -- same turn-2 history
# (U1 + A1 + U2 from the freshly built clean bank), vendor-default sampled, no early
# break -- for exactly 8 attempts on the maxtok schedule [4096, 4096, 4096, 8192] x2, and
# banks every attempt. Nothing else about the cell changes; d7-drive.py is byte-verbatim
# and this file only imports it, so the request path is literally the same code.
import importlib.util, json, os, sys

LANE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d7drive", LANE + "/d7-drive.py")
d7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d7)

bank = json.load(open(LANE + "/raw/transcript-clean.json"))
U = {int(k): v for k, v in bank["U"].items()}
A = {int(k): v for k, v in bank["A"].items()}
out = open(LANE + "/raw/t2-stopinthink-probe.jsonl", "a", buffering=1)
done = 0
if os.path.exists(LANE + "/raw/t2-stopinthink-probe.jsonl"):
    done = sum(1 for _ in open(LANE + "/raw/t2-stopinthink-probe.jsonl"))
sched = [4096, 4096, 4096, 8192, 4096, 4096, 4096, 8192]
for ai in range(done + 1, 9):
    mt = sched[ai - 1]
    res, err = d7.guarded(d7.msgs_through(U, A, 2), "d7RR-t2probe-a%d" % ai, mt)
    row = dict(turn=2, attempt=ai, maxtok=mt, err=err or (res and res["err"]),
               finish=res and res["finish"],
               reasoning_chars=res and len(res["reasoning"]),
               content_chars=res and len(res["content"]),
               stop_inside_think=bool(res and res["finish"] == "stop"
                                      and len(res["content"].strip()) < 200
                                      and len(res["reasoning"].strip()) >= 200),
               ttft=res and res["ttft"], total=res and res["total"],
               spec=d7.spec_of(res), panic=res and res["panic"],
               reasoning=res and res["reasoning"], content=res and res["content"])
    out.write(json.dumps(row) + "\n")
    print("T2PROBE a%d maxtok=%d finish=%s content=%s reasoning=%s stop_inside_think=%s"
          % (ai, mt, row["finish"], row["content_chars"], row["reasoning_chars"],
             row["stop_inside_think"]), flush=True)
rows = [json.loads(l) for l in open(LANE + "/raw/t2-stopinthink-probe.jsonl")]
n = len(rows)
hits = sum(1 for r in rows if r["stop_inside_think"])
print("T2PROBE_DONE stop_inside_think=%d/%d" % (hits, n), flush=True)
