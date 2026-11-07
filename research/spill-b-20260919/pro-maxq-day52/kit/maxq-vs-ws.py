#!/usr/bin/env python3
"""Max-Q kit reading, no bound (PREREG.md 7): per boot, the Max-Q rows against the same boot's rows on the
Workstation receipt (DAY40.md 2.1). Sequential rows (every class but the burst) with status 200 on both cards and
equal prompt_sha256 compare message_sha256, completion tokens and finish_reason; the burst prints both cards' status
counts. A reading of cross-card identity; no verdict term reads it. Never edits its inputs.

usage: maxq-vs-ws.py <max-q boots dir> <workstation boots dir>
"""
import collections, json, os, sys

mq, ws = sys.argv[1], sys.argv[2]
BOOTS = ["O1-off", "O1-on2048", "O1-on8192", "O1-on32768", "O2-off", "O2-on2048", "O2-on8192", "O2-on32768"]


def rows(d):
    p = os.path.join(d, "client.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


tot = collections.Counter()
for b in BOOTS:
    a, w = rows(os.path.join(mq, b)), rows(os.path.join(ws, b))
    if a is None or w is None:
        print(f"MAXQ-VS-WS READING boot={b} missing={'maxq' if a is None else 'ws'}")
        continue
    wt = {r["tag"]: r for r in w}
    c = collections.Counter()
    first = None
    for r in a:
        if r["class"] == "burst":
            continue
        o = wt.get(r["tag"])
        c["seq"] += 1
        if o is None or r["status"] != 200 or o["status"] != 200:
            c["not_both_200"] += 1
            continue
        if r.get("prompt_sha256") != o.get("prompt_sha256"):
            c["prompt_differs"] += 1
            continue
        g = lambda x: (x.get("usage") or {}).get("completion_tokens")
        if r.get("message_sha256") == o.get("message_sha256") and g(r) == g(o) and r.get("finish_reason") == o.get("finish_reason"):
            c["equal"] += 1
        else:
            c["differ"] += 1
            first = first or r["tag"]
    bs = lambda rs: dict(sorted(collections.Counter(r["status"] for r in rs if r["class"] == "burst").items(), key=str))
    tot.update(c)
    print(f"MAXQ-VS-WS READING boot={b} seq_rows={c['seq']} compared={c['equal'] + c['differ']} equal={c['equal']} "
          f"differ={c['differ']} prompt_differs={c['prompt_differs']} not_both_200={c['not_both_200']} first_differ={first} "
          f"burst_maxq={bs(a)} burst_ws={bs(w)}")
print(f"MAXQ-VS-WS READING all compared={tot['equal'] + tot['differ']} equal={tot['equal']} differ={tot['differ']} "
      f"prompt_differs={tot['prompt_differs']} not_both_200={tot['not_both_200']} (no bound)")
