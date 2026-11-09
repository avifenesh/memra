#!/usr/bin/env python3
"""Which pool path each agentic request took, from the server log (memra#1093 follow-up).

The worker prints `[meter] admit id=<id>` before it books, reclaims and admits a request, all on one
thread, so every admission line up to the next `[meter] admit` belongs to that request whatever the
concurrency. Park lines (`dspark-park ...`) are printed when a session retires and land in whichever
segment is open then; they are counted per log, not per request.

  python3 trace_requests.py SERVE_LOG [REQUESTS_JSONL ...]

Without request rows: one line per admission. With them: each timed request's path, joined on the
response id, and a summary per session turn (first or second body of an agent session).
"""
import collections
import json
import re
import sys

EVENTS = {
    "resume": re.compile(r"dspark-reuse: (\d+) committed tokens resumed at the prime boundary \(parked at (\d+)\)"),
    "grow": re.compile(r"dspark-reuse: grew the rewound session from (\d+) to (\d+) rows"),
    "miss": re.compile(r"dspark-reuse MISS: (\d+) parked, best text overlap (\d+)B of (\d+)B"),
    "seat_decline": re.compile(r"dspark-reuse DECLINED: the rewound session cannot be seated"),
    "suffix_decline": re.compile(r"dspark-reuse DECLINED: suffix (\d+) < prime floor"),
    "evicted": re.compile(r"reclaim-on-defer: evicted (\d+) prefix entries \+ (\d+) plain \+ (\d+) spec \+ (\d+) dspark"),
    "cache_preferred": re.compile(r"\[dspark\] cache-preferred: prompt=(\d+)"),
    "prefix_hit": re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens"),
    "plain_resume": re.compile(r"kv-reuse: (\d+) of (\d+) prompt tokens resumed"),
    "cost": re.compile(r"request cost: model=\"[^\"]+\" ctx=(\d+) path=(\w+)"),
    "speck": re.compile(r"\[spec-k\] .*prompt=(\d+) cached=(\d+)"),
}
PARKS = {
    "park_rewound": re.compile(r"dspark-park REWOUND \(grid law\)"),
    "park_skipped": re.compile(r"dspark-park SKIPPED \(grid law\)"),
}
ADMIT = re.compile(r"\[meter\] admit id=(\S+)")


def segments(path):
    segs, cur, parks = {}, None, collections.Counter()
    order = []
    for line in open(path, errors="replace"):
        m = ADMIT.search(line)
        if m:
            cur = collections.defaultdict(list)
            segs[m.group(1)] = cur
            order.append(m.group(1))
            continue
        for k, rx in PARKS.items():
            if rx.search(line):
                parks[k] += 1
        if cur is None:
            continue
        for k, rx in EVENTS.items():
            m = rx.search(line)
            if m:
                cur[k].append(m.groups())
    return segs, order, parks


def path_of(seg):
    if seg is None:
        return "no-admit-line"
    if seg.get("resume"):
        return "dspark-resume" + ("+grow" if seg.get("grow") else "")
    if seg.get("seat_decline"):
        return "dspark-seat-declined"
    if seg.get("cache_preferred"):
        return "plain-cache-preferred"
    if seg.get("plain_resume"):
        return "plain-pool"
    if seg.get("prefix_hit"):
        return "prefix-hit"
    if seg.get("miss"):
        return "dspark-miss"
    return "cold"


def main():
    segs, order, parks = segments(sys.argv[1])
    if len(sys.argv) == 2:
        for i in order:
            print(i, path_of(segs[i]), dict(segs[i]))
        print(json.dumps(dict(parks)))
        return
    per_turn = collections.defaultdict(collections.Counter)
    for f in sys.argv[2:]:
        for line in open(f):
            r = json.loads(line)
            seg = segs.get(r["id"]) if r.get("id") else None
            p = path_of(seg) if r.get("ok") else f"http-{r.get('status')}"
            own = ""
            per_turn[r["session_turn"]][p] += 1
            if own:
                per_turn[r["session_turn"]][own] += 1
            print(json.dumps({"file": f.rsplit("/", 1)[-1], "k": r["k"], "turn": r["session_turn"], "path": p,
                              "own": own, "ttft_s": r.get("ttft_s"), "wall_s": round(r["wall_s"], 3),
                              "prompt": (r.get("usage") or {}).get("prompt_tokens")}))
    print(json.dumps({"per_turn": {t: dict(c) for t, c in sorted(per_turn.items())}, "parks": dict(parks)}))


if __name__ == "__main__":
    main()
