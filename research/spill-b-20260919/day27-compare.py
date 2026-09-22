#!/usr/bin/env python3
"""Day 27: compare two cells of the day-26 mix request by request (client.jsonl): P, G, cached_tokens and the completion
sha256 per tag. Prints every differing tag verbatim and the counts. Usage: day27-compare.py <cell-A> <cell-B>"""
import json, os, sys

A, B = sys.argv[1], sys.argv[2]


def load(c):
    rows = {}
    for l in open(os.path.join(c, "client.jsonl")):
        r = json.loads(l)
        if r["arm"] == "warmup":
            continue
        u = r["usage"] or {}
        rows[r["tag"]] = dict(P=u.get("prompt_tokens"), G=u.get("completion_tokens"),
                              cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                              sha=r.get("content_sha256"), finish=r.get("finish_reason"), status=r["status"])
    return rows


a, b = load(A), load(B)
tags = sorted(set(a) | set(b), key=lambda t: (t.split("-")[0], t))
same_sha = diff_sha = missing = 0
for t in tags:
    ra, rb = a.get(t), b.get(t)
    if ra is None or rb is None:
        missing += 1; print(f"{t}: only in {'A' if rb is None else 'B'}"); continue
    if ra["sha"] == rb["sha"]:
        same_sha += 1
    else:
        diff_sha += 1
    if ra != rb:
        print(f"{t}: A P={ra['P']} G={ra['G']} cached={ra['cached']} finish={ra['finish']} sha={ra['sha'] and ra['sha'][:16]} | "
              f"B P={rb['P']} G={rb['G']} cached={rb['cached']} finish={rb['finish']} sha={rb['sha'] and rb['sha'][:16]}"
              f"{'  <- digest differs' if ra['sha'] != rb['sha'] else '  (digest equal)'}")
print(f"tags={len(tags)} digest_equal={same_sha} digest_differs={diff_sha} missing={missing}")
print(f"cached_tokens A: " + str([a[t]['cached'] for t in tags if t in a and t.startswith('iii')]))
print(f"cached_tokens B: " + str([b[t]['cached'] for t in tags if t in b and t.startswith('iii')]))
