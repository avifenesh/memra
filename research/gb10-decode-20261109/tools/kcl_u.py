#!/usr/bin/env python3
"""kcl_u.py OUT: the GB10 cold-long K=2 vs K=3 A/B (cold_long_ab.py runs.jsonl, PROMPT_SET=gb10).
Per class and arm: each boot's median decode rate (tokens after the first content chunk over the time
after it) and TTFT; the k2 / k3 ratio of the per-arm medians of those boot medians; U = the number of
(k2 boot, k3 boot) pairs where k2 is faster (ties count half; 36 pairs at 6 boots per arm), and the
same for TTFT (k2 lower). Also the drift-normalized decode (over the boot's t1-short median) and the
K each arm's rows actually ran. Greedy texts must match across arms."""
import json, pathlib, statistics, sys

rows = [json.loads(l) for l in open(pathlib.Path(sys.argv[1]) / "runs.jsonl")]
CLASSES = [("T=1 1.3k", "t1-long", "-1k"), ("T=1 3.6k", "t1-long", "-4k"), ("T=1 5.8k", "t1-long", "-6k"),
           ("T=1 all long", "t1-long", ""), ("greedy 3.6k", "greedy-long", ""), ("T=1 short (reference)", "t1-short", "")]
boots = sorted({(r["boot"], r["arm"]) for r in rows})
ref = {b: statistics.median([r["decode_tok_s"] for r in rows if r["boot"] == b and r["class"] == "t1-short"]) for b, _ in boots}


def per_boot(cls, tag, key, norm=False):
    out = {"k2": [], "k3": []}
    for b, arm in boots:
        v = [r[key] for r in rows if r["boot"] == b and r["class"] == cls and r["prompt"].endswith(tag) and r[key]]
        if v:
            m = statistics.median(v)
            out[arm].append(m / ref[b] if norm else m)
    return out


def u(a, b, higher=True):
    s = 0.0
    for x in a:
        for y in b:
            s += 1.0 if (x > y if higher else x < y) else 0.5 if x == y else 0.0
    return s


print("| class | k3 decode (median of boot medians, range) | k2 decode | k2/k3 raw | U raw | k2/k3 normalized | U normalized | TTFT k2/k3 | U TTFT (k2 lower) | K ran (k3 / k2) |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
for name, cls, tag in CLASSES:
    d = per_boot(cls, tag, "decode_tok_s"); n = per_boot(cls, tag, "decode_tok_s", True); t = per_boot(cls, tag, "ttft_s")
    if not d["k2"] or not d["k3"]:
        continue
    ks = {arm: sorted({r["k"][0] for r in rows if r["arm"] == arm and r["class"] == cls and r["prompt"].endswith(tag) and r["k"]}) for arm in ("k3", "k2")}
    m3, m2 = statistics.median(d["k3"]), statistics.median(d["k2"])
    pairs = len(d["k2"]) * len(d["k3"])
    print(f"| {name} | {m3:.2f} ({min(d['k3']):.2f}-{max(d['k3']):.2f}) | {m2:.2f} ({min(d['k2']):.2f}-{max(d['k2']):.2f}) | {m2 / m3:.3f} | {u(d['k2'], d['k3']):g}/{pairs} | "
          f"{statistics.median(n['k2']) / statistics.median(n['k3']):.3f} | {u(n['k2'], n['k3']):g}/{pairs} | "
          f"{statistics.median(t['k2']) / statistics.median(t['k3']):.3f} | {u(t['k2'], t['k3'], False):g}/{pairs} | {','.join(ks['k3'])} / {','.join(ks['k2'])} |")
g = {arm: sorted({r["sha"] for r in rows if r["class"] == "greedy-long" and r["arm"] == arm}) for arm in ("k3", "k2")}
print(f"\nboots: {len(boots)} ({sum(1 for _, a in boots if a == 'k3')} k3, {sum(1 for _, a in boots if a == 'k2')} k2); greedy texts k3 {g['k3']} k2 {g['k2']}: {'identical' if g['k3'] == g['k2'] and len(g['k3']) == 1 else 'DIFFER'}")
