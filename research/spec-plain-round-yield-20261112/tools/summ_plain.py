#!/usr/bin/env python3
"""Tables of the plain cell (memra#1244) and the text checks.
usage: summ_plain.py RUNDIR [RUNDIR...]   (each holds plain.jsonl and texts/)
Arms are row["arm"] with a "-N" boot suffix removed. Rows are grouped by (long streams, long
class, plain class); the plain requests of all boots of an arm are pooled.
Text checks: a plain request's text is keyed by (cell, n) and must be equal across every boot
and arm, beside the streams and alone (solo request n repeats the prompt of request n, so one
key holds both); for each (cell, stream) the long texts of all boots must be prefixes of the
longest one."""
import collections
import json
import pathlib
import re
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


rows = []
for d in sys.argv[1:]:
    d = pathlib.Path(d)
    for l in open(d / "plain.jsonl"):
        if l.strip():
            r = json.loads(l)
            r["_dir"] = d
            r["_arm"] = re.sub(r"-\d+$", "", r["arm"])
            rows.append(r)

groups = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    groups[(r["longs"], r["class"], r["plain_class"])][r["_arm"]].append(r)

print("| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | "
      "E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |")
print("|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|")
for key in sorted(groups):
    for arm, rs in sorted(groups[key].items()):
        news = [x for r in rs for x in r["news"]]
        solos = [x for r in rs for x in r["solos"]]
        tt = [x["ttft_s"] for x in news if x["ttft_s"] is not None]
        gaps = [g for x in news for g in x["gaps_s"]]
        sg = [g for x in solos for g in x["gaps_s"]]
        e2e = [x["e2e_s"] for x in news]
        live = sum(r["live_s"] for r in rs)
        ptok = sum(x["chunks"] for x in news)
        ltok = sum(s["tokens_live"] for r in rs for s in r["long_streams"])
        idle = [r["long_rate_idle"] for r in rs]
        lg = max((s["longest_gap_in_window_s"] for r in rs for s in r["long_streams"]), default=0)
        ended = sum(r["longs_ended_in_window"] for r in rs)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {arm} | {len(rs)} | {len(news)} | {pct(tt, .5):.3f} / {max(tt):.3f} | "
              f"{pct(gaps, .5):.3f} / {pct(gaps, .95):.3f} / {max(gaps):.3f} | {pct(e2e, .5):.2f} / {max(e2e):.2f} | "
              f"{ptok / live:.2f} | {ltok / live:.2f} | {statistics.mean(idle):.2f} | {lg:.3f} | "
              f"{pct(sg, .5):.3f} | {' '.join(f'{x['e2e_s']:.2f}' for x in solos)} |"
              + (f" ({ended} long ended in window)" if ended else ""))

# long prime (PAD > 0): time to the first token and the streams' rate until then
padded = [r for r in rows if r.get("pad")]
if padded:
    print("\n| long streams | arm | boot | request | TTFT s | streams tok/s until its first token | alone TTFT s |")
    print("|---:|---|---|---:|---:|---:|---|")
    for r in sorted(padded, key=lambda r: (r["longs"], r["_arm"], r["arm"])):
        for x in r["news"]:
            print(f"| {r['longs']} | {r['_arm']} | {r['arm']} | {x['n']} | {x['ttft_s']:.2f} | {x.get('long_rate_to_first', float('nan')):.2f} | "
                  f"{' '.join(f'{y['ttft_s']:.2f}' for y in r['solos'])} |")

# text checks
bad = 0
plain_sha = collections.defaultdict(set)
for r in rows:
    for kind in ("news", "solos"):
        for x in r[kind]:
            # padded requests carry their own filler per kind, so beside and alone are two keys
            plain_sha[((r["longs"], r["class"], r["plain_class"]), x["n"], kind if r.get("pad") else "")].add(
                (x["sha"], x["chunks"]))
            if x["error"]:
                bad += 1
                print("ERROR", r["arm"], r["cell"], kind, x["n"], x["error"])
mixed = {k: v for k, v in plain_sha.items() if len(v) > 1}
print(f"\nplain texts: {len(plain_sha)} (cell, n) keys over {sum(len(r['news']) + len(r['solos']) for r in rows)} requests "
      f"(beside streams and solo, all arms): {'IDENTICAL' if not mixed else 'DIFFER ' + str(mixed)}")
longs = collections.defaultdict(list)
for r in rows:
    for s in r["long_streams"]:
        longs[(r["longs"], r["class"], r["plain_class"], s["stream"])].append((r["_dir"] / s["text_file"]).read_text())
fail = 0
for k, texts in longs.items():
    longest = max(texts, key=len)
    for t in texts:
        if not longest.startswith(t):
            fail += 1
            print("PREFIX FAIL", k, len(t), len(longest))
print(f"long texts: {sum(len(v) for v in longs.values())} texts over {len(longs)} (cell, stream) keys: "
      f"{'PREFIX PASS' if not fail else f'{fail} PREFIX FAIL'}")
sys.exit(1 if (mixed or fail or bad) else 0)
