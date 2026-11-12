#!/usr/bin/env python3
"""Tables of the riders cell (memra#1244 review) and the text checks.
usage: summ_riders.py RUNDIR [RUNDIR...]   (each holds riders.jsonl and texts/)
Arms are row["arm"] with a "-N" boot suffix removed. Rows are grouped by (long streams, riders,
plain class). Text checks: a plain request's text is keyed by (riders, class, n) and must be
equal across every boot and arm, beside the streams and alone; the long texts of a (riders,
class, stream) key must be prefixes of the longest one."""
import collections
import json
import pathlib
import re
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


rows = []
for d in sys.argv[1:]:
    d = pathlib.Path(d)
    for line in open(d / "riders.jsonl"):
        if line.strip():
            r = json.loads(line)
            r["_dir"] = d
            r["_arm"] = re.sub(r"-\d+$", "", r["arm"])
            rows.append(r)

groups = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    groups[(r["longs"], r["riders"], r["plain_class"])][r["_arm"]].append(r)
print("| long streams | riders | class | arm | boots | window s | streams tok/s before | streams tok/s live | "
      "riders tok/s live | all tok/s live | rider gap p50 / p95 s | rider TTFT p50 s | rider E2E max s | riders alone tok/s |")
print("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
for key in sorted(groups):
    for arm, rs in sorted(groups[key].items()):
        live = sum(r["live_s"] for r in rs)
        lt = sum(s["tokens_live"] for r in rs for s in r["long_streams"])
        pt = sum(x["chunks"] for r in rs for x in r["news"])
        gaps = [g for r in rs for x in r["news"] for g in x["gaps_s"]]
        tt = [x["ttft_s"] for r in rs for x in r["news"] if x["ttft_s"] is not None]
        before = sum(r["long_rate_before"] for r in rs) / len(rs)
        alone = sum(r["solo_tok_s"] for r in rs) / len(rs)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {arm} | {len(rs)} | {live:.1f} | {before:.2f} | {lt / live:.2f} | "
              f"{pt / live:.2f} | {(lt + pt) / live:.2f} | {pct(gaps, .5):.3f} / {pct(gaps, .95):.3f} | {pct(tt, .5):.3f} | "
              f"{max(x['e2e_s'] for r in rs for x in r['news']):.2f} | {alone:.2f} |")

bad = 0
plain = collections.defaultdict(set)
for r in rows:
    for kind in ("news", "solos"):
        for x in r[kind]:
            plain[(r["riders"], r["plain_class"], x["n"])].add((x["sha"], x["chunks"]))
            if x["error"]:
                bad += 1
                print("ERROR", r["arm"], r["cell"], kind, x["n"], x["error"])
mixed = {k: v for k, v in plain.items() if len(v) > 1}
print(f"\nplain texts: {len(plain)} (riders, class, n) keys over {sum(len(r['news']) + len(r['solos']) for r in rows)} "
      f"requests (beside streams and alone, all arms): {'IDENTICAL' if not mixed else 'DIFFER ' + str(mixed)}")
longs = collections.defaultdict(list)
for r in rows:
    for s in r["long_streams"]:
        longs[(r["riders"], r["plain_class"], s["stream"])].append((r["_dir"] / s["text_file"]).read_text())
fail = 0
for k, texts in longs.items():
    longest = max(texts, key=len)
    fail += sum(1 for t in texts if not longest.startswith(t))
print(f"long texts: {sum(len(v) for v in longs.values())} texts over {len(longs)} keys: "
      f"{'PREFIX PASS' if not fail else f'{fail} PREFIX FAIL'}")
sys.exit(1 if (mixed or fail or bad) else 0)
