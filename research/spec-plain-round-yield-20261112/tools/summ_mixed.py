#!/usr/bin/env python3
"""Summarize tools/mixed_cell.py rows (memra#1244 round-2 review). usage: summ_mixed.py DIR...
Per (cell spec, arm): the streams' rate before the plain requests and while the interactive ones
are live, every request's gap p50, TTFT and E2E, its rate inside that window, and the total. Then
the text checks: every plain request writes one text per (class, n) beside the streams and alone,
in every arm; every stream text is a prefix of the longest run of its stream. Exit 0 when both
hold."""
import json
import pathlib
import statistics as st
import sys

rows = []
for d in sys.argv[1:]:
    dp = pathlib.Path(d)
    for line in open(dp / "mixed.jsonl"):
        if line.strip():
            r = json.loads(line)
            r["_dir"] = dp
            rows.append(r)


def arm_of(tag):
    return tag.rsplit("-", 1)[0]


groups = {}
for r in rows:
    groups.setdefault((r["spec"], arm_of(r["arm"])), []).append(r)
print("| cell | arm | boots | streams tok/s before | streams tok/s live | all tok/s live | request | lane | "
      "gap p50 s | TTFT s | E2E s | tok/s live |")
print("|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|")
for (spec, arm), rs in sorted(groups.items()):
    before = round(st.mean(r["long_rate_before"] for r in rs), 2)
    live = round(st.mean(r["long_rate_live"] for r in rs), 2)
    agg = round(st.mean(r["agg_tok_s_live"] for r in rs), 2)
    for k, item in enumerate(rs[0]["beside"]):
        xs = [r["beside"][k] for r in rs]
        gap = st.median([x["itl_p50_s"] for x in xs if x["itl_p50_s"] is not None]) if any(
            x["itl_p50_s"] is not None for x in xs) else None
        ttft = st.median([x["ttft_s"] for x in xs if x["ttft_s"] is not None]) if any(
            x["ttft_s"] is not None for x in xs) else None
        e2e = st.median(x["e2e_s"] for x in xs)
        rate = round(st.mean(x["rate_live"] for x in xs), 2)
        head = f"| {spec} | {arm} | {len(rs)} | {before} | {live} | {agg} |" if k == 0 else "| | | | | | |"
        print(f"{head} {item['cls']}-{item['n']} | {item['lane']} | {gap} | {ttft} | {round(e2e, 3)} | {rate} |")
print()
ok = True
texts = {}
for r in rows:
    for phase in ("beside", "alone"):
        for x in r[phase]:
            if x["error"]:
                ok = False
                print(f"ERROR {r['arm']} cell {r['cell']} {phase} {x['spec']}: {x['error']}")
            texts.setdefault((x["cls"], x["n"], x["max_tokens"]), set()).add(x["sha"])
diff = {k: v for k, v in texts.items() if len(v) > 1}
n_req = sum(len(r["beside"]) + len(r["alone"]) for r in rows)
print(f"plain texts: {len(texts)} (class, n, max tokens) keys over {n_req} requests (beside streams and alone, all arms): "
      f"{'IDENTICAL' if not diff else 'DIFFER ' + str(diff)}")
ok &= not diff
longs = {}
for r in rows:
    for s in r["long_streams"]:
        longs.setdefault(s["stream"], []).append((r["_dir"] / s["text_file"]).read_text())
bad = 0
for i, ts in longs.items():
    longest = max(ts, key=len)
    bad += sum(1 for t in ts if not longest.startswith(t))
print(f"long texts: {sum(len(v) for v in longs.values())} texts over {len(longs)} keys: "
      f"{'PREFIX PASS' if not bad else f'PREFIX FAIL ({bad})'}")
ok &= not bad
sys.exit(0 if ok else 1)
