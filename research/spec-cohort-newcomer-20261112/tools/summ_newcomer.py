#!/usr/bin/env python3
"""Tables of the newcomer cell (memra#1227) and the prefix check of every stream.
usage: summ_newcomer.py RUNDIR [RUNDIR...]   (each holds newcomer.jsonl and texts/)
Arms are row["arm"] with the boot suffix removed ("base-1" -> base). Rows are grouped by
(long streams, class, newcomer class); newcomers of all boots of an arm are pooled.
Prefix check: for each (cell, stream) the long texts of all boots must be prefixes of the
longest one, and each newcomer's text sha must be equal across boots.
"all tok/s" is the long streams' tokens inside the newcomer window plus the newcomers' own
tokens, over the window: the arms' windows differ in length (a newcomer that waits less ends
sooner), so the long streams' rate alone is not comparable across arms."""
import collections
import json
import pathlib
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


rows = []
for d in sys.argv[1:]:
    d = pathlib.Path(d)
    for l in open(d / "newcomer.jsonl"):
        if l.strip():
            r = json.loads(l)
            r["_dir"] = d
            rows.append(r)
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    groups[(r["longs"], r["class"], r.get("newcomer_class", r["class"]) + ("+long" if r.get("long_prompts") else ""))][r["arm"].rsplit("-", 1)[0]].append(r)
print("| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |")
print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|")
for key in sorted(groups):
    for arm in sorted(groups[key]):
        rs = groups[key][arm]
        news = [n for r in rs for n in r["news"]]
        tt = [n["ttft_s"] for n in news if n["ttft_s"] is not None]
        sec = [n["second_s"] for n in news if n["second_s"] is not None]
        e2e = [n["e2e_s"] for n in news]
        ls = [s for r in rs for s in r["long_streams"]]
        gap = max((s["longest_gap_in_window_s"] for s in ls), default=None)
        p50 = [s["itl_p50_ms"] for s in ls if s.get("itl_p50_ms") is not None]
        p99 = [s["itl_p99_ms"] for s in ls if s.get("itl_p99_ms") is not None]
        errs = sum(r["newcomer_errors"] for r in rs)
        print(f"| {key[0]} | {key[1]}/{key[2]} | {arm} | {len(rs)} | {len(news)}{' (%d errors)' % errs if errs else ''} | {pct(tt, .5):.3f} | {pct(tt, .95):.3f} | {max(tt):.3f} | "
              f"{pct(sec, .5):.3f} | {pct(e2e, .5):.3f} | {', '.join(str(r['long_rate_during']) for r in rs) if key[0] else '-'} | {', '.join(str(round((r['long_tokens_during'] + sum(n['chunks'] for n in r['news'])) / r['window_s'], 2)) for r in rs)} | {gap if gap is not None else '-'} | "
              f"{(str(round(statistics.median(p50), 1)) + ' / ' + str(max(p99))) if p50 else '-'} |")
# prefix / sha equality
bad = 0
checked_long = checked_new = 0
bycell = collections.defaultdict(list)
for r in rows:
    bycell[(r["cell"], r["longs"], r["class"], r.get("newcomer_class", r["class"]) + ("+long" if r.get("long_prompts") else ""))].append(r)
for key, rs in sorted(bycell.items()):
    for i in range(key[1]):
        texts = [(r["arm"], (r["_dir"] / r["long_streams"][i]["text_file"]).read_text()) for r in rs]
        longest = max(texts, key=lambda t: len(t[1]))[1]
        for arm, t in texts:
            checked_long += 1
            if not longest.startswith(t) or not t:
                bad += 1
                print(f"PREFIX MISMATCH cell {key} stream {i} arm {arm} len {len(t)}")
    for n in range(len(rs[0]["news"])):
        shas = {(r["news"][n]["sha"], r["news"][n]["chunks"]) for r in rs}
        checked_new += len(rs)
        if len(shas) != 1:
            bad += 1
            print(f"NEWCOMER TEXT MISMATCH cell {key} newcomer {n}: {sorted(shas)}")
print(f"\nprefix check: {checked_long} long texts, {checked_new} newcomer texts, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")
