#!/usr/bin/env python3
"""memra#1211 tables from the cell rows.

usage:
  tables1211.py long STALL_JSONL...        per 30 s segment: long tokens and short p50
  tables1211.py normal REF_NORMAL_JSONL NORMAL_JSONL...
                                           c = 1..8 census; texts against REF's first boot
  tables1211.py follow STALL_JSONL...      recovery follows; texts across every arm given
  tables1211.py c2 NORMAL_JSONL...         every c = 2 phase: aggregate, start mode, K, marks
"""
import collections
import json
import sys


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def plain(k):
    return sum(v for key, v in k.items() if key.startswith("K0/")), sum(k.values())


mode, paths = sys.argv[1], sys.argv[2:]
if mode == "long":
    print("| arm | cell | long tokens per 30 s segment | total | short p50 s per segment | short p50 / p95 s, whole | newcomers plain / all | marks | refreshes |")
    print("|---|---|---|---:|---|---|---|---:|---:|")
    for p in paths:
        for r in load(p):
            segs = [s for s in r["segments"] if s["to_s"] - s["from_s"] > 5]
            cw = r.get("census_window") or {}
            a, b = plain(cw.get("spec_k") or {})
            print(f"| {r['arm']} | {r['clients']}:{int(r['think_ms'])} | {' / '.join(str(s['long_tokens']) for s in segs)} | "
                  f"{r['tokens_during']} | {' / '.join(str(s['short_p50_s']) for s in segs)} | "
                  f"{r['short_lat_p50_s']} / {r['short_lat_p95_s']} | {a} / {b} | {cw.get('stall_marks')} | {cw.get('stall_refreshes', '-')} |")
elif mode == "normal":
    ref_rows = load(paths[0])
    first = ref_rows[0]["arm"]
    ref = {}
    for r in ref_rows:
        if r["arm"] == first:
            for q in r["rows"]:
                ref[(r["c"], q["prompt"])] = q.get("sha")
    print(f"| boot | c | agg tok/s | TTFT p50 s | K at admission | marks | stalled verdicts | texts equal to {first} |")
    print("|---|---:|---:|---:|---|---:|---:|---|")
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        for r in load(p):
            ks = collections.Counter(r.get("spec_k", []))
            eq = sum(1 for q in r["rows"] if ref.get((r["c"], q["prompt"])) == q.get("sha"))
            print(f"| {r['arm']} | {r['c']} | {r['agg_tok_s']} | {r.get('ttft_p50_s')} | "
                  f"{' '.join(f'{k} x{v}' for k, v in sorted(ks.items()))} | {r.get('stall_marks')} | {r.get('stalled_verdicts')} | {eq} of {len(r['rows'])} |")
elif mode == "follow":
    rows = [r for p in paths for r in load(p)]
    shas = collections.defaultdict(dict)
    for r in rows:
        f = r.get("follow") or {}
        for q in f.get("rows") or []:
            shas[(r["clients"], r["think_ms"], f.get("after_s"), q["prompt"])][r["arm"]] = q.get("sha")
    print("| arm | pressure cell | follow starts after s | window marks | window newcomers plain / all | long open through the follow | follow K at admission | follow refreshes | follow agg tok/s | long tokens in the follow | follow texts equal across arms |")
    print("|---|---|---:|---:|---|---|---|---:|---:|---:|---|")
    for r in rows:
        f = r.get("follow") or {}
        cw = r.get("census_window") or {}
        a, b = plain(cw.get("spec_k") or {})
        fk = f.get("spec_k") or {}
        eq = []
        for q in f.get("rows") or []:
            arms = shas[(r["clients"], r["think_ms"], f.get("after_s"), q["prompt"])]
            eq.append(len(set(arms.values())) == 1 and len(arms) > 1)
        opened = "yes" if f.get("long_open_at_start") and f.get("long_open_at_end") else f"{f.get('long_open_at_start')} / {f.get('long_open_at_end')}"
        print(f"| {r['arm']} | {r['clients']}:{int(r['think_ms'])} | {f.get('after_s')} | {cw.get('stall_marks')} | {a} / {b} | {opened} | "
              f"{' '.join(f'{k} x{v}' for k, v in sorted(fk.items()))} | {f.get('stall_refreshes', '-')} | {f.get('agg_tok_s')} | "
              f"{f.get('long_tokens_during_follow')} | {sum(eq)} of {len(eq)} |")
elif mode == "c2":
    # A c = 2 phase starts in one of two modes: both clients' requests stay in lockstep (each
    # pair ends within 50 ms) or they run one tick apart.
    print("| boot | c = 2 phases | agg tok/s per phase | lockstep phases | TTFT p50 s per phase | K at admission | marks |")
    print("|---|---:|---|---:|---|---|---:|")
    tot = collections.defaultdict(lambda: [0, 0])
    for p in paths:
        by = collections.OrderedDict()
        for r in load(p):
            if r["c"] == 2:
                by.setdefault(r["arm"], []).append(r)
        for arm, rs in by.items():
            lock = 0
            for r in rs:
                ends = sorted(q["s"] for q in r["rows"])
                lock += all(abs(ends[i + 1] - ends[i]) < 0.05 for i in range(0, len(ends) - 1, 2))
            ks = collections.Counter(k for r in rs for k in r.get("spec_k", []))
            name = arm.rsplit("-", 1)[0]
            tot[name][0] += lock
            tot[name][1] += len(rs)
            print(f"| {arm} | {len(rs)} | {' / '.join(str(r['agg_tok_s']) for r in rs)} | {lock} | "
                  f"{' / '.join(str(r.get('ttft_p50_s')) for r in rs)} | {' '.join(f'{k} x{v}' for k, v in sorted(ks.items()))} | "
                  f"{sum(r.get('stall_marks') or 0 for r in rs)} |")
    print()
    for name, (a, b) in tot.items():
        print(f"- {name}: {a} of {b} phases in lockstep")
else:
    sys.exit(__doc__)
