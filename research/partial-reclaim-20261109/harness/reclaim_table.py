"""Tables of the partial-reclaim cells from each arm's serve.log and rows.json.

  python3 reclaim_table.py limit LIMIT_DIR    the at-the-limit cell (fill, pressure, repeat)
  python3 reclaim_table.py mixed MIXED_DIR    the mixed shape (regression, counts only)
  python3 reclaim_table.py waves WAVES_DIR    the at-the-limit cell with repeated pressure waves (limit_waves.py)

Every reclaim is one `reclaim-on-defer: evicted N prefix entries` line; a partial pass also prints a
`reclaim-on-defer (partial)` line ("covered" or "still short", the latter followed by the flush of the rest), and
the settle lines carry the bytes each reclaim handed back. Greedy texts are compared per row tag against the first
A arm of the same directory.
"""
import json, re, statistics, sys
from pathlib import Path

kind, root = sys.argv[1], Path(sys.argv[2])
def order(p):
    m = re.search(r"-(\d+)-[AB]$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


# An arm stopped mid-run keeps its receipts under a `.stopped` suffix and stays out of the tables.
arms = sorted((p for p in root.iterdir() if (p / "rows.json").is_file() and not p.name.endswith(".stopped")), key=order)
ref = next((p for p in arms if p.name.endswith("-A")), None)
ref_text = {r["tag"]: r["text"] for r in json.loads((ref / "rows.json").read_text())["rows"]} if ref else {}


def logstats(log):
    reclaims = [int(x) for x in re.findall(r"reclaim-on-defer: evicted (\d+) prefix entries", log)]
    settled = [int(x) for x in re.findall(r"reclaim settle \(reclaim-on-defer\): dev0 evicted_prefix_bytes=(\d+)", log)]
    partial = re.findall(r"reclaim-on-defer \(partial\): evicted (\d+) prefix entries \((\d+)MB, oldest first\) for "
                         r"(\d+)MB \((\d+)MB of it owed by (\d+) priming sessions(?: and the arrival's own entry)?\); "
                         r"kept (\d+) unleased entries \((\d+)MB\); effective free (\d+)MB -> (\d+)MB; (.+?) \(model", log)
    return {
        "budget": (re.search(r"\[prefix-cache\] on: budget (\d+)MB", log) or [None, "?"])[1],
        "reclaims": reclaims, "settled": settled, "partial": partial,
        "short": sum(1 for p in partial if p[9].startswith("still short")),
        "booking_on": len(re.findall(r"\[admit-oom\] reclaim booking on:", log)),
        "booking_off": len(re.findall(r"\[admit-oom\] reclaim booking off:", log)),
        "defers": len(re.findall(r"\[admit-oom\] VRAM defer:", log)),
        "oom": len(re.findall(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY", log, re.I)),
        "preflight": len(re.findall(r"\[prefix-cache\] evict \(snapshot preflight, LRU\)", log)),
        "lru": len(re.findall(r"\[prefix-cache\] evict \(LRU\)", log)),
        "snapshot_failed": len(re.findall(r"\[prefix-cache\] snapshot failed", log)),
        "step_oom": len(re.findall(r"step-OOM reclaim", log)),
    }


def same_texts(rows):
    rs = [r for r in rows if r["tag"] in ref_text]
    differ = sorted(r["tag"] for r in rs if r["text"] != ref_text[r["tag"]])
    return len(rs) - len(differ), len(rs), differ


differs = {}
if kind == "limit":
    print("| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off "
          "| evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, "
          "TTFT p50 s | greedy texts = A |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in arms:
        st = logstats((d / "serve.log").read_text(errors="replace"))
        j = json.loads((d / "rows.json").read_text())
        s, rows = j["summary"], j["rows"]
        p, r = s["pressure"], s["repeat"]
        big = max(zip(st["reclaims"], st["settled"]), key=lambda x: x[1], default=(0, 0))
        same, n, differ = same_texts(rows)
        differs[d.name] = [(t, next((x.get("error") or "" for x in rows if x["tag"] == t), "")[:60]) for t in differ]
        print(f"| {d.name} | {st['budget']} | {len(st['reclaims'])} ({len(st['partial'])}, {st['short']}) | "
              f"{sum(st['reclaims'])} / {sum(st['settled']) / 1e9:.1f} ({big[0]} / {big[1] / 1e9:.1f}) | "
              f"{st['booking_on']} / {st['booking_off']} | {st['preflight']} / {st['lru']} | {st['oom']} | "
              f"{p['errors']} of {p['rows']}, {p['ttft_ms_p50'] / 1000:.1f} / {p['ttft_ms_max'] / 1000:.1f} | "
              f"{r['hits']} of {r['rows']}, {r['cached_tokens']}, {r['ttft_ms_p50'] / 1000:.2f} | {same}/{n} |")
    print()
    print("Partial passes (arm: entries freed, MB freed, MB needed, of it owed, priming sessions, entries kept, MB kept, "
          "free before -> after MB, outcome):")
    for d in arms:
        st = logstats((d / "serve.log").read_text(errors="replace"))
        for q in st["partial"]:
            print(f"- {d.name}: {q[0]} entries, {q[1]} MB for {q[2]} MB ({q[3]} owed, {q[4]} priming); kept {q[5]} "
                  f"({q[6]} MB); free {q[7]} -> {q[8]} MB; {q[9]}")
elif kind == "mixed":
    print("| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) "
          "| evict lines | OOM lines | greedy texts = A |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in arms:
        st = logstats((d / "serve.log").read_text(errors="replace"))
        j = json.loads((d / "rows.json").read_text())
        s, rows = j["summary"], j["rows"]
        rep = [x for x in s["repeat_ttft_ms"] if x is not None]
        same, n, differ = same_texts(rows)
        differs[d.name] = [(t, "") for t in differ]
        print(f"| {d.name} | {st['budget']} | {s['repeat_hits']} of {len(s['repeat_cached'])} | "
              f"{statistics.median(rep):.0f} | {s['batch_agg_tok_s']} | {s['batch_ttft_ms_p50']:.0f} | "
              f"{len(st['reclaims'])} ({len(st['partial'])}) | {st['preflight'] + st['lru']} | {st['oom']} | {same}/{n} |")
elif kind == "waves":
    print("| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines "
          "| pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |")
    print("|---|---|---|---|---|---|---|---|")
    for d in arms:
        st = logstats((d / "serve.log").read_text(errors="replace"))
        j = json.loads((d / "rows.json").read_text())
        s, rows = j["summary"], j["rows"]
        r = s["repeat"]
        big = max(zip(st["reclaims"], st["settled"]), key=lambda x: x[1], default=(0, 0))
        same, n, differ = same_texts(rows)
        differs[d.name] = [(t, next((x.get("error") or "" for x in rows if x["tag"] == t), "")[:60]) for t in differ]
        w = "; ".join(f"{x['errors']}, {x['ttft_ms_p50'] / 1000:.1f} / {x['ttft_ms_max'] / 1000:.1f}" for x in s["waves"])
        print(f"| {d.name} | {len(st['reclaims'])} ({len(st['partial'])}, {st['short']}) | "
              f"{sum(st['reclaims'])} / {sum(st['settled']) / 1e9:.1f} ({big[0]} / {big[1] / 1e9:.1f}) | "
              f"{st['booking_on']} / {st['booking_off']} | {st['oom']} | {w} | "
              f"{r['hits']} of {r['rows']}, {r['cached_tokens']}, {r['ttft_ms_p50'] / 1000:.2f} | {same}/{n} |")
else:
    sys.exit(f"unknown table {kind}")
print()
print("Rows whose greedy text differs from the first A arm (tag, the row's error if any):")
for name, rows in differs.items():
    if rows:
        print(f"- {name}: " + "; ".join(f"{t} {e}".strip() for t, e in rows))
