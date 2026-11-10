"""Compare a stall-cell run with the #1213 lane's GB10 rows of the same cells: tokens in the window, longest gap,
short-request latency, and whether the long text is prefix-equal (the shorter text is a prefix of the longer) to
the lane's fix and main v0.141.0 texts. usage: compare_stall.py OURS_DIR LANE_DIR OUT_MD"""
import json
import sys
from pathlib import Path

ours, lane, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def rows(d):
    return [json.loads(l) for l in (d / "stall.jsonl").read_text().splitlines() if l.strip()]


def text(d, r):
    return (d / r["text_file"]).read_text() if r.get("text_file") else ""


def prefix_equal(a, b):
    s, l = sorted((a, b), key=len)
    return l.startswith(s)


ref = {name: {(r["clients"], r["think_ms"], r["class"]): r for r in rows(lane / name)} for name in ("sweep-fix", "sweep-main141")}
lines = ["| clients | think ms | long tokens in 30 s (lane fix / main v0.141.0) | longest gap s (lane fix) | short requests | short p50 s | short p95 s | demotions | text vs lane fix | text vs main v0.141.0 |",
         "|---:|---:|---|---|---:|---:|---:|---:|---|---|"]
ok = True
for r in rows(ours):
    key = (r["clients"], r["think_ms"], r["class"])
    f, m = ref["sweep-fix"].get(key), ref["sweep-main141"].get(key)
    t = text(ours, r)
    pf = prefix_equal(t, text(lane / "sweep-fix", f)) if f else None
    pm = prefix_equal(t, text(lane / "sweep-main141", m)) if m else None
    ok = ok and r.get("error") is None and pf is not False and pm is not False and r["longest_gap_in_window_s"] < 2.0 and r["tokens_during"] > 2
    lines.append(f"| {r['clients']} | {r['think_ms']:.0f} | {r['tokens_during']} ({f['tokens_during'] if f else '-'} / {m['tokens_during'] if m else '-'}) | "
                 f"{r['longest_gap_in_window_s']} ({f['longest_gap_in_window_s'] if f else '-'}) | {r['short_requests']} | {r['short_lat_p50_s']} | "
                 f"{r['short_lat_p95_s']} | {r['demotions']} | {'prefix-equal' if pf else pf} | {'prefix-equal' if pm else pm} |")
lines.append("")
lines.append(f"verdict: {'PASS' if ok else 'FAIL'} (the long request advances in every cell, longest gap under 2 s, texts prefix-equal)")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
