#!/usr/bin/env python3
"""Ladder table + escalation check for the hbatch-battery cell 2.

Reads c2/l{r}-{arm}/c{c}/conc-{c}-greedy.json (and c1/timed.json for the c=1 rung),
prints per-(arm, c): aggregate median [min..max] rel spread, decode-window median,
per-session p50/p95 pooled, TTFT p50/max pooled — and applies the owner escalation rules:
(a) within-arm aggregate rel spread > 0.5% at any rung, or (b) |on-off| median gap < 2x
pooled spread. Exit 2 = ESCALATE (x5), 0 = x3 sufficient.

Usage: ladder_table.py /root/out-hbatch/c2 [rounds...]
"""
import glob, json, os, statistics, sys


def collect(base, rounds):
    data = {}  # (arm, c) -> list of dicts per round
    for r in rounds:
        for arm in ("off", "on"):
            d = os.path.join(base, f"l{r}-{arm}")
            if not os.path.isdir(d):
                continue
            t = os.path.join(d, "c1", "timed.json")
            if os.path.exists(t):
                tj = json.load(open(t))
                med = tj.get("decode_pool_median_tok_s")
                if med:
                    data.setdefault((arm, 1), []).append(
                        {"aggregate_tok_s": med, "decode_window_tok_s": med,
                         "per_session_tok_s_p50": med, "per_session_tok_s_p95": None,
                         "ttft_p50_s": statistics.median(
                             [x["ttft_s"] for x in tj["pool_rows"] if x["kind"] != "l3deep" and x["ttft_s"]]),
                         "ttft_max_s": None})
            for f in glob.glob(os.path.join(d, "c*", "conc-*-greedy.json")):
                j = json.load(open(f))
                data.setdefault((arm, j["n"]), []).append(j)
    return data


def main():
    base = sys.argv[1]
    rounds = sys.argv[2:] or ["1", "2", "3", "4", "5"]
    data = collect(base, rounds)
    escalate = []
    print(f"{'arm':<5} {'c':>3} {'runs':>4} {'agg med':>8} {'spread%':>8} {'dw med':>8} "
          f"{'p50':>6} {'p95':>6} {'ttft50':>7} {'ttftmax':>8}")
    spreads = {}
    for (arm, c) in sorted(data, key=lambda k: (k[1], k[0])):
        runs = data[(arm, c)]
        aggs = [r["aggregate_tok_s"] for r in runs if r.get("aggregate_tok_s")]
        med = statistics.median(aggs)
        spread = (max(aggs) - min(aggs)) / med * 100 if len(aggs) > 1 else 0.0
        spreads[(arm, c)] = (med, spread)
        dws = [r["decode_window_tok_s"] for r in runs if r.get("decode_window_tok_s")]
        p50s = [r["per_session_tok_s_p50"] for r in runs if r.get("per_session_tok_s_p50")]
        p95s = [r["per_session_tok_s_p95"] for r in runs if r.get("per_session_tok_s_p95")]
        t50 = [r["ttft_p50_s"] for r in runs if r.get("ttft_p50_s")]
        tmx = [r["ttft_max_s"] for r in runs if r.get("ttft_max_s")]
        if spread > 0.5:
            escalate.append(f"rule-a arm={arm} c={c} spread {spread:.3f}% > 0.5%")
        print(f"{arm:<5} {c:>3} {len(aggs):>4} {med:>8.2f} {spread:>8.3f} "
              f"{statistics.median(dws) if dws else 0:>8.2f} "
              f"{statistics.median(p50s) if p50s else 0:>6.1f} "
              f"{statistics.median(p95s) if p95s else 0:>6.1f} "
              f"{statistics.median(t50) if t50 else 0:>7.3f} "
              f"{statistics.median(tmx) if tmx else 0:>8.2f}")
    for c in sorted({c for (_, c) in spreads}):
        if ("off", c) in spreads and ("on", c) in spreads:
            om, osp = spreads[("off", c)]
            nm, nsp = spreads[("on", c)]
            pooled = (osp + nsp) / 2 / 100 * (om + nm) / 2
            gap = abs(nm - om)
            ratio = nm / om if om else 0
            print(f"c={c:>2}: ON/OFF = {ratio:.4f}x (gap {gap:+.2f} tok/s, 2x pooled spread {2*pooled:.2f})")
            if gap < 2 * pooled:
                escalate.append(f"rule-b c={c} gap {gap:.2f} < 2x pooled spread {2*pooled:.2f}")
    if escalate:
        print("ESCALATE (x5):")
        for e in escalate:
            print("  " + e)
        sys.exit(2)
    print("X3 SUFFICIENT: no escalation rule fired")


if __name__ == "__main__":
    main()
