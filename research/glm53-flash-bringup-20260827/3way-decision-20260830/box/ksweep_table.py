#!/usr/bin/env python3
"""K-sweep table + round-wall fit + break-even ceiling arithmetic (cell 6).

The sweep's value is not the winning K (there isn't one here) but the SHAPE: fitting round wall
against K separates the spec machinery's FIXED per-round cost from its per-draft cost, and the
fixed term is what decides whether any K can ever win on this placement.
"""
import json
import os
import statistics as st
import sys

PLAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 35.41
CDIR = sys.argv[2] if len(sys.argv) > 2 else "/root/out-3way/c6"
KS = (1, 2, 3, 5, 7)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main():
    pw = 1000.0 / PLAIN  # ms per token, plain
    print(f"plain reference: {PLAIN:.2f} tok/s => {pw:.2f} ms per decoded token")
    print(f"{'K':>3} {'dec tok/s':>10} {'ratio':>7} {'deep':>7} {'ttft0.4k':>9} {'ttft3.7k':>9} "
          f"{'tok/cyc':>8} {'roundwall_ms':>12} {'need tok/cyc':>13} {'short by':>9}")
    pts = []
    for k in KS:
        p = os.path.join(CDIR, f"k{k}", "timed.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        dec = [r for r in t["pool_rows"] if r["kind"] != "l3deep" and not r["err"]]
        dp = [r for r in t["pool_rows"] if r["kind"] == "l3deep" and not r["err"]]
        sp = [r["spec"] for r in t["pool_rows"] if r.get("spec")]
        acc = sum(s["accepted"] for s in sp)
        rnd = sum(s["rounds"] for s in sp)
        tpc = (acc + rnd) / rnd
        d = med([r["decode_tok_s"] for r in dec])
        dd = med([r["decode_tok_s"] for r in dp])
        dt = {r["tag"]: r["ttft_s"] for r in t["deep_ttft"]}
        rw = tpc / d * 1000.0
        need = rw / pw
        pts.append((k, rw))
        print(f"{k:>3} {d:>10.2f} {d / PLAIN:>7.3f} {dd:>7.2f} {dt.get('l3-WARM', 0):>9.3f} "
              f"{dt.get('l3-A4630', 0):>9.3f} {tpc:>8.3f} {rw:>12.1f} {need:>13.3f} {need - tpc:>+9.3f}")

    if len(pts) >= 3:
        n = len(pts)
        sx = sum(k for k, _ in pts); sy = sum(r for _, r in pts)
        sxx = sum(k * k for k, _ in pts); sxy = sum(k * r for k, r in pts)
        m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        c = (sy - m * sx) / n
        print(f"\nround-wall linear fit over {n} K points: round_wall = {c:.1f} + {m:.1f} * K  ms")
        print(f"  per-draft marginal cost        : {m:.1f} ms per extra draft token")
        print(f"  FIXED per-round cost (K=0)     : {c:.1f} ms")
        print(f"  plain per-token decode cost    : {pw:.2f} ms")
        print(f"  => the spec round's fixed cost alone is {c / pw:.3f}x a plain decode step,")
        print(f"     i.e. every round starts {c - pw:+.1f} ms in the hole before a single draft is judged.")
        rw1 = c + m
        print(f"\nceiling arithmetic at K=1 (round wall {rw1:.1f} ms):")
        print(f"  a PERFECT drafter (acc@1 = 1.0, tok/cyc 2.0) reaches {2000.0 / rw1:.2f} tok/s "
              f"= {(2000.0 / rw1) / PLAIN:.3f}x plain — that is the CEILING for spec here")
        print(f"  to merely TIE plain at K=1 the drafter needs tok/cyc >= {rw1 / pw:.3f}, "
              f"i.e. acc@1 >= {rw1 / pw - 1:.3f}")


if __name__ == "__main__":
    main()
