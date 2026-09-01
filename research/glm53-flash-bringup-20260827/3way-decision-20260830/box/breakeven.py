#!/usr/bin/env python3
"""Per-prompt break-even analysis for the 3way decision packet.

The pool-median ratio answers "does spec win on average". The owner's decision needs the
sharper question: WHERE does it win, and what acceptance does it need to win. This pairs each
prompt's spec row against the SAME prompt's plain row (same round, same boot pair), so the
comparison is per-prompt rather than per-pool, and solves for the tie point.

Model: a spec round emits tok/cycle tokens for one round-wall. Holding the measured round wall
fixed, the arm ties plain at  tok/cycle_needed = plain_tok_s * round_wall.
Equivalently the measured surplus is  tok_s_spec / tok_s_plain.

Usage: breakeven.py <s4dir> --arm dfl --rounds 1,2,3,4,5
"""
import argparse
import glob
import json
import os
import statistics as st


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def rows_for(d, arm, rounds, prefix):
    """tag -> list of (tok_s, tok_per_cycle, ttft) across rounds."""
    out = {}
    for i in rounds:
        p = os.path.join(d, f"{prefix}{arm}{i}", "timed.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        for r in t["pool_rows"]:
            if r["err"]:
                continue
            s = r.get("spec")
            tpc = None
            if s and s.get("rounds"):
                tpc = (s["accepted"] + s["rounds"]) / s["rounds"]
            out.setdefault(r["tag"], []).append((r["decode_tok_s"], tpc, r["ttft_s"], r["kind"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s4dir")
    ap.add_argument("--arm", default="dfl")
    ap.add_argument("--base", default="plain")
    ap.add_argument("--rounds", default="1,2,3,4,5")
    ap.add_argument("--prefix", default="s4-", help="boot-dir prefix, e.g. s4- or c6b-")
    a = ap.parse_args()
    rounds = [int(x) for x in a.rounds.split(",")]

    spec = rows_for(a.s4dir, a.arm, rounds, a.prefix)
    base = rows_for(a.s4dir, a.base, rounds, a.prefix)

    print(f"=== per-prompt: {a.arm} vs {a.base} (median over {len(rounds)} interleaved rounds) ===")
    print(f"{'tag':<13} {'kind':<7} {'plain t/s':>10} {'spec t/s':>9} {'ratio':>6} "
          f"{'tok/cyc':>8} {'needed':>7} {'margin':>7}  verdict")
    wins = losses = 0
    pts = []
    for tag in sorted(set(spec) & set(base)):
        sp = med([x[0] for x in spec[tag]])
        bs = med([x[0] for x in base[tag]])
        tpc = med([x[1] for x in spec[tag]])
        kind = spec[tag][0][3]
        if not (sp and bs and tpc):
            continue
        round_wall = tpc / sp
        needed = bs * round_wall
        ratio = sp / bs
        verdict = "WIN " if ratio > 1 else "loss"
        wins += ratio > 1
        losses += ratio <= 1
        pts.append((tpc, ratio))
        print(f"{tag:<13} {kind:<7} {bs:>10.2f} {sp:>9.2f} {ratio:>6.3f} "
              f"{tpc:>8.3f} {needed:>7.3f} {tpc-needed:>+7.3f}  {verdict}")
    print(f"\nper-prompt record: {wins} WIN / {losses} loss of {wins+losses}")

    # Solve the tie point from the measured (tok/cycle -> ratio) relation by least squares
    # through the origin-shifted linear fit ratio = m*tpc + c, then report tpc at ratio == 1.
    if len(pts) >= 3:
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if den:
            m = (n * sxy - sx * sy) / den
            c = (sy - m * sx) / n
            print(f"\nfit: ratio = {m:.4f} * tok/cycle + {c:.4f}")
            if m:
                tie = (1.0 - c) / m
                print(f"TIE POINT: this arm matches {a.base} at tok/cycle = {tie:.3f} "
                      f"(acc/cycle = {tie-1:.3f})")
                obs = med([p[0] for p in pts])
                print(f"measured median tok/cycle = {obs:.3f} -> "
                      f"{'ABOVE' if obs>tie else 'BELOW'} the tie point by {abs(obs-tie):.3f}")
    # TTFT shape: is the penalty flat or depth-scaling?
    print(f"\n=== TTFT penalty shape ({a.arm} minus {a.base}, median per prompt) ===")
    print(f"{'tag':<13} {'plain':>7} {'spec':>7} {'delta':>7}")
    for tag in sorted(set(spec) & set(base)):
        sp = med([x[2] for x in spec[tag]]); bs = med([x[2] for x in base[tag]])
        if sp and bs:
            print(f"{tag:<13} {bs:>7.3f} {sp:>7.3f} {sp-bs:>+7.3f}")


if __name__ == "__main__":
    main()
