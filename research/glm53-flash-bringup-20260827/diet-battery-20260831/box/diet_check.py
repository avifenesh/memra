#!/usr/bin/env python3
"""Diet A/B analyzer + escalation check (owner protocol, amended law x3 -> x5 on anomaly).

Generalized from flip-battery-20260830/box/flip_check.py: arms and the baseline arm are
arguments, boot dirs are <base>/<arm>-<round>/timed.json.

Per boot: decode-pool tok/s median, deep-pool tok/s median, pool TTFT median, deep TTFT
rows, vendor row (128-token floor guard, excluded rows named). Aggregates per arm as
median of boot-medians, then the escalation rules:
  (a) within-arm relative spread of the decode-tok/s boot-medians
      ((max-min)/median) > 0.5%  -> escalate that arm (+baseline)
  (b) |arm_median - baseline_median| <= 2 * pooled_spread -> too close to call
Definitions stated here ARE the receipt. Exit 0 = x3 sufficient, exit 3 = ESCALATE.

Usage: diet_check.py --base /root/out-diet/c1 --baseline off --arms don[,dfk1,...]
"""
import argparse, glob, json, os, re, statistics as st, sys

VENDOR_FLOOR = 128  # the 3way measurement trap: short sampled completions collapse the estimator


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def boot_stats(d):
    t = json.load(open(os.path.join(d, "timed.json")))
    dec = [r for r in t["pool_rows"] if r["kind"] != "l3deep" and not r["err"]]
    deep = [r for r in t["pool_rows"] if r["kind"] == "l3deep" and not r["err"]]
    dt = {r["tag"]: r["ttft_s"] for r in t["deep_ttft"]}
    v = t.get("vendor_row") or {}
    v_ok = v and not v.get("err") and (v.get("completion_tokens") or 0) >= VENDOR_FLOOR
    return {
        "dec_tok_s": med([r["decode_tok_s"] for r in dec]),
        "deep_tok_s": med([r["decode_tok_s"] for r in deep]),
        "pool_ttft": med([r["ttft_s"] for r in dec]),
        "ttft_04k": dt.get("l3-WARM"),
        "ttft_37k": dt.get("l3-A4630"),
        "vendor_tok_s": v.get("decode_tok_s") if v_ok else None,
        "vendor_excluded": (None if v_ok else
                            f"ct={v.get('completion_tokens')} finish={v.get('finish')} err={v.get('err')}"),
        "vendor_spec": v.get("spec"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated non-baseline arms")
    a = ap.parse_args()
    want = [a.baseline] + [x for x in a.arms.split(",") if x]

    arms = {}
    for d in sorted(glob.glob(os.path.join(a.base, "*"))):
        m = re.match(r"^([a-z0-9]+)-(\d+)$", os.path.basename(d))
        if not m or m.group(1) not in want:
            continue
        if not os.path.exists(os.path.join(d, "timed.json")):
            continue
        arms.setdefault(m.group(1), []).append((int(m.group(2)), boot_stats(d)))

    summary, escalate = {}, []
    for arm in want:
        boots = sorted(arms.get(arm, []))
        if not boots:
            continue
        dm = [b["dec_tok_s"] for _, b in boots]
        m_ = med(dm)
        spread_abs = (max(dm) - min(dm)) if len(dm) > 1 else 0.0
        spread_rel = spread_abs / m_ if m_ else 0.0
        summary[arm] = {
            "n_boots": len(boots),
            "dec_tok_s_median": m_,
            "dec_boot_medians": [round(x, 3) for x in dm],
            "spread_abs_tok_s": round(spread_abs, 4),
            "spread_rel_pct": round(100 * spread_rel, 4),
            "deep_tok_s_median": med([b["deep_tok_s"] for _, b in boots]),
            "pool_ttft_median": med([b["pool_ttft"] for _, b in boots]),
            "ttft_04k_median": med([b["ttft_04k"] for _, b in boots]),
            "ttft_37k_median": med([b["ttft_37k"] for _, b in boots]),
            "vendor_tok_s_median": med([b["vendor_tok_s"] for _, b in boots]),
            "vendor_excluded": [b["vendor_excluded"] for _, b in boots if b["vendor_excluded"]],
        }
        if len(dm) > 1 and spread_rel > 0.005:
            escalate.append(f"RULE(a) arm={arm} spread_rel={100*spread_rel:.3f}% > 0.5%")

    p = summary.get(a.baseline, {})
    for arm in want[1:]:
        s = summary.get(arm)
        if not s or not p:
            continue
        pooled = max(s["spread_abs_tok_s"], p["spread_abs_tok_s"])
        gap = s["dec_tok_s_median"] - p["dec_tok_s_median"]
        s["gap_vs_baseline_tok_s"] = round(gap, 3)
        s["ratio_vs_baseline"] = round(s["dec_tok_s_median"] / p["dec_tok_s_median"], 4)
        s["pooled_spread_tok_s"] = round(pooled, 4)
        s["verdict"] = "WIN" if gap > 0 else "LOSS"
        if s["n_boots"] > 1 and p["n_boots"] > 1 and abs(gap) <= 2 * pooled:
            escalate.append(f"RULE(b) arm={arm} |gap|={abs(gap):.3f} <= 2*pooled_spread={2*pooled:.3f} (too close)")
            s["verdict"] = "TOO-CLOSE"

    print(json.dumps(summary, indent=1))
    if escalate:
        print("ESCALATE_TO_X5:")
        for e in escalate:
            print("  " + e)
        return 3
    print("X3_SUFFICIENT: no escalation rule fired (spreads receipted above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
