#!/usr/bin/env python3
"""Flip-table analyzer + escalation check (owner protocol 2026-08-30; flip-reprice window).

Reads /root/out-flip3/c3/<arm><round>/timed.json for arms plain/k1/k2/k3 (the batched-walk
arms; the zctl-* =0 control boots are reported separately and never enter the table),
computes per boot: decode-pool tok/s median, deep-pool tok/s median, pool TTFT median,
deep TTFT rows, vendor row (128-token floor guard, excluded rows named), and the round
wall K/dec*1000*(tok_cyc) identity via acceptance. Aggregates per arm as median of
boot-medians, then applies the escalation rules:
  (a) within-arm relative spread of the decode-tok/s boot-medians
      ((max-min)/median) > 0.5%  -> escalate that arm (+plain)
  (b) |spec_median - plain_median| <= 2 * pooled_spread -> too close to call
Definitions stated here ARE the receipt. Exit 0 = x3 sufficient, exit 3 = ESCALATE.
"""
import glob, json, os, re, statistics as st, sys

BASE = "/root/out-flip3/c3"
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
    # acceptance across the decode pool rows that carry usage.spec (tok/cyc + round wall)
    specs = [r["spec"] for r in t["pool_rows"] if r.get("spec") and not r["err"]]
    acc = sum(s["accepted"] for s in specs)
    rnd = sum(s["rounds"] for s in specs)
    tok_cyc = (acc + rnd) / rnd if rnd else None
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
        "tok_cyc": tok_cyc,
    }


def main():
    arms = {}
    for d in sorted(glob.glob(os.path.join(BASE, "*"))):
        name = os.path.basename(d)
        m = re.match(r"(plain|k1-|k2-|k3-|zctl-)(\d+)$", name) or re.match(r"(plain)(\d+)$", name)
        if not m or not os.path.exists(os.path.join(d, "timed.json")):
            continue
        arm = m.group(1).rstrip("-")
        arms.setdefault(arm, []).append((int(m.group(2)), boot_stats(d)))

    summary, escalate = {}, []
    for arm in ("plain", "k1", "k2", "k3", "zctl"):
        boots = sorted(arms.get(arm, []))
        if not boots:
            continue
        dm = [b["dec_tok_s"] for _, b in boots]
        m_ = med(dm)
        spread_abs = (max(dm) - min(dm)) if len(dm) > 1 else 0.0
        spread_rel = spread_abs / m_ if m_ else 0.0
        tc = med([b["tok_cyc"] for _, b in boots])
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
            "tok_cyc_median": tc and round(tc, 3),
            "round_wall_ms": (round(tc / m_ * 1000, 2) if (tc and m_) else None),
        }
        if arm != "zctl" and spread_rel > 0.005:
            escalate.append(f"RULE(a) arm={arm} spread_rel={100*spread_rel:.3f}% > 0.5%")

    p = summary.get("plain", {})
    for arm in ("k1", "k2", "k3"):
        s = summary.get(arm)
        if not s or not p:
            continue
        pooled = max(s["spread_abs_tok_s"], p["spread_abs_tok_s"])
        gap = s["dec_tok_s_median"] - p["dec_tok_s_median"]
        s["gap_vs_plain_tok_s"] = round(gap, 3)
        s["ratio_vs_plain"] = round(s["dec_tok_s_median"] / p["dec_tok_s_median"], 4)
        s["pooled_spread_tok_s"] = round(pooled, 4)
        s["verdict"] = "FLIP" if gap > 0 else "NO-FLIP"
        if abs(gap) <= 2 * pooled:
            escalate.append(f"RULE(b) arm={arm} |gap|={abs(gap):.3f} <= 2*pooled_spread={2*pooled:.3f} (too close)")
            s["verdict"] = "TOO-CLOSE"
    if "zctl" in summary:
        summary["zctl"]["note"] = ("=0 control (old per-row walk): seam receipt only, "
                                   "never enters the flip table; round_wall_ms reads "
                                   "against the flip-battery 91.08 ms @ K=3")

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
