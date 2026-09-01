#!/usr/bin/env python3
"""Build THE DECISION TABLE from the 3way-decision timed receipts.

Aggregation rule (spec-battery-20260830 precedent, kept identical so the rows compare):
per boot take the MEDIAN over the pool rows, then the MEDIAN of the per-boot medians across
the interleaved rounds. Rows with an error, or flagged by the loop law, are excluded and the
exclusion is printed (never silently dropped).

Usage: summarize.py <s4dir> [--arms plain,nat,dfl]
"""
import argparse
import json
import os
import statistics as st


VENDOR_TOK_FLOOR = 128


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def collect(d, arm, rounds, prefix):
    """Returns per-boot dicts for one arm."""
    boots = []
    for i in rounds:
        p = os.path.join(d, f"{prefix}{arm}{i}", "timed.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        decode_pool = [r for r in t["pool_rows"] if r["kind"] != "l3deep" and not r["err"]]
        deep_pool = [r for r in t["pool_rows"] if r["kind"] == "l3deep" and not r["err"]]
        errs = [r["tag"] for r in t["pool_rows"] if r["err"]]
        specs = [r["spec"] for r in t["pool_rows"] if r.get("spec")]
        accpc = None
        if specs:
            acc = sum(s["accepted"] for s in specs)
            rnd = sum(s["rounds"] for s in specs)
            accpc = (acc / rnd) if rnd else None
        dt = {r["tag"]: r["ttft_s"] for r in t["deep_ttft"]}
        vr = t["vendor_row"] or {}
        vendor_ct = vr.get("completion_tokens")
        vendor_excluded = bool(vendor_ct is not None and vendor_ct < VENDOR_TOK_FLOOR)
        vendor_tok_s = None if vendor_excluded else vr.get("decode_tok_s")
        boots.append({
            "boot": f"{prefix}{arm}{i}",
            "decode_tok_s": med([r["decode_tok_s"] for r in decode_pool]),
            "deep_tok_s": med([r["decode_tok_s"] for r in deep_pool]),
            "pool_ttft": med([r["ttft_s"] for r in decode_pool]),
            "ttft_short": dt.get("l3-WARM"),
            "ttft_deep": dt.get("l3-A4630"),
            "acc_per_cycle": accpc,
            "tok_per_cycle": (accpc + 1) if accpc is not None else None,
            # VENDOR TOK/S FLOOR (measurement trap caught in this window, s4-dfl2): the
            # streamed estimator is (ct-1)/(t_last_chunk - t_first_chunk). On a SHORT sampled
            # completion that ends quickly the server flushes the tail in a couple of SSE
            # chunks, the span collapses, and tok/s explodes (s4-dfl2: ct=35, finish=stop,
            # span 0.109 s -> 310.8 tok/s against a real ~31.7). Rows under
            # VENDOR_TOK_FLOOR tokens are EXCLUDED from the vendor median and the exclusion is
            # printed. Greedy pool rows are unaffected (every one of them >= 232 tokens).
            "vendor_tok_s": vendor_tok_s,
            "vendor_ct": vendor_ct,
            "vendor_excluded": vendor_excluded,
            "vendor_spec": bool((t["vendor_row"] or {}).get("spec")),
            "n_spec_rows": len(specs),
            "errors": errs,
        })
    return boots


def boot_seconds(logdir, boot):
    p = os.path.join(logdir, f"boot-{boot}.identity")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("boot_s="):
            return int(line.strip().split("=", 1)[1])
    return None


def vram(logdir, boot):
    p = os.path.join(logdir, f"boot-{boot}.vram")
    if not os.path.exists(p):
        return None
    used = []
    for line in open(p):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            used.append(int(parts[1].split()[0]))
    return used[:3] if used else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s4dir")
    ap.add_argument("--logdir", default="/root/out-3way/logs")
    ap.add_argument("--arms", default="plain,nat,dfl")
    ap.add_argument("--prefix", default="s4-", help="boot-dir prefix, e.g. s4- or c6b-")
    ap.add_argument("--rounds", default="1,2,3,4,5")
    a = ap.parse_args()
    rounds = [int(x) for x in a.rounds.split(",")]
    arms = a.arms.split(",")

    per_arm = {}
    for arm in arms:
        boots = collect(a.s4dir, arm, rounds, a.prefix)
        per_arm[arm] = boots
        print(f"\n=== per-boot medians, arm={arm} (n_boots={len(boots)}) ===")
        print(f"{'boot':<12} {'dec tok/s':>10} {'deep tok/s':>11} {'poolTTFT':>9} "
              f"{'ttft0.4k':>9} {'ttft3.7k':>9} {'acc/cyc':>8} {'vendor t/s':>11} {'boot_s':>7} {'specrows':>9}")
        for b in boots:
            print(f"{b['boot']:<12} {fmt(b['decode_tok_s'],2):>10} {fmt(b['deep_tok_s'],2):>11} "
                  f"{fmt(b['pool_ttft']):>9} {fmt(b['ttft_short']):>9} {fmt(b['ttft_deep']):>9} "
                  f"{fmt(b['acc_per_cycle']):>8} {fmt(b['vendor_tok_s'],1):>11} "
                  f"{str(boot_seconds(a.logdir,b['boot'])):>7} {b['n_spec_rows']:>9}")
            if b["errors"]:
                print(f"    EXCLUDED rows (error): {b['errors']}")
            if b["vendor_excluded"]:
                print(f"    EXCLUDED vendor row: completion_tokens={b['vendor_ct']} "
                      f"< floor {VENDOR_TOK_FLOOR} (short-sampled streamed-span artifact)")

    print("\n\n################ THE DECISION TABLE ################")
    print(f"{'arm':<8} {'decode tok/s':>13} {'deep tok/s':>11} {'TTFT 0.4k':>10} {'TTFT 3.7k':>10} "
          f"{'acc/cyc':>8} {'tok/cyc':>8} {'vendor t/s':>11} {'boot s':>7} {'VRAM d0/d1/d2 MiB':>22}")
    base = None
    rows = {}
    for arm in arms:
        b = per_arm[arm]
        if not b:
            continue
        row = {
            "decode": med([x["decode_tok_s"] for x in b]),
            "deep": med([x["deep_tok_s"] for x in b]),
            "ttft_s": med([x["ttft_short"] for x in b]),
            "ttft_d": med([x["ttft_deep"] for x in b]),
            "acc": med([x["acc_per_cycle"] for x in b]),
            "vendor": med([x["vendor_tok_s"] for x in b]),
            "boot_s": med([boot_seconds(a.logdir, x["boot"]) for x in b]),
            "vram": vram(a.logdir, b[0]["boot"]),
            "vendor_spec_all": all(x["vendor_spec"] for x in b),
        }
        row["tokcyc"] = (row["acc"] + 1) if row["acc"] is not None else None
        rows[arm] = row
        vr = "/".join(str(v) for v in (row["vram"] or [])) or "n/a"
        print(f"{arm:<8} {fmt(row['decode'],2):>13} {fmt(row['deep'],2):>11} {fmt(row['ttft_s']):>10} "
              f"{fmt(row['ttft_d']):>10} {fmt(row['acc']):>8} {fmt(row['tokcyc']):>8} "
              f"{fmt(row['vendor'],1):>11} {fmt(row['boot_s'],0):>7} {vr:>22}")
        if arm == arms[0]:
            base = row

    if base and base["decode"]:
        print(f"\n--- ratios vs {arms[0]} (decode leads; acceptance is a diagnostic) ---")
        for arm in arms[1:]:
            r = rows.get(arm)
            if not r or not r["decode"]:
                continue
            print(f"{arm:<8} decode {r['decode']/base['decode']:.3f}x  "
                  f"deep {r['deep']/base['deep']:.3f}x  "
                  f"TTFT0.4k {r['ttft_s']/base['ttft_s']:.2f}x  "
                  f"TTFT3.7k {r['ttft_d']/base['ttft_d']:.2f}x  "
                  f"vendor {r['vendor']/base['vendor']:.3f}x")
        # break-even arithmetic: what tok/cycle a spec arm needs to match plain, given its
        # measured per-round cost implied by its own decode tok/s and tok/cycle.
        print(f"\n--- break-even arithmetic (plain = {base['decode']:.2f} tok/s) ---")
        for arm in arms[1:]:
            r = rows.get(arm)
            if not r or not r["decode"] or not r["tokcyc"]:
                continue
            # measured: decode = tokcyc / round_wall  =>  round_wall = tokcyc / decode
            round_wall = r["tokcyc"] / r["decode"]
            need = base["decode"] * round_wall
            print(f"{arm:<8} round wall {round_wall*1000:.2f} ms/cycle at tok/cyc {r['tokcyc']:.3f}; "
                  f"needs tok/cyc >= {need:.3f} (i.e. acc/cyc >= {need-1:.3f}) to match plain — "
                  f"measured {r['tokcyc']:.3f} => {'BEATS' if r['tokcyc']>need else 'LOSES TO'} plain")
    print("\nvendor-default spec engagement present on every boot:",
          {k: rows[k]["vendor_spec_all"] for k in rows})
    nex = {arm: sum(1 for x in per_arm[arm] if x["vendor_excluded"]) for arm in arms if per_arm[arm]}
    print(f"vendor rows excluded by the {VENDOR_TOK_FLOOR}-token floor (per arm):", nex)


if __name__ == "__main__":
    main()
