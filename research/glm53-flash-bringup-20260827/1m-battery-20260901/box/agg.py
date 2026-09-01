#!/usr/bin/env python3
"""THE DEPTH CURVES, side by side: 1m-demo baseline vs plain-on-current-head vs ship config.

Reads the rung JSONs from receipts/c1..c3 and prints the tables the window exists to produce.
Every demo number is quoted from the banked receipts (never paraphrased):
research/glm53-flash-bringup-20260827/1m-demo-20260829/box-receipts/.
usage: agg.py /root/out-1m
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/out-1m")

# rung -> (chars, demo prompt_tokens, demo prefill_s, demo prefill_tok_s, demo decode span)
DEMO = {
    "R16K":  (64400,   15766,   91.303,  172.68, 24.47),
    "R131K": (527000,  128566,  749.729, 171.48, 22.82),
    "R262K": (1054000, 257775, 1519.864, 169.60, 21.13),
    "R525K": (2161700, 525616, 3169.777, 165.82, 18.92),
    "R1M":   (4282700, 1035357, 6419.765, 161.28, 16.04),
}
ORDER = ["W1K", "R16K", "R131K", "R262K", "R525K", "R1M"]


def load(cell):
    out = {}
    d = ROOT / "receipts" / cell
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if "label" not in j:
            continue
        out[(j["label"], j.get("mode"))] = j
    return out


c1, c2, c3 = load("c1"), load("c2"), load("c3")
plain = {**c2, **c1}     # cell 1 owns the 1M plain rung
ship = c3


def fmt(v, w=8, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def ok(j):
    return j and j.get("status") == 200 and not j.get("error") and (j.get("usage") or {}).get("prompt_tokens")


print("=" * 108)
print("TABLE 1 — THE 1M PRIME, and prefill vs depth: 1m-demo baseline vs CURRENT HEAD (plain)")
print("=" * 108)
print(f"{'rung':6} {'prompt_tok':>11} {'demo TTFD s':>12} {'demo tok/s':>11} "
      f"{'now TTFD s':>11} {'now tok/s':>10} {'speedup':>8}")
for r in ORDER:
    if r not in DEMO:
        continue
    j = plain.get((r, "greedy"))
    ch, dpt, dts, dtps, _ = DEMO[r]
    if not ok(j):
        print(f"{r:6} {dpt:>11} {dts:>12.1f} {dtps:>11.2f} {'MISSING':>11} {'-':>10} {'-':>8}")
        continue
    pt = j["usage"]["prompt_tokens"]
    ntps, nts = j["prefill_tok_s"], j["prefill_s"]
    print(f"{r:6} {pt:>11} {dts:>12.1f} {dtps:>11.2f} {nts:>11.1f} {ntps:>10.2f} "
          f"{ntps/dtps:>7.2f}x")
    if pt != dpt:
        print(f"       NOTE prompt_tokens {pt} != demo {dpt} (same char slice; tokenizer count is the truth)")

print()
print("=" * 108)
print("TABLE 2 — THE DEPTH DECAY CURVES, decode tok/s: demo | plain now | SHIP CONFIG now")
print("           (greedy = the instrument; the ship column's vendor twin is table 3)")
print("=" * 108)
print(f"{'rung':6} {'prompt_tok':>11} {'demo':>7} | {'plain span':>10} {'plain p50':>10} | "
      f"{'ship span':>10} {'ship p50':>9} | {'ship/plain':>10}")
for r in ORDER:
    pj, sj = plain.get((r, "greedy")), ship.get((r, "greedy"))
    dd = DEMO.get(r, (None,) * 5)[4]
    pt = (pj or sj or {}).get("usage", {}).get("prompt_tokens") if (pj or sj) else None
    if not (ok(pj) or ok(sj)):
        continue
    ratio = ""
    if ok(pj) and ok(sj) and pj["decode_tok_s"]:
        ratio = f"{sj['decode_tok_s']/pj['decode_tok_s']:>9.3f}x"
    print(f"{r:6} {str(pt):>11} {fmt(dd,7)} | "
          f"{fmt(pj.get('decode_tok_s') if pj else None,10)} {fmt(pj.get('decode_steady_tok_s') if pj else None,10)} | "
          f"{fmt(sj.get('decode_tok_s') if sj else None,10)} {fmt(sj.get('decode_steady_tok_s') if sj else None,9)} | "
          f"{ratio:>10}")

print()
print("=" * 108)
print("TABLE 3 — VENDOR-DEFAULT SAMPLED rows (the real traffic shape: a request with NO sampling params)")
print("=" * 108)
print(f"{'rung':6} {'arm':6} {'prompt_tok':>11} {'TTFD s':>9} {'prefill':>9} {'decode':>8} {'p50':>8} {'ct':>5}")
for r in ORDER:
    for arm, src in (("plain", plain), ("ship", ship)):
        j = src.get((r, "vendor"))
        if not ok(j):
            continue
        u = j["usage"]
        print(f"{r:6} {arm:6} {u['prompt_tokens']:>11} {fmt(j['prefill_s'],9,1)} "
              f"{fmt(j['prefill_tok_s'],9)} {fmt(j['decode_tok_s'],8)} "
              f"{fmt(j['decode_steady_tok_s'],8)} {u.get('completion_tokens'):>5}")

print()
print("=" * 108)
print("FAILED / MISSING RUNGS (named, never silently dropped)")
print("=" * 108)
any_fail = False
for arm, src in (("plain", plain), ("ship", ship)):
    for (label, mode), j in sorted(src.items()):
        if not ok(j):
            any_fail = True
            print(f"  {arm} {label}/{mode}: status={j.get('status')} err={str(j.get('error'))[:150]!r}")
if not any_fail:
    print("  none: every attempted rung returned a primed, decoded answer")

print()
print("SOURCES: demo columns quoted from 1m-demo-20260829/box-receipts/phase3-*.json,")
print("phase4-R524K.json, phase7-R1M.json. Demo arm was PRE-MLA-TC, PRE-grouped-prefill,")
print("PRE-spec and did NOT carry BF16_MMV/PP_BF16 - so 'speedup' is a whole-head re-price,")
print("not an isolated lever delta.")
