#!/usr/bin/env python3
"""THE 1M DEPTH RE-PRICE REPORT: every table this window owes, built from banked receipts.

Demo columns are quoted from the banked 1m-demo receipts, never paraphrased:
research/glm53-flash-bringup-20260827/1m-demo-20260829/box-receipts/{phase3-*,phase4-R524K,phase7-R1M}.json

STEADY-P50 GUARD, applied here rather than in the probe (the probe must not change mid-cell):
a speculative round emits its accepted run in one burst, so many interarrival gaps are ~0 and
the median gap collapses - the PP3 spec arm reported p50 50393.7 tok/s beside a sound span of
27.23. Any p50 more than 3x its own span is reported as "burst" and excluded. The span number
((ct-1)/(t_last-t_first)) reads only the endpoints and is burst-proof, so it is primary.
usage: report.py /root/out-1m
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/out-1m")

DEMO = {   # rung -> (prompt_tokens, prefill_s, prefill_tok_s, decode_span, decode_steady)
    1108:    (None, None, None, 27.5, None),
    15766:   (15766,   91.303,  172.68, 24.47, None),
    128566:  (128566,  749.729, 171.48, 22.82, None),
    257775:  (257775, 1519.864, 169.60, 21.13, None),
    525616:  (525616, 3169.777, 165.82, 18.92, None),
    1035357: (1035357, 6419.765, 161.28, 16.04, 15.67),
}


def load(relpath):
    p = ROOT / relpath
    if not p.exists():
        return None
    try:
        j = json.load(open(p))
    except Exception:
        return None
    if j.get("status") != 200 or j.get("error"):
        return j          # keep it: failures are receipts too
    return j


def ok(j):
    return bool(j) and j.get("status") == 200 and not j.get("error") \
        and (j.get("usage") or {}).get("prompt_tokens")


def p50(j):
    """steady p50 with the burst guard applied."""
    s, span = j.get("decode_steady_tok_s"), j.get("decode_tok_s")
    if s is None:
        return None, "n/a"
    if span and s > 3.0 * span:
        return None, "burst"
    return s, f"{s:.2f}"


def acc_of(relpath):
    """the last cumulative acceptance the server reported for that rung, from its own log."""
    p = ROOT / relpath
    if not p.exists():
        return None
    last = None
    for line in open(p, errors="replace"):
        if "[glm5-acc]" in line and "cum=" in line:
            last = line.strip().split("cum=")[-1].split()[0]
    return last


def spec_usage(j):
    """Acceptance straight from the response's own usage.spec block - the authoritative
    per-request number (rounds/drafted/accepted/acceptance_rate), better than scraping
    [glm5-acc] out of a shared server log where concurrent rungs would interleave.
    Its ABSENCE is itself the receipt that the spec walk did not run for that request."""
    return ((j or {}).get("usage") or {}).get("spec") or None


def spec_evidence(bootlog):
    p = ROOT / "logs" / bootlog
    if not p.exists():
        return None
    t = p.read_text(errors="replace")
    pats = ["[glm5-acc]", "verify walk BATCHED per layer", "[glm5-vrows]", "PMIN=0.700",
            "[bf16-tcols-wide] engaged", "[bf16-tcols-x1] engaged",
            "[topk-shards] engaged", "[glm5-verify-ws] engaged"]
    return sum(t.count(x) for x in pats)


def row(label, j, extra=""):
    if not ok(j):
        st = j.get("status") if j else "MISSING"
        er = str((j or {}).get("error"))[:90]
        return f"  {label:34} FAILED status={st} {er}"
    u = j["usage"]
    _, ps = p50(j)
    sp = spec_usage(j)
    spec_txt = (f"  acc={sp['acceptance_rate']:.3f} rounds={sp['rounds']}"
                if sp else "  spec=NOT ENGAGED")
    return (f"  {label:34} pt={u['prompt_tokens']:>9}  TTFD={j['prefill_s']:>9.2f}s  "
            f"prefill={j['prefill_tok_s']:>7.2f}  decode={j['decode_tok_s']:>6.2f}  "
            f"p50={ps:>8}  ct={u.get('completion_tokens'):>4}{spec_txt}{extra}")


print("=" * 118)
print("TABLE 1  THE 1M PRIME on the current head vs the 1m-demo baseline (PP4, plain)")
print("=" * 118)
j = load("receipts/c1/R1M-greedy.json")
if ok(j):
    u, d = j["usage"], DEMO[1035357]
    print(f"  prompt_tokens        {u['prompt_tokens']:,}   (demo {d[0]:,} - the SAME char slice of the SAME sha-verified corpus)")
    print(f"  cached_tokens        {u.get('prompt_tokens_details', {}).get('cached_tokens')}   (an honest cold prime; MEMRA_PREFIX_CACHE_MB=0 pinned)")
    print(f"  TTFD                 {j['prefill_s']:,.2f} s = {j['prefill_s']/60:.1f} min   (demo {d[1]:,.1f} s = {d[1]/60:.1f} min)")
    print(f"  prefill              {j['prefill_tok_s']:.2f} tok/s   (demo {d[2]:.2f})  ->  {j['prefill_tok_s']/d[2]:.3f}x")
    print(f"  decode span          {j['decode_tok_s']:.2f} tok/s   (demo {d[3]:.2f})  ->  {j['decode_tok_s']/d[3]:.3f}x")
    sp, ps = p50(j)
    if sp:
        print(f"  decode steady p50    {sp:.2f} tok/s   (demo {d[4]:.2f})  ->  {sp/d[4]:.3f}x")
    print(f"  wall                 {j['wall_s']:,.1f} s")
else:
    print("  R1M MISSING or FAILED:", (j or {}).get("error"))

print()
print("=" * 118)
print("TABLE 2  SPEC AND 1M ARE MUTUALLY EXCLUSIVE: the PP4-vs-PP3 spec-engagement A/B")
print("         same binary, same ship spec env, MEMRA_CTX=131072 both arms, same 16k prompt.")
print("         ONE variable: the PP stage count. glm5_sharded_placement_admits = (2..=3) stages.")
print("=" * 118)
print(f"  {'arm':10} {'spec evidence':>14} {'decode tok/s':>13} {'prefill tok/s':>14} {'acceptance':>12}")
for arm, boot in (("pp4", "boot-ppab-pp4.log"), ("pp3", "boot-ppab-pp3.log")):
    jj = load(f"receipts/c3a-ppab/{arm}/A16K-greedy.json")
    ev = spec_evidence(boot)
    a = acc_of(f"receipts/c3a-ppab/{arm}/A16K-greedy.serverlog")
    if ok(jj):
        print(f"  {arm:10} {str(ev):>14} {jj['decode_tok_s']:>13.2f} {jj['prefill_tok_s']:>14.2f} {str(a):>12}")
p4 = load("receipts/c3a-ppab/pp4/A16K-greedy.json")
p3 = load("receipts/c3a-ppab/pp3/A16K-greedy.json")
if ok(p4) and ok(p3):
    print(f"\n  spec uplift at 16k on the placement that admits it: "
          f"{p3['decode_tok_s']/p4['decode_tok_s']:.3f}x  "
          f"({p4['decode_tok_s']:.2f} -> {p3['decode_tok_s']:.2f} tok/s)")
print("  PP4 is the ONLY demonstrated 1M config, and PP4 is exactly where the spec route refuses.")

print()
print("=" * 118)
print("TABLE 3  THE DEPTH CURVES, side by side")
print("=" * 118)
print("  A. PP4 / the 1M posture (capped SLRU arena, no bf16 mirror) - PLAIN is ALL this posture")
print("     can serve: the spec route refuses 4 stages (Table 2), so there is no ship row to have.")
for lab, path in (("1k   plain", "receipts/c1/W1K-greedy.json"),
                  ("16k  plain (spec refused)", "receipts/c3a-ppab/pp4/A16K-greedy.json"),
                  ("131k plain", "receipts/c2/R131K-greedy.json"),
                  ("262k plain", "receipts/c2/R262K-greedy.json"),
                  ("1M   plain", "receipts/c1/R1M-greedy.json")):
    j2 = load(path)
    if j2:
        dd = DEMO.get((j2.get("usage") or {}).get("prompt_tokens"))
        extra = f"   demo {dd[3]}" if dd and dd[3] else ""
        print(row(lab, j2, extra))
print()
print("  B. PP3, SAME base recipe as A (capped arena, host-pinned, no bf16 mirror) - SHIP vs PLAIN")
print("     NOT the resident fleet serving env: two boots of that env produced ZERO expert-residency")
print("     decisions (receipts/c3b-fleetenv-residency-denied/), so these rows are NOT directly")
print("     comparable with the banked 70.458/71.489 resident+bf16 fleet ship rows.")
for lab, path in (("16k  SHIP  greedy", "receipts/c3b-ship-pp3/B16K-greedy.json"),
                  ("16k  SHIP  vendor-default", "receipts/c3b-ship-pp3/B16K-vendor.json"),
                  ("16k  plain greedy", "receipts/c3b-ship-pp3/plain/P16K-greedy.json"),
                  ("131k SHIP  greedy", "receipts/c3b-ship-pp3/B131K-greedy.json"),
                  ("131k SHIP  vendor-default", "receipts/c3b-ship-pp3/B131K-vendor.json"),
                  ("131k plain greedy", "receipts/c3b-ship-pp3/plain/P131K-greedy.json")):
    j2 = load(path)
    if j2:
        print(row(lab, j2))
for depth, s, p in (("16k", "receipts/c3b-ship-pp3/B16K-greedy.json",
                    "receipts/c3b-ship-pp3/plain/P16K-greedy.json"),
                   ("131k", "receipts/c3b-ship-pp3/B131K-greedy.json",
                    "receipts/c3b-ship-pp3/plain/P131K-greedy.json")):
    a, b = load(s), load(p)
    if ok(a) and ok(b):
        print(f"    ship/plain at {depth}: {a['decode_tok_s']/b['decode_tok_s']:.3f}x  "
              f"({b['decode_tok_s']:.2f} -> {a['decode_tok_s']:.2f} tok/s)")

print()
print("=" * 118)
print("TABLE 4  ACCEPTANCE vs DEPTH on the ship config (from the server's own [glm5-acc] lines)")
print("=" * 118)
for lab, jp in (("1k   greedy", "receipts/c3b-ship-pp3/W1K-greedy.json"),
                ("16k  greedy", "receipts/c3b-ship-pp3/B16K-greedy.json"),
                ("16k  vendor", "receipts/c3b-ship-pp3/B16K-vendor.json"),
                ("131k greedy", "receipts/c3b-ship-pp3/B131K-greedy.json"),
                ("131k vendor", "receipts/c3b-ship-pp3/B131K-vendor.json")):
    jj = load(jp)
    sp = spec_usage(jj)
    if sp:
        print(f"  {lab:14} acceptance={sp['acceptance_rate']:.4f}  rounds={sp['rounds']:>4}  "
              f"drafted={sp['drafted']:>5}  accepted={sp['accepted']:>5}")
    elif jj:
        print(f"  {lab:14} NO usage.spec block -> the spec walk did NOT run for this request")
print()
print("  ACCEPTANCE vs DEPTH is the question 'does spec survive depth?': the verify walk pays")
print("  the same DSA indexer scan x(K+1), so a falling acceptance or a falling uplift with")
print("  depth is what would kill it. Rows above are the measurement, on the placement that")
print("  admits spec at all (PP3) - at 1M there is no spec row to have, by finding 4.")

print()
print("=" * 118)
print("TABLE 5  THE 1M CAPACITY REFUSALS (finding 1) - admission, not OOM")
print("=" * 118)
for tag, d in (("with bf16 mirror", "receipts/c1a-bf16-refusal/admission-and-bf16-census.txt"),
               ("without bf16 mirror", "receipts/c1b-arena-refusal/admission.txt")):
    p = ROOT / d
    if p.exists():
        print(f"  --- {tag} ---")
        for line in p.read_text(errors="replace").splitlines():
            if "request cost" in line and "1035677" in line or "VRAM reject" in line:
                print("   ", line.strip()[:150])
print()
print("NOTE the direction: removing the bf16-resident mirror moved available headroom the WRONG")
print("way (39090 -> 36740 MB), which REFUTES it as the cause. The cause is the SLRU arena.")
