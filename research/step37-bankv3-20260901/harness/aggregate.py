#!/usr/bin/env python3
"""Aggregate milestone-4 pricing rows into the cumulative attribution table, and APPLY the
amendment rules rather than leaving them to a reader's judgement.

The protocol this implements was pinned in RESULTS.md before any card time was spent, so it
cannot be tuned to the answer: interleaved fresh boots x3, escalated to x5 when either rule
fires, every arm reporting its spread and every escalation naming the rule that fired.

  RULE A  within-arm spread of the per-boot decision median > 0.5%
  RULE B  the verdict (delta between adjacent cumulative arms) falls within 2x the pooled spread

Both are computed here and printed as ESCALATE/OK per arm and per delta. A cell that reports a
delta smaller than its own noise has measured nothing, and saying so is the point.

Aggregation rules, each with a reason:
  * only kind == "rep" rows count. smoke and warmup are excluded BY CONSTRUCTION (the warmup is
    the first full-length generation on a cold cache; including it would price the cache).
  * looped == true rows are excluded and COUNTED, because a looped completion repeats cheap
    high-accept tokens and inflates both tok/s and acceptance. The exclusion count is printed
    per arm; an exclusion that is not reported is indistinguishable from dropping a number.
  * the per-boot statistic is the MEDIAN over prompts, and the per-arm statistic is the MEDIAN
    over boots. Medians at both levels because one slow prompt is a prompt property, not an arm
    property, and one slow boot is a box property.
  * arms are keyed by peeling the row's OWN boot suffix (V3OFF1 + boot 1 -> V3OFF), so boots
    pool per arm and an arm name ending in a digit is not mangled (see `arm_key`).

Usage: aggregate.py <rows.jsonl> [--order V3OFF,V3SM,V3SMGU,V3SMGUD8]
"""
import json, statistics, sys
from collections import defaultdict

ROWS = sys.argv[1]
ORDER = None
for a in sys.argv[2:]:
    if a.startswith("--order="):
        ORDER = a.split("=", 1)[1].split(",")

def arm_key(row):
    """Strip THIS ROW'S OWN boot suffix, not a trailing-digit regex.

    BUG FOUND 2026-09-01, and it silently dropped a whole arm from the x3 table: `run-ab.sh`
    tags an arm `<ARM><boot>`, and stripping a trailing-digit regex takes the maximal digit
    run — so `V3SMGUD8` + boot `1` gave `V3SMGUD`, and the `--order` filter then discarded the
    one arm the cell was really about. Any arm name ENDING in a digit hits this. Peeling exactly
    the boot value cannot guess wrong, because the boot is carried in its own field.
    """
    tag, boot = row["arm"], str(row["boot"])
    return tag[: -len(boot)] if boot and tag.endswith(boot) else tag


rows = [json.loads(l) for l in open(ROWS) if l.strip()]
reps = [r for r in rows if r.get("kind") == "rep"]
by_arm_boot = defaultdict(lambda: defaultdict(list))
excluded = defaultdict(int)
md5s = defaultdict(set)
nonces = defaultdict(set)
for r in reps:
    arm = arm_key(r)
    md5s[arm].add(r.get("bin_md5"))
    nonces[arm].add(r.get("boot_nonce"))
    if r.get("looped"):
        excluded[arm] += 1
        continue
    by_arm_boot[arm][r["boot"]].append(r)

METRICS = ("wall_tok_s", "decode_tok_s", "ttft_s", "spec_acc")


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


summary = {}
for arm, boots in by_arm_boot.items():
    per_boot = {}
    for b, rs in sorted(boots.items()):
        per_boot[b] = {m: med([r.get(m) for r in rs]) for m in METRICS}
        per_boot[b]["n"] = len(rs)
    dec_meds = [per_boot[b]["decode_tok_s"] for b in per_boot if per_boot[b]["decode_tok_s"]]
    wall_meds = [per_boot[b]["wall_tok_s"] for b in per_boot if per_boot[b]["wall_tok_s"]]
    summary[arm] = {
        "per_boot": per_boot,
        "boots": len(per_boot),
        "excluded_loops": excluded[arm],
        "bin_md5": sorted(x for x in md5s[arm] if x),
        "n_distinct_nonces": len(nonces[arm]),
        "decode_median": med(dec_meds),
        "wall_median": med(wall_meds),
        # spread = full range of the per-boot DECISION median, as a fraction of that median.
        "decode_spread_pct": (100.0 * (max(dec_meds) - min(dec_meds)) / med(dec_meds))
        if len(dec_meds) > 1 and med(dec_meds) else 0.0,
        "wall_spread_pct": (100.0 * (max(wall_meds) - min(wall_meds)) / med(wall_meds))
        if len(wall_meds) > 1 and med(wall_meds) else 0.0,
        "ttft_median": med([per_boot[b]["ttft_s"] for b in per_boot]),
        "acc_median": med([per_boot[b]["spec_acc"] for b in per_boot]),
    }

order = ORDER or sorted(summary)
missing = [a for a in order if a not in summary]
extra = [a for a in summary if a not in order]
order = [a for a in order if a in summary]
# A requested arm that is not in the data, or a measured arm that is not in the order, is LOUD.
# Silently filtering either is how the x3 pass reported a three-row table for a four-arm cell.
for a in missing:
    print("ORDER_FAIL: arm %r was requested but has no rows" % a)
for a in extra:
    print("ORDER_WARN: arm %r has rows but is not in --order (it will NOT be priced)" % a)
if missing:
    raise SystemExit("refusing to print a table that is missing a requested arm")

print("arm            boots  n_nonce  excl  decode  spread%%   wall  spread%%   ttft   acc   md5")
for a in order:
    s = summary[a]
    print(
        "%-14s %5d  %7d  %4d  %6.2f  %6.2f  %6.2f  %6.2f  %5.3f  %4.3f  %s"
        % (a, s["boots"], s["n_distinct_nonces"], s["excluded_loops"], s["decode_median"],
           s["decode_spread_pct"], s["wall_median"], s["wall_spread_pct"],
           s["ttft_median"], s["acc_median"], ",".join(x[:8] for x in s["bin_md5"]))
    )
    # IDENTITY: one binary per arm, one nonce per boot. Both are gates, not notes.
    if len(s["bin_md5"]) != 1:
        print("   IDENTITY_FAIL %s: %d distinct binaries in one arm" % (a, len(s["bin_md5"])))
    if s["n_distinct_nonces"] != s["boots"]:
        print("   IDENTITY_FAIL %s: %d nonces for %d boots — arms may share a boot"
              % (a, s["n_distinct_nonces"], s["boots"]))

print()
escalate = []
for a in order:
    s = summary[a]
    for label, key in (("decode", "decode_spread_pct"), ("wall", "wall_spread_pct")):
        if s[key] > 0.5:
            print("RULE A FIRES  arm=%s %s within-arm spread %.2f%% > 0.5%%" % (a, label, s[key]))
            escalate.append((a, "A", label, s[key]))
        else:
            print("rule A ok     arm=%s %s within-arm spread %.2f%%" % (a, label, s[key]))

print()
print("CUMULATIVE ATTRIBUTION (each row is the delta to the row above — that delta IS the")
print("program's contribution, which is what one env var arming three programs could not give)")
print("step                       decode   d_decode    wall    d_wall   pooled_spread  verdict")
prev = None
for a in order:
    s = summary[a]
    if prev is None:
        print("%-24s  %7.2f        --  %6.2f        --" % (a, s["decode_median"], s["wall_median"]))
    else:
        dd = s["decode_median"] - prev["decode_median"]
        dw = s["wall_median"] - prev["wall_median"]
        ddp = 100.0 * dd / prev["decode_median"]
        dwp = 100.0 * dw / prev["wall_median"]
        # pooled spread: the larger of the two arms' decode spreads, in tok/s.
        pooled = max(s["decode_spread_pct"], prev["decode_spread_pct"]) / 100.0 * s["decode_median"]
        within2x = abs(dd) < 2 * pooled
        verdict = "WITHIN 2x POOLED SPREAD -> RULE B FIRES" if within2x else "separated"
        if within2x:
            escalate.append((a, "B", "decode", abs(dd)))
        print(
            "%-24s  %7.2f  %+6.2f (%+5.2f%%)  %6.2f  %+6.2f (%+5.2f%%)  %6.3f tok/s  %s"
            % (a, s["decode_median"], dd, ddp, s["wall_median"], dw, dwp, pooled, verdict)
        )
    prev = s

# total, against the bundle price the perf chain could only report as one number
if len(order) >= 2:
    lo, hi = summary[order[0]], summary[order[-1]]
    print()
    print("TOTAL %s -> %s: decode %.2f -> %.2f (%+.2f%%), wall %.2f -> %.2f (%+.2f%%)"
          % (order[0], order[-1], lo["decode_median"], hi["decode_median"],
             100.0 * (hi["decode_median"] - lo["decode_median"]) / lo["decode_median"],
             lo["wall_median"], hi["wall_median"],
             100.0 * (hi["wall_median"] - lo["wall_median"]) / lo["wall_median"]))

print()
if escalate:
    print("ESCALATE TO x5 — rules fired:")
    for a, rule, label, v in escalate:
        print("  arm=%s rule=%s metric=%s value=%.3f" % (a, rule, label, v))
else:
    print("NO ESCALATION — every arm's within-arm spread <= 0.5%% and every verdict clears 2x pooled")
json.dump(summary, open(ROWS.replace(".jsonl", "-summary.json"), "w"), indent=1, default=str)
