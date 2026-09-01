#!/usr/bin/env bash
# tpd-battery CELL 2 + CELL 4 — DIET PRICING and THE PLACEMENT A/B ON THE DIETED WALK, in
# ONE interleaved timed block (TIMED: the caller holds /root/TIMING-IN-FLIGHT for the whole
# block). Three arms per round, always in the same order, fresh boot each:
#
#   v1       TP-2 pair, MEMRA_GLM5_EP_DIET=0  (the banked v1 walk: 22.65 tok/s engine twin)
#   diet     TP-2 pair, MEMRA_GLM5_EP_DIET=1  (cell 2's headline)
#   dietmap  diet + MEMRA_GLM5_EP_MAP=<coactivation>  (cell 4: the composition verdict)
#
# One block rather than two because the map arm's ONLY meaningful comparator is the dieted
# even arm measured under the same box conditions: interleaving all three ×3 gives cell 2's
# diet/v1 ratio and cell 4's map/diet ratio from the same drift envelope (interleaved-A/B
# protocol law: box clock drift invalidates cross-run perf claims).
#
# Escalation to ×5 on anomaly: within-arm decode-median relative spread > 0.5%, or a verdict
# gap within 2× the pooled spread. `report` prints the rule hits.
#
# Greedy + ONE vendor-default sampled row per boot (BOXP_SAMPLED=1), 256-token cap, 128-token
# floor applied by agg.py with named exclusions. TRAP (struct-battery, receipted): tpd_arm.sh
# SCRUBS inherited BOXP_*, so these ride as TRAILING ARGS — never env-prefix.
set -uo pipefail
OUT=/root/out-tpd
A=$OUT/tpd_arm.sh
cd "$OUT"
ARMS=(v1 diet dietmap)
# The DEEP (l3) block is CELL 3's instrument: TTFT/prefill at 0.4k + 4626 tok. Its third arm
# is the grouped EP prime rather than the map, because the map's prime effect is already
# measured on the pool block (struct-battery saw prime +10.7% under the map) while the
# grouped prime is a PREFILL door and only the deep row can price it.
ARMS_DEEP=(v1 diet dietgp)

case "${1:?pool <round..> | deep <round..> | report | report-deep}" in
pool)
  shift; rc=0
  for i in "$@"; do
    for arm in "${ARMS[@]}"; do
      echo "######## C2 POOL round=$i arm=$arm ########"
      bash "$A" "$arm" timed "$OUT/prompts-decode" "t$i" BOXP_SAMPLED=1 BOXP_MAX_NEW=256 || rc=1
    done
  done
  echo "C2_POOL_ROUNDS_DONE: $* rc=$rc"
  ;;
deep)
  shift; rc=0
  for i in "$@"; do
    for arm in "${ARMS_DEEP[@]}"; do
      echo "######## C3 DEEP round=$i arm=$arm ########"
      bash "$A" "$arm" timed "$OUT/prompts-l3" "t${i}l" BOXP_MAX_NEW=256 || rc=1
    done
  done
  echo "C2_DEEP_ROUNDS_DONE: $* rc=$rc"
  ;;
report)
  echo "=== C2 AGGREGATE (medians, 128-token floor named) ==="
  python3 "$OUT/agg.py" "$OUT"/v1-timed-* "$OUT"/diet-timed-* "$OUT"/dietmap-timed-* \
    | tee "$OUT/analysis/agg-c2.txt"
  echo "=== C2 LOOP-LAW SCREEN ==="
  python3 "$OUT/looplaw_screen.py" "$OUT"/v1-timed-* "$OUT"/diet-timed-* "$OUT"/dietmap-timed-* \
    | tee "$OUT/analysis/looplaw-c2.txt"
  echo "=== C2/C4 RELATIVE DELTAS + ENGAGEMENT-COUNTER RECONCILIATION ==="
  python3 - "$OUT" <<'PY' | tee "$OUT/analysis/verdict-c2.txt"
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]
ARMS = ("v1", "diet", "dietmap")

def boots(arm, suffix=""):
    """(decode-median per boot, per-boot counter dicts) over <arm>-timed-t<N><suffix>."""
    meds, prime, ttft, vend, ctrs = [], [], [], [], []
    pat = os.path.join(out, f"{arm}-timed-t*")
    for d in sorted(glob.glob(pat)):
        base = os.path.basename(d)
        tail = base.split("-timed-")[1]
        if suffix == "l" and not tail.endswith("l"):
            continue
        if suffix == "" and tail.endswith("l"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
        ok = [r for r in rows if r["arm"] == "greedy" and r["out_tokens"] >= 128]
        if ok:
            meds.append(st.median(r["decode_tok_s"] for r in ok))
            prime.append(st.median(r["prime_s"] for r in ok))
            ttft.append(st.median(r["ttft_s"] for r in ok))
        for r in rows:
            if r["arm"] == "vendor" and r["out_tokens"] >= 128:
                vend.append(r["decode_tok_s"])
        log = os.path.join(out, "logs", f"probe-{base}.log")
        txt = open(log).read()
        c = {}
        m = re.findall(r"ep-peer-slot-dispatches=(\d+)", txt)
        if m:
            c["peer_slots"] = int(m[-1])
        m = re.findall(r"ep-diet-counters (.*)", txt)
        if m:
            for kv in m[-1].split():
                k, _, v = kv.partition("=")
                if v.isdigit():
                    c[k] = int(v)
        c["boot"] = base
        ctrs.append(c)
    return meds, prime, ttft, vend, ctrs

def rel(a, b):
    return f"{a/b:.4f}" if b else "n/a"

res = {}
for arm in ARMS:
    res[arm] = boots(arm)
    meds, prime, ttft, vend, ctrs = res[arm]
    if not meds:
        print(f"{arm}: NO ROWS")
        continue
    med = st.median(meds)
    spread = (max(meds) - min(meds)) / med if len(meds) > 1 else 0.0
    print(f"{arm:8} decode boot-medians {[round(x,3) for x in meds]} median {med:.3f} "
          f"spread {100*spread:.3f}% | prime med {st.median(prime):.3f}s "
          f"| pool TTFT med {st.median(ttft):.3f}s | vendor {[round(x,2) for x in vend]}")
    for c in ctrs:
        print(f"         counters {c}")

base = res["v1"][0]
pooled = 0.0
for arm in ARMS:
    m = res[arm][0]
    if len(m) > 1:
        pooled = max(pooled, max(m) - min(m))
print(f"\npooled spread (max within-arm range) = {pooled:.3f} tok/s")
if base:
    bm = st.median(base)
    for arm in ("diet", "dietmap"):
        m = res[arm][0]
        if not m:
            continue
        am = st.median(m)
        gap = am - bm
        print(f"{arm}/v1 = {rel(am, bm)}  (gap {gap:+.3f} tok/s)"
              + ("  ESCALATE_TO_X5: RULE(b) |gap| <= 2x pooled" if abs(gap) <= 2 * pooled else ""))
    dm, mm = res["diet"][0], res["dietmap"][0]
    if dm and mm:
        d, m2 = st.median(dm), st.median(mm)
        gap = m2 - d
        print(f"PLACEMENT SIGN ON THE DIETED WALK: dietmap/diet = {rel(m2, d)} "
              f"(gap {gap:+.3f} tok/s; struct-battery measured 0.9686 on the v1 walk)"
              + ("  ESCALATE_TO_X5: RULE(b)" if abs(gap) <= 2 * pooled else ""))
for arm in ARMS:
    m = res[arm][0]
    if len(m) > 1 and (max(m) - min(m)) / st.median(m) > 0.005:
        print(f"ESCALATE_TO_X5: RULE(a) within-arm spread > 0.5% on {arm}")
PY
  ;;
report-deep)
  echo "=== C3 DEEP (l3) AGGREGATE — the grouped-EP-prime deliverable ==="
  python3 "$OUT/agg.py" "$OUT"/v1-timed-*l "$OUT"/diet-timed-*l "$OUT"/dietgp-timed-*l \
    | tee "$OUT/analysis/agg-c3-deep.txt"
  echo "=== C3 LOOP-LAW SCREEN ==="
  python3 "$OUT/looplaw_screen.py" "$OUT"/v1-timed-*l "$OUT"/diet-timed-*l "$OUT"/dietgp-timed-*l \
    | tee "$OUT/analysis/looplaw-c3.txt"
  echo "=== C3 PER-PROMPT PRIME / TTFT / PREFILL tok/s (banked v1: 94.2 s @4626 tok = 49.1 tok/s) ==="
  python3 - "$OUT" <<'PY' | tee "$OUT/analysis/verdict-c3.txt"
import glob, json, os, re, statistics as st, sys
out = sys.argv[1]
rows = {}
for arm in ("v1", "diet", "dietgp"):
    for d in sorted(glob.glob(os.path.join(out, f"{arm}-timed-t*l"))):
        for line in open(os.path.join(d, "rows.jsonl")):
            r = json.loads(line)
            if r["arm"] != "greedy":
                continue
            rows.setdefault((arm, r["tag"]), []).append(r)
print(f"{'arm':8} {'tag':10} {'p_tok':>6} {'prime_s':>9} {'prefill tok/s':>13} "
      f"{'ttft_s':>8} {'decode tok/s':>12} {'n':>3}")
med = {}
for (arm, tag), rs in sorted(rows.items()):
    pt = rs[0]["prompt_tokens"]
    pr = st.median(r["prime_s"] for r in rs)
    tt = st.median(r["ttft_s"] for r in rs)
    dk = st.median(r["decode_tok_s"] for r in rs)
    med[(arm, tag)] = (pt, pr, tt, dk)
    print(f"{arm:8} {tag:10} {pt:6d} {pr:9.3f} {pt/pr:13.1f} {tt:8.3f} {dk:12.3f} {len(rs):3d}")
for tag in sorted({t for _, t in med}):
    b = med.get(("diet", tag)); g = med.get(("dietgp", tag)); v = med.get(("v1", tag))
    if b and g:
        print(f"[{tag}] grouped-prime TTFT {b[2]:.3f}s -> {g[2]:.3f}s "
              f"({b[2]/g[2]:.2f}x faster), prefill {b[0]/b[1]:.1f} -> {g[0]/g[1]:.1f} tok/s")
    if v and g:
        print(f"[{tag}] vs v1: TTFT {v[2]:.3f}s -> {g[2]:.3f}s ({v[2]/g[2]:.2f}x), "
              f"prefill {v[0]/v[1]:.1f} -> {g[0]/g[1]:.1f} tok/s")
print("\n--- grouped-prime engagement counters per deep boot ---")
for arm in ("v1", "diet", "dietgp"):
    for d in sorted(glob.glob(os.path.join(out, f"{arm}-timed-t*l"))):
        base = os.path.basename(d)
        txt = open(os.path.join(out, "logs", f"probe-{base}.log")).read()
        m = re.findall(r"ep-diet-counters (.*)", txt)
        ex = len(re.findall(r"\[glm5-ep-grouped-prime\] execute", txt))
        print(f"{base:24} grouped_prime_execute_lines={ex:6d} | {m[-1] if m else '<missing>'}")
PY
  ;;
esac
