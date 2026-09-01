#!/usr/bin/env bash
# STRUCT-BATTERY CELL 4 — MAP MINTS (CPU, free; ep-place LANE §4 step 2): the SHARED tool
# per class x strategy {coactivation, frequency, even}, ranks 2, entry-rank 0,
# expert-count 288 (glm-config text_config: 288 routed experts), --decode-only (t=1 rows;
# matches the t=1 decode walk the cell-5 engine A/B prices). The ship-shape trace
# additionally gets a t<=8 pre-filter (verify rows kept, prime rows t>=16 dropped — a
# prime line's union is a whole chunk and would drown the co-activation signal) and a
# coactivation + even mint as a SENSITIVITY row, labeled, never the A/B input.
# Every map self-receipts per-layer stats; the summary below is the deliverable: the
# predicted per-class single-rank fraction (1 - peer_touch_fraction) BEFORE any A/B.
set -euo pipefail
TR=/root/out-struct/traces
MAPS=/root/out-struct/maps
TOOL=/root/memra-struct/tools/build_expert_placement_map.py
mkdir -p "$MAPS"

# ship-shape sensitivity input: keep decode + verify rows (t<=8), drop prime rows
awk '$2 <= 8' "$TR/agentic-ship.ids" > "$TR/agentic-ship-t1to8.ids"
awk '$2 <= 8' "$TR/agentic-ship.w"   > "$TR/agentic-ship-t1to8.w"
wc -l "$TR"/agentic-ship-t1to8.* | tee "$MAPS/ship-filter-receipt.txt"
sha256sum "$TR"/agentic-ship-t1to8.* | tee -a "$MAPS/ship-filter-receipt.txt"

mint() { # class strategy decode_only(1/'')
  local class="$1" strat="$2" donly="$3"
  local args=(--trace "$TR/$class.ids" --weight-trace "$TR/$class.w"
              --ranks 2 --entry-rank 0 --expert-count 288
              --strategy "$strat" --out "$MAPS/$class-$strat.json")
  [ -n "$donly" ] && args+=(--decode-only)
  python3 "$TOOL" "${args[@]}"
  sha256sum "$MAPS/$class-$strat.json"
}

for class in agentic-t1 prose-t1 l3-t1; do
  for strat in coactivation frequency even; do
    mint "$class" "$strat" 1
  done
done
# ship-shape sensitivity (t<=8 filtered; NOT --decode-only — verify rows are the point)
for strat in coactivation even; do
  mint "agentic-ship-t1to8" "$strat" ""
done

echo "=== C4 SELF-RECEIPTING STATS SUMMARY (the pre-A/B deliverable) ==="
python3 - "$MAPS" <<'PY' | tee "$MAPS/mint-stats-summary.txt"
import json, os, statistics as st, sys
maps = sorted(f for f in os.listdir(sys.argv[1]) if f.endswith(".json"))
print(f"{'map':44} {'layers':>6} {'peer_touch mean':>15} {'single-rank':>11} "
      f"{'worst-layer pt':>14} {'exp_max_touch':>13} {'even_max':>9}")
rows = {}
for f in maps:
    m = json.load(open(os.path.join(sys.argv[1], f)))
    pts = [r["stats"]["peer_touch_fraction"] for r in m["layers"]]
    exm = [r["stats"]["expected_max_rank_touch"] for r in m["layers"]]
    evm = [r["stats"]["even_baseline_expected_max_rank_touch"] for r in m["layers"]]
    pt = st.mean(pts)
    rows[f] = {r["layer"]: r["stats"]["peer_touch_fraction"] for r in m["layers"]}
    print(f"{f:44} {len(pts):>6} {pt:>15.4f} {1-pt:>11.4f} "
          f"{max(pts):>14.4f} {st.mean(exm):>13.3f} {st.mean(evm):>9.3f}")
# per-layer coactivation-vs-even scan (the fixture demo's layer-1 finding: greedy can
# LOSE to even on individual layers — count and name them per class)
for cls in ("agentic-t1", "prose-t1", "l3-t1", "agentic-ship-t1to8"):
    co, ev = rows.get(f"{cls}-coactivation.json"), rows.get(f"{cls}-even.json")
    if not co or not ev:
        continue
    worse = [(l, co[l], ev[l]) for l in co if co[l] > ev[l] + 1e-12]
    print(f"[{cls}] coactivation vs even: {len(worse)}/{len(co)} layers WORSE than even"
          + (f" -> {[(l, round(c,3), round(e,3)) for l, c, e in worse[:8]]}" if worse else ""))
PY
echo "C4_DONE"
