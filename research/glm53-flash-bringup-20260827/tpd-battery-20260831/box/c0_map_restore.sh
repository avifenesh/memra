#!/usr/bin/env bash
# tpd-battery CELL 0 — RESTORE THE BANKED COACTIVATION MAP (untimed, no timing leaves this
# cell). Why this cell exists: struct-battery's cell-4 mints lived in /root/out-struct,
# which its close removed, and the map JSONs were NOT banked (only c4-mint.log with their
# sha256s). The placement A/B input must therefore be RE-MINTED and its sha256 checked
# against the banked value:
#
#   agentic-t1-coactivation.json  56dea5ca5a2502f2b7558e8339f5c49eac5259e5b7a1489c1b2843b7fe81e2d4
#
# The mint is stdlib Python, no RNG, and the map embeds its input trace PATH + sha256, so an
# exact sha match is a THREE-way receipt at once: the trace reproduced byte-identically on a
# different build (the ep-diet head with both diet doors OFF is routing-identical to the
# struct head on the served plain path), the mint is deterministic, and the A/B input is the
# banked artifact rather than a look-alike. Paths are therefore recreated EXACTLY as
# struct-battery had them (/root/out-struct/traces/...), which is what its map rows name.
#
# Trace capture is struct-battery c3_trace.sh's `ag-plain` boot verbatim: served PLAIN boot,
# MEMRA_MOE_TRACE + MEMRA_MOE_WEIGHT_TRACE armed together, agentic tags d00..d05, greedy 256,
# then the class-purity truncation that drops the boot-health sample's rows.
set -uo pipefail
OUT=/root/out-tpd/c0
TR=/root/out-struct/traces          # the banked map rows name THIS path — reproduce it
MAPS=/root/out-tpd/maps
TOOL=/root/memra-tpd/tools/build_expert_placement_map.py
AGENTIC_TAGS=d00-code,d01-code,d02-code,d03-code,d04-code,d05-code
WANT=56dea5ca5a2502f2b7558e8339f5c49eac5259e5b7a1489c1b2843b7fe81e2d4
mkdir -p "$OUT" "$TR" "$MAPS"

ids="$TR/agentic-t1.ids"; w="$TR/agentic-t1.w"
: > "$ids"; : > "$w"
extras=(MEMRA_MOE_TRACE="$ids" MEMRA_MOE_WEIGHT_TRACE="$w")
echo "######## C0 TRACE BOOT (served plain, agentic-t1) ########"
/root/out-tpd/serve.sh start c0-ag-plain "${extras[@]}" || { echo "C0_EXIT=BOOTFAIL"; exit 1; }
python3 /root/out-tpd/run_pool.py sample --out "$OUT/ag-plain" || { echo "C0_EXIT=SAMPLEFAIL"; exit 1; }
# class purity: the tap re-opens the file per appended line, so truncating here drops the
# boot-health sample's rows and leaves the class pool as the trace's ONLY content.
: > "$ids"; : > "$w"
python3 /root/out-tpd/run_pool.py cell --out "$OUT/ag-plain" --pool decode --mode greedy \
  --max-tokens 256 --tags "$AGENTIC_TAGS" || { echo "C0_EXIT=CELLFAIL"; exit 1; }
/root/out-tpd/serve.sh stop
{
  echo "--- trace receipts agentic-t1 ---"
  wc -l "$ids" "$w"
  sha256sum "$ids" "$w"
  awk '{h[$2]++} END {for (t in h) print "t="t" rows="h[t]}' "$ids" | sort -t= -k2 -n
} | tee "$OUT/trace-receipts.txt"

echo "=== LOOP-LAW SCREEN (c0 tapes; looped rows would bias the routing trace) ==="
python3 /root/out-tpd/looplaw_screen.py "$OUT"/*/

echo "######## C0 MINT (shared tool, ranks 2, entry-rank 0, expert-count 288, --decode-only) ########"
python3 "$TOOL" --trace "$ids" --weight-trace "$w" --ranks 2 --entry-rank 0 \
  --expert-count 288 --strategy coactivation --decode-only \
  --out "$MAPS/agentic-t1-coactivation.json" | tee -a "$OUT/mint.log"
python3 "$TOOL" --trace "$ids" --weight-trace "$w" --ranks 2 --entry-rank 0 \
  --expert-count 288 --strategy even --decode-only \
  --out "$MAPS/agentic-t1-even.json" | tee -a "$OUT/mint.log"
got=$(sha256sum "$MAPS/agentic-t1-coactivation.json" | cut -d' ' -f1)
sha256sum "$MAPS"/*.json | tee -a "$OUT/mint.log"
echo "banked_sha=$WANT" | tee -a "$OUT/mint.log"
echo "minted_sha=$got" | tee -a "$OUT/mint.log"
if [ "$got" = "$WANT" ]; then
  echo "C0_MAP_SHA=REPRODUCED" | tee -a "$OUT/mint.log"
else
  echo "C0_MAP_SHA=DIFFERS — comparing per-layer assignments+stats against the banked stats" \
    | tee -a "$OUT/mint.log"
  python3 - "$MAPS/agentic-t1-coactivation.json" <<'PY' | tee -a "$OUT/mint.log"
import json, statistics as st, sys
m = json.load(open(sys.argv[1]))
pts = [r["stats"]["peer_touch_fraction"] for r in m["layers"]]
exm = [r["stats"]["expected_max_rank_touch"] for r in m["layers"]]
evm = [r["stats"]["even_baseline_expected_max_rank_touch"] for r in m["layers"]]
print(f"layers={len(pts)} peer_touch_mean={st.mean(pts):.4f} single_rank={1-st.mean(pts):.4f} "
      f"worst_layer_pt={max(pts):.4f} exp_max_touch={st.mean(exm):.3f} even_max={st.mean(evm):.3f}")
print("banked (struct c4): layers=42 peer_touch_mean=0.6084 single_rank=0.3916 "
      "worst_layer_pt=0.9173 exp_max_touch=6.894 even_max=5.079")
print("traces row:", m.get("traces"))
PY
fi
echo "C0_DONE"
