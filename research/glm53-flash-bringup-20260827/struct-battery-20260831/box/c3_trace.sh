#!/usr/bin/env bash
# STRUCT-BATTERY CELL 3 — REAL-TRAFFIC TRACE CAPTURE (untimed; ep-place LANE §4 step 1):
# MEMRA_MOE_TRACE (ids) + MEMRA_MOE_WEIGHT_TRACE (hotness) armed together (the exact
# pair gate arm T proved byte-identical + row-exact), serving the real pools per traffic
# class through the SERVED path. Three PLAIN boots give pure t=1 decode rows (the shape
# the TP-2 engine A/B decodes and the tool's --decode-only consumes); one SPEC ship-shape
# boot (agentic pool) banks the verify-row structure (t=K+1 rows; split at mint time by
# the row's own t). The tap forces observation mode and door D fail-closes under it BY
# DESIGN — no timing number leaves any traced boot. Greedy instrument on real prompts;
# traces carry expert ids/weights only (no prompt text), so they bank cleanly.
set -uo pipefail
OUT=/root/out-struct/c3
TR=/root/out-struct/traces
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
AGENTIC_TAGS=d00-code,d01-code,d02-code,d03-code,d04-code,d05-code
PROSE_TAGS=d06-prose,d07-prose,d08-prose,d09-prose
mkdir -p "$OUT" "$TR"

trace_boot() { # name class pool tags(or '') spec(1/'')
  local name="$1" class="$2" pool="$3" tags="$4" spec="$5"
  local ids="$TR/$class.ids" w="$TR/$class.w"
  : > "$ids"; : > "$w"
  local extras=(MEMRA_MOE_TRACE="$ids" MEMRA_MOE_WEIGHT_TRACE="$w")
  [ -n "$spec" ] && extras=("${DFL[@]}" "${extras[@]}")
  echo "######## C3 TRACE BOOT $name (class=$class spec=${spec:-0}) ########"
  /root/out-struct/serve.sh start "c3-$name" "${extras[@]}" || { echo "C3_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-struct/run_pool.py sample --out "$OUT/$name" || { echo "C3_${name}_EXIT=SAMPLEFAIL"; return 1; }
  # class purity: the tap re-opens the file per appended line (trace_moe_routes), so
  # truncating here drops the boot-health sample's rows (d00 greedy 64, prime included)
  # and leaves the class pool as the trace's ONLY content.
  : > "$ids"; : > "$w"
  if [ -n "$tags" ]; then
    python3 /root/out-struct/run_pool.py cell --out "$OUT/$name" --pool "$pool" --mode greedy --max-tokens 256 --tags "$tags"
  else
    python3 /root/out-struct/run_pool.py cell --out "$OUT/$name" --pool "$pool" --mode greedy --max-tokens 256
  fi
  /root/out-struct/serve.sh engage "c3-$name" "${extras[@]}" || { echo "C3_${name}_EXIT=ENGAGEFAIL"; return 1; }
  /root/out-struct/serve.sh stop
  echo "--- trace receipts $class ---" | tee -a "$OUT/trace-receipts.txt"
  for f in "$ids" "$w"; do
    { wc -l "$f"; sha256sum "$f"; } | tee -a "$OUT/trace-receipts.txt"
  done
  # t-histogram: how many rows per t value (decode t=1 vs verify 2..15 vs prime >=16)
  awk '{h[$2]++} END {for (t in h) print "t="t" rows="h[t]}' "$ids" | sort -t= -k2 -n \
    | tee -a "$OUT/trace-receipts.txt"
  echo "C3_${name}_EXIT=0"
}

rc=0
trace_boot ag-plain   agentic-t1 decode "$AGENTIC_TAGS" ""  || rc=1
trace_boot pr-plain   prose-t1   decode "$PROSE_TAGS"   ""  || rc=1
trace_boot l3-plain   l3-t1      l3     ""              ""  || rc=1
trace_boot ag-ship    agentic-ship decode "$AGENTIC_TAGS" 1 || rc=1

echo "=== LOOP-LAW SCREEN (c3 tapes; looped rows would bias routing traces) ==="
python3 /root/out-struct/looplaw_screen.py "$OUT"/*/
gzip -kf "$TR"/*.ids "$TR"/*.w
ls -la "$TR"
echo "C3_DONE rc=$rc"
