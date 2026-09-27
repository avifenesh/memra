#!/usr/bin/env bash
# Day 23 (lead ruling 26), TARGET CARD (one RTX PRO 6000 Blackwell, run from /root/wt-b on the box; --rig pro-single, /tmp/memra-gpu.lock): the standard `run-gen` argmax gate and the `run-spec` K=1..8
# self-consistency gate on the MERGED tree (origin/main 653c997f4, #614 `small_m_tier_max`; the lane's scope removed)
# with the served Qwen3.8-27B NVFP4-Q5K mint, whose `-mtp.gguf` carries the MTP drafter (no MEMRA_MTP_DRAFT: the
# embedded NextN head is the drafter; run-spec refuses loudly if the artifact has none). Prompts:
#   std     CONTRIBUTING.md's raw-id validation prompt `9419 11 1814 0` (4 tokens; the run-gen validation-gate path)
#   probe   tools/fast-gate/prompts/probe.txt (the run-spec battery prompt; text, >= 16 tokens, so run-gen also
#           prints the batched-prime line)
#   p16     exactly 16 raw ids (rtx5090-day23/prompts/p16-ids.txt: the first 16 tokens of the SERVING.md head)
#   p4112   exactly 4112 raw ids (= 16 mod 4096; rtx5090-day23/prompts/p4112-ids.txt), the cold-prime schedule
#           4096 + 16, whose last chunk is the 16-row shape memra#427 named
# run-gen: MEMRA_NGEN=8; run-spec: MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32, the naked K=1..8 sweep (as tools/local-ci.sh).
# Pass/fail, not timed, through the collector (--rig pro-single, /tmp/memra-gpu.lock). Bounded lock retries, never
# kills a holder. Verdict lines are grepped out of each arm's full log, which is kept.
# usage: run-day23-genspec-box3.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
WT=${WT:-/root/wt-b}
R=${R:-/root/spill-receipts/b-day23}
GEN=${GEN:-$WT/target/release/run-gen}
SPEC=${SPEC:-$WT/target/release/run-spec}
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
PROBE=${PROBE:-$WT/tools/fast-gate/prompts/probe.txt}
P16=${P16:-/root/spill-receipts/b-day23/prompts/p16-ids.txt}
P4112=${P4112:-/root/spill-receipts/b-day23/prompts/p4112-ids.txt}
run() { "$@"; }
cd "$WT" || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; GEN="$1"; SPEC="$2"; M="$3"; PROBE="$4"; P16="$5"; P4112="$6"; O="$7"; mkdir -p "$O"
gen() { name="$1"; shift; echo "== run-gen $name: $*"
  env MEMRA_NGEN=8 "$GEN" "$M" "$@" > "$O/gen-$name.log" 2>&1; echo "exit=$?" >> "$O/gen-$name.log"
  grep -E "argmax=|MATCH|MISMATCH|batched-prime|prompt tokens|text prompt|^exit=" "$O/gen-$name.log" | cut -c1-300; }
genf() { name="$1"; f="$2"; echo "== run-gen $name: MEMRA_PROMPT_FILE=$f"
  env MEMRA_NGEN=8 MEMRA_PROMPT_FILE="$f" "$GEN" "$M" > "$O/gen-$name.log" 2>&1; echo "exit=$?" >> "$O/gen-$name.log"
  grep -E "argmax=|MATCH|MISMATCH|batched-prime|prompt tokens|text prompt|^exit=" "$O/gen-$name.log" | cut -c1-300; }
spec() { name="$1"; shift; echo "== run-spec $name: $*"
  env MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$SPEC" "$M" "$@" > "$O/spec-$name.log" 2>&1; echo "exit=$?" >> "$O/spec-$name.log"
  grep -E "text prompt|^K=|self-consistency|acceptance|PASS|FAIL|ERROR|^exit=" "$O/spec-$name.log" | cut -c1-300; }
specf() { name="$1"; f="$2"; echo "== run-spec $name: MEMRA_PROMPT_FILE=$f"
  env MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$f" "$SPEC" "$M" > "$O/spec-$name.log" 2>&1; echo "exit=$?" >> "$O/spec-$name.log"
  grep -E "text prompt|^K=|self-consistency|acceptance|PASS|FAIL|ERROR|^exit=" "$O/spec-$name.log" | cut -c1-300; }
gen std 9419 11 1814 0
genf probe "$PROBE"
gen p16 $(cat "$P16")
gen p4112 $(cat "$P4112")
specf probe "$PROBE"
spec p16 $(cat "$P16")
spec p4112 $(cat "$P4112")'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" genspec "$GEN" "$SPEC" "$MODEL" "$PROBE" "$P16" "$P4112" "$out/cell" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$GEN" "$SPEC" "$MODEL" > "$out/binary.sha256"; sha256sum "$PROBE" "$P16" "$P4112" > "$out/prompt.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
