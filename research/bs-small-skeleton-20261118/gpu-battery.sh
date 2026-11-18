#!/usr/bin/env bash
# The GPU battery the shared small-batch skeleton change needs before it merges. Prepared on 2026-11-18 and NOT
# run: no RTX PRO 6000 box was free. The change's device code is the same SASS kernel for kernel (README.md), so
# every count below is the base arm's count, and the two arms must agree line for line.
#
# Arms: BASE = the commit the change sits on (801243f06e while it stacks on memra#1337; after the rebase, the main
# commit it rebased onto), HEAD = the PR head. Each arm is built from its own source tree into its own target dir.
#
# usage: gpu-battery.sh PHASE        PHASE: build | kgate | tp4 | pair | all
# env (no defaults; the script refuses an unset one):
#   OUT        receipt dir (created; one subdir a phase)
#   SRC_BASE, SRC_HEAD   source trees of the two arms (full checkouts)
#   DATA       the V4.1 lane's data dir: fixtures-gpu/rand, captures-v2/needle, oracle/engram_constants.token_map.i32
#   CKPT       DeepSeek-V4.1-Flash-1017 checkpoint dir (rev 1bae0e10)
#   CARD1      one RTX PRO 6000 (index in nvidia-smi order) for the kernel gate
#   CARDS4     four RTX PRO 6000 of one box, comma-separated, for the tensor-parallel gates
#   PAIR       two RTX PRO 6000 sharing a box, comma-separated, for the pair battery (not a serving box)
#   BOARD      a TSV of the board models present on the pair box: name <TAB> model path <TAB> token-ids file
#              (every model of research/tune-data/current-board.json `supported_models` that the box holds;
#              the report lists the absent ones)
#   MEMRA_MODELS_DIR   as tools/release-battery.sh and tools/local-ci.sh read it
# Locks: the caller holds the box's per-card locks for the cards a phase uses (one scored campaign a box,
# AGENTS.md "Evidence discipline"); this script takes none and prints the cards' state at each phase start.
#
# Expected (the base arm's figures from memra#1337's receipts, research/modelplan-onboarding-dsv41-20261109/perf/
# receipts/moe1324/; both arms must print the same):
#   kgate  `dsv41-gate kernels DATA/fixtures-gpu/rand`: "dsv41-gate: 1089 checks, 0 failed" on each arm
#          (80 small-row checks, 40 on each ring; the ring seam forces both rings on the one card)
#   tp4    `dsv41-gate tp` needle, layers 1,2,14,20,21,24, four ranks, decode 2, MEMRA_FULL_PREC=1,
#          MEMRA_DSV41_TP_ISSUE=parallel: default 916 checks 0 failed, peer-pull 914 / 0, host-canonical 914 / 0;
#          180 HASH lines a transport, equal between the arms
#   pair   the full battery of AGENTS.md on the pair: kernel-check with both cells manifests "ALL GREEN (<n>
#          cells, <m> skipped)", m <= 11 and n, m equal between the arms; tools/release-battery.sh exit 0 (the
#          argmax margin gate on every roster row, both decode programs, and run-spec K=1..8 with its
#          STEP-IDENTITY line); run-gen argmax "MATCH" on every BOARD model; run-spec K=1..8 on every BOARD model
#          with a drafter: eight "self-consistency: PASS" lines and "=== SELF-CONSISTENCY PASS ===".
#          No board model reaches the V4.1 route's kernels (only cu/dsv41_gpu.cu and cu/dsv41_cutlass.cu include
#          the headers), so this phase is expected green on both arms; it is run because AGENTS.md asks the full
#          battery of every .cu kernel and kernel dispatch change.
set -u
PHASE=${1:?usage: gpu-battery.sh build|kgate|tp4|pair|all}
for v in OUT SRC_BASE SRC_HEAD; do [ -n "${!v:-}" ] || { echo "set $v"; exit 2; }; done
mkdir -p "$OUT"
say() { echo "$(date +%T) $*" | tee -a "$OUT/battery.log"; }
cards() { nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu,memory.used --format=csv,noheader \
  | tee -a "$OUT/battery.log"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | tee -a "$OUT/battery.log"; }
bin() { echo "$OUT/target-$1/release/$2"; }
arms="BASE HEAD"
src() { [ "$1" = BASE ] && echo "$SRC_BASE" || echo "$SRC_HEAD"; }

build() {
  for a in $arms; do
    s=$(src "$a")
    say "build $a $(git -C "$s" rev-parse HEAD)"
    (cd "$s" && MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR="$OUT/target-$a" \
      cargo build --release --locked -p memra-engine --bin dsv41-gate) > "$OUT/build-$a.log" 2>&1 \
      || { say "build $a FAILED"; return 1; }
  done
  # the pair battery's binaries, HEAD only first; BASE's kernel-check for the cell count comparison
  for a in $arms; do
    s=$(src "$a")
    (cd "$s" && MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR="$OUT/target-$a" \
      cargo build --release --locked -p memra-engine --bins) > "$OUT/build-bins-$a.log" 2>&1 || { say "bins $a FAILED"; return 1; }
  done
  for a in $arms; do sha256sum "$(bin "$a" dsv41-gate)" "$(bin "$a" kernel-check)" | tee -a "$OUT/battery.log"; done
}

kgate() {
  for v in DATA CARD1; do [ -n "${!v:-}" ] || { echo "set $v"; return 2; }; done
  mkdir -p "$OUT/kgate"; cards
  for a in $arms; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD1 MEMRA_FULL_PREC=1 \
      "$(bin "$a" dsv41-gate)" kernels "$DATA/fixtures-gpu/rand" --out "$OUT/kgate/kgate-$a.json" \
      > "$OUT/kgate/kgate-$a.log" 2>&1
    say "kgate $a rc $?: $(grep -a -o '[0-9]* checks, [0-9]* failed' "$OUT/kgate/kgate-$a.log" | tail -1)"
  done
  # equal check names and verdicts, line for line
  for a in $arms; do grep -a -E '^(PASS|FAIL)' "$OUT/kgate/kgate-$a.log" | cut -d: -f1 > "$OUT/kgate/names-$a.txt"; done
  cmp -s "$OUT/kgate/names-BASE.txt" "$OUT/kgate/names-HEAD.txt" && say "kgate: the same checks and verdicts" \
    || say "kgate: CHECK LISTS DIFFER"
}

tp4() {
  for v in DATA CKPT CARDS4; do [ -n "${!v:-}" ] || { echo "set $v"; return 2; }; done
  mkdir -p "$OUT/tp4"; cards
  for t in default peer-pull host-canonical; do
    for a in $arms; do
      if [ $t = default ]; then tr=(); else tr=(MEMRA_TP_TRANSPORT=$t); fi
      env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS4 MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel \
        "${tr[@]}" "$(bin "$a" dsv41-gate)" tp "$CKPT" "$DATA/captures-v2/needle" --layers 1,2,14,20,21,24 \
        --devices 0,1,2,3 --decode 2 --token-map "$DATA/oracle/engram_constants.token_map.i32" \
        --out "$OUT/tp4/tp-$t-$a.json" > "$OUT/tp4/tp-$t-$a.log" 2>&1
      rc=$?
      grep -a '^HASH' "$OUT/tp4/tp-$t-$a.log" > "$OUT/tp4/tp-$t-$a.hash"
      say "tp4 $t $a rc $rc: $(grep -a -o '[0-9]* checks, [0-9]* failed' "$OUT/tp4/tp-$t-$a.log" | tail -1)," \
        "$(wc -l < "$OUT/tp4/tp-$t-$a.hash") HASH lines"
    done
    cmp -s "$OUT/tp4/tp-$t-BASE.hash" "$OUT/tp4/tp-$t-HEAD.hash" && say "tp4 $t: HASH lines equal" \
      || say "tp4 $t: HASH LINES DIFFER"
  done
}

pair() {
  for v in PAIR BOARD MEMRA_MODELS_DIR; do [ -n "${!v:-}" ] || { echo "set $v"; return 2; }; done
  mkdir -p "$OUT/pair"; cards
  for a in $arms; do
    s=$(src "$a")
    (cd "$s" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PAIR "$(bin "$a" kernel-check)" \
      --require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells) \
      > "$OUT/pair/kernel-check-$a.log" 2>&1
    say "kernel-check $a rc $?: $(grep -a 'ALL GREEN' "$OUT/pair/kernel-check-$a.log" | tail -1)"
  done
  (cd "$SRC_HEAD" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PAIR CARGO_TARGET_DIR="$OUT/target-HEAD" \
    tools/release-battery.sh) > "$OUT/pair/release-battery-HEAD.log" 2>&1
  say "release-battery HEAD rc $?"
  while IFS=$'\t' read -r name model ids; do
    [ -z "$name" ] || [ "${name#\#}" != "$name" ] && continue
    if [ ! -e "$model" ]; then say "board $name: ABSENT ($model)"; continue; fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PAIR%%,*} "$(bin HEAD run-gen)" "$model" $(cat "$ids") \
      > "$OUT/pair/run-gen-$name.log" 2>&1
    say "run-gen $name rc $?: $(grep -a -c 'MATCH$' "$OUT/pair/run-gen-$name.log") MATCH," \
      "$(grep -a -c 'MISMATCH-STRUCTURED' "$OUT/pair/run-gen-$name.log") MISMATCH-STRUCTURED"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PAIR%%,*} MEMRA_SPEC_TEMP=0 "$(bin HEAD run-spec)" "$model" \
      $(cat "$ids") > "$OUT/pair/run-spec-$name.log" 2>&1
    say "run-spec $name rc $?: $(grep -a -c 'self-consistency: PASS' "$OUT/pair/run-spec-$name.log") per-K PASS," \
      "$(grep -a -c '=== SELF-CONSISTENCY PASS ===' "$OUT/pair/run-spec-$name.log") overall"
  done < "$BOARD"
}

case $PHASE in
  build) build ;;
  kgate) kgate ;;
  tp4) tp4 ;;
  pair) pair ;;
  all) build && kgate && tp4 && pair ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
say "BATTERY-DONE $PHASE"
