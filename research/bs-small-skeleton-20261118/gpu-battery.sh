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
#   BOARD      a TSV of the board models: name <TAB> model path <TAB> token-ids file [<TAB> run-spec env words,
#              such as MEMRA_MTP_DRAFT=PATH] (every model of research/tune-data/current-board.json
#              `supported_models`; a row whose model file is absent fails the phase)
#   MEMRA_MODELS_DIR   as tools/release-battery.sh and tools/local-ci.sh read it
# Locks: the script takes the box lock AGENTS.md names for a 2x RTX PRO 6000 pair or a multi-card box
# (/tmp/memra-gpu.lock, MEMRA_GPU_LOCK the override) for every GPU phase, waits up to 30 minutes for it, and prints
# the cards' state at each phase start. Every check that misses its expected figure is a FAIL line; the last line is
# "BATTERY PASS PHASE" (exit 0) or "BATTERY FAIL PHASE: n failure(s)" (exit 1).
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
FAILS=0
say() { echo "$(date +%T) $*" | tee -a "$OUT/battery.log"; }
fail() { FAILS=$((FAILS + 1)); say "FAIL: $*"; }
cards() { nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu,memory.used --format=csv,noheader \
  | tee -a "$OUT/battery.log"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | tee -a "$OUT/battery.log"; }
bin() { echo "$OUT/target-$1/release/$2"; }
arms="BASE HEAD"
src() { [ "$1" = BASE ] && echo "$SRC_BASE" || echo "$SRC_HEAD"; }
count() { grep -a -o '[0-9]* checks, [0-9]* failed' "$1" | tail -1; }

# The box lock: one scored campaign a multi-card box (AGENTS.md, "Lock names are a correctness surface"). Every GPU
# phase runs under it; MEMRA_GPU_LOCK is the override seam AGENTS.md names.
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-gpu.lock}
gpu_lock() {
  exec {LFD}>>"$LOCK"
  flock -w 1800 "$LFD" || { fail "box lock $LOCK not taken in 30 minutes"; return 1; }
  say "box lock $LOCK held"
}

build() {
  for a in $arms; do
    s=$(src "$a")
    say "build $a $(git -C "$s" rev-parse HEAD)"
    (cd "$s" && MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR="$OUT/target-$a" \
      cargo build --release --locked -p memra-engine --bins) > "$OUT/build-$a.log" 2>&1 || { fail "build $a"; return 1; }
    sha256sum "$(bin "$a" dsv41-gate)" "$(bin "$a" kernel-check)" "$(bin "$a" run-gen)" "$(bin "$a" run-spec)" \
      | tee -a "$OUT/battery.log"
  done
}

kgate() {
  for v in DATA CARD1; do [ -n "${!v:-}" ] || { fail "set $v"; return 1; }; done
  mkdir -p "$OUT/kgate"; cards
  local want=${EXPECT_KGATE:-1089}
  for a in $arms; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD1 MEMRA_FULL_PREC=1 \
      "$(bin "$a" dsv41-gate)" kernels "$DATA/fixtures-gpu/rand" --out "$OUT/kgate/kgate-$a.json" \
      > "$OUT/kgate/kgate-$a.log" 2>&1
    rc=$?
    c=$(count "$OUT/kgate/kgate-$a.log")
    say "kgate $a rc $rc: $c"
    [ $rc -eq 0 ] && [ "$c" = "$want checks, 0 failed" ] || fail "kgate $a: rc $rc, '$c' (want $want checks, 0 failed)"
    # every check line in full (name, bytes, differing bytes, verdict), not only names and verdicts
    grep -a -E '^(PASS|FAIL)' "$OUT/kgate/kgate-$a.log" > "$OUT/kgate/lines-$a.txt"
  done
  if cmp -s "$OUT/kgate/lines-BASE.txt" "$OUT/kgate/lines-HEAD.txt"; then
    say "kgate: $(wc -l < "$OUT/kgate/lines-HEAD.txt") check lines equal between the arms"
  else
    fail "kgate: check lines differ between the arms (diff lines-BASE.txt lines-HEAD.txt)"
  fi
}

tp4() {
  for v in DATA CKPT CARDS4; do [ -n "${!v:-}" ] || { fail "set $v"; return 1; }; done
  mkdir -p "$OUT/tp4"; cards
  for t in default peer-pull host-canonical; do
    if [ $t = default ]; then want=${EXPECT_TP_DEFAULT:-916}; tr=(-u MEMRA_TP_TRANSPORT)
    else want=${EXPECT_TP_OTHER:-914}; tr=(MEMRA_TP_TRANSPORT=$t); fi
    for a in $arms; do
      env "${tr[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARDS4 MEMRA_FULL_PREC=1 \
        MEMRA_DSV41_TP_ISSUE=parallel "$(bin "$a" dsv41-gate)" tp "$CKPT" "$DATA/captures-v2/needle" \
        --layers 1,2,14,20,21,24 --devices 0,1,2,3 --decode 2 --token-map "$DATA/oracle/engram_constants.token_map.i32" \
        --out "$OUT/tp4/tp-$t-$a.json" > "$OUT/tp4/tp-$t-$a.log" 2>&1
      rc=$?
      grep -a '^HASH' "$OUT/tp4/tp-$t-$a.log" > "$OUT/tp4/tp-$t-$a.hash"
      c=$(count "$OUT/tp4/tp-$t-$a.log"); h=$(wc -l < "$OUT/tp4/tp-$t-$a.hash")
      say "tp4 $t $a rc $rc: $c, $h HASH lines"
      [ $rc -eq 0 ] && [ "$c" = "$want checks, 0 failed" ] && [ "$h" -eq "${EXPECT_TP_HASH:-180}" ] \
        || fail "tp4 $t $a: rc $rc, '$c', $h HASH lines (want $want checks, 0 failed, ${EXPECT_TP_HASH:-180} HASH)"
    done
    cmp -s "$OUT/tp4/tp-$t-BASE.hash" "$OUT/tp4/tp-$t-HEAD.hash" && say "tp4 $t: HASH lines equal" \
      || fail "tp4 $t: HASH lines differ between the arms"
  done
}

pair() {
  for v in PAIR BOARD MEMRA_MODELS_DIR; do [ -n "${!v:-}" ] || { fail "set $v"; return 1; }; done
  mkdir -p "$OUT/pair"; cards
  for a in $arms; do
    s=$(src "$a")
    (cd "$s" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PAIR "$(bin "$a" kernel-check)" \
      --require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells) \
      > "$OUT/pair/kernel-check-$a.log" 2>&1
    rc=$?
    g=$(grep -a -o 'ALL GREEN ([0-9]* cells, [0-9]* skipped)' "$OUT/pair/kernel-check-$a.log" | tail -1)
    say "kernel-check $a rc $rc: ${g:-no ALL GREEN line}"
    echo "$g" > "$OUT/pair/kernel-check-$a.green"
    sk=$(echo "$g" | grep -o '[0-9]* skipped' | grep -o '[0-9]*')
    [ $rc -eq 0 ] && [ -n "$g" ] && [ "${sk:-99}" -le 11 ] || fail "kernel-check $a: rc $rc, '${g:-none}'"
  done
  cmp -s "$OUT/pair/kernel-check-BASE.green" "$OUT/pair/kernel-check-HEAD.green" \
    || fail "kernel-check: cell or skip counts differ between the arms"
  (cd "$SRC_HEAD" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$PAIR CARGO_TARGET_DIR="$OUT/target-HEAD" \
    tools/release-battery.sh) > "$OUT/pair/release-battery-HEAD.log" 2>&1
  rc=$?; say "release-battery HEAD rc $rc"; [ $rc -eq 0 ] || fail "release-battery HEAD rc $rc"
  while IFS=$'\t' read -r name model ids spec_env; do
    [ -z "$name" ] || [ "${name#\#}" != "$name" ] && continue
    # the full battery needs every board model: an absent one fails the phase
    if [ ! -e "$model" ]; then fail "board $name: model ABSENT ($model)"; continue; fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PAIR%%,*} "$(bin HEAD run-gen)" "$model" $(cat "$ids") \
      > "$OUT/pair/run-gen-$name.log" 2>&1
    rc=$?
    # the argmax line ends in two spaces and MATCH; MISMATCH (and MISMATCH-STRUCTURED) counted on their own
    mt=$(grep -a -c '  MATCH$' "$OUT/pair/run-gen-$name.log"); mm=$(grep -a -c 'MISMATCH' "$OUT/pair/run-gen-$name.log")
    say "run-gen $name rc $rc: $mt MATCH, $mm MISMATCH"
    [ $rc -eq 0 ] && [ "$mt" -ge 1 ] && [ "$mm" -eq 0 ] || fail "run-gen $name: rc $rc, $mt MATCH, $mm MISMATCH"
    # shellcheck disable=SC2086  # spec_env: the model's run-spec settings, NAME=VALUE words
    env ${spec_env:-} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${PAIR%%,*} MEMRA_SPEC_TEMP=0 \
      "$(bin HEAD run-spec)" "$model" $(cat "$ids") > "$OUT/pair/run-spec-$name.log" 2>&1
    rc=$?
    pk=$(grep -a -c 'self-consistency: PASS' "$OUT/pair/run-spec-$name.log")
    ov=$(grep -a -c '=== SELF-CONSISTENCY PASS ===' "$OUT/pair/run-spec-$name.log")
    say "run-spec $name rc $rc: $pk per-K PASS, $ov overall"
    [ $rc -eq 0 ] && [ "$pk" -eq 8 ] && [ "$ov" -eq 1 ] || fail "run-spec $name: rc $rc, $pk per-K PASS, $ov overall"
  done < "$BOARD"
}

case $PHASE in
  build) build ;;
  kgate) gpu_lock && kgate ;;
  tp4) gpu_lock && tp4 ;;
  pair) gpu_lock && pair ;;
  all) build && gpu_lock && { kgate; tp4; pair; } ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
if [ "$FAILS" -eq 0 ]; then say "BATTERY PASS $PHASE"; exit 0; fi
say "BATTERY FAIL $PHASE: $FAILS failure(s), the FAIL lines above"
exit 1
