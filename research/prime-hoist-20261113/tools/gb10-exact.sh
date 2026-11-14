#!/usr/bin/env bash
# Runs ON GB10 unit-b (memra#1258), inside a sparklock lease: step 1 byte-identity on GB10 (its
# route takes the F8F4 tile, where the slab-writing form returns false, but the W placeholder, the
# embedding drop and the dequant reserve apply). Arms A and B: run-gen-<arm>-121a,
# run-spec-<arm>-121a, dspark-q38-gate-<arm>-121a. Prompts of 2,000 / 4,106 / 8,200 tokens (ids
# from run-gen's prefill-only mode on long.txt, texts from the GGUF vocabulary, made once in
# $R/run/exact-prompts): run-gen argmax at 4,106, run-spec K=1..8 at 4,106, dspark_q38_gate at all
# three. One arm per call (one lease each); gb10-compare.sh compares two arms' digests.
# usage: gb10-exact.sh ARM
set -uo pipefail
A=$1
R=$HOME/.cache/memra-1258
M=$HOME/models/memra-q38
GG=$M/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
echo "$$" >> "$R/pids.txt"
OUT=$R/run/exact-$A
P=$R/run/exact-prompts
mkdir -p "$OUT" "$P"
{ date -Is; for a in "$A"; do sha256sum "$R/bins/run-gen-$a-121a" "$R/bins/run-spec-$a-121a" "$R/bins/dspark-q38-gate-$a-121a"; done
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw --format=csv,noheader; uptime; } > "$OUT/guard.log"
if [ ! -s "$P/p8200.ids" ]; then
  MEMRA_PROMPT_FILE=$R/long.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
    timeout 5m "$R/bins/run-gen-$A-121a" "$GG" > "$P/ids.log" 2>&1
  python3 "$R/gguf-detok.py" "$GG" "$P/ids.log" "$P" 2000 4106 8200 | tee -a "$OUT/guard.log" || exit 2
fi
sha256sum "$P"/*.ids >> "$OUT/guard.log"
rc=0
for arm in "$A"; do
  for n in 4106; do
    ids=$(cat "$P/p$n.ids")
    # shellcheck disable=SC2086
    MEMRA_NGEN=32 timeout 15m "$R/bins/run-gen-$arm-121a" "$GG" $ids > "$OUT/$arm-rungen-$n.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$arm run-gen $n rc=$r $(grep -E 'MATCH|MISMATCH|FLIP-NEARTIE' "$OUT/$arm-rungen-$n.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  done
  ids=$(cat "$P/p4106.ids")
  # shellcheck disable=SC2086
  MEMRA_SPEC_TEMP=0 timeout 12m "$R/bins/run-spec-$arm-121a" "$GG" $ids > "$OUT/$arm-runspec-4106.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm run-spec 4106 rc=$r $(grep -E 'STEP-IDENTITY|=== SELF-CONSISTENCY' "$OUT/$arm-runspec-4106.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  MEMRA_PROMPT_DIR=$P timeout 10m "$R/bins/dspark-q38-gate-$arm-121a" "$GG" "$DR" 96 > "$OUT/$arm-dspark.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm dspark rc=$r $(grep -E '== dspark_q38_gate' "$OUT/$arm-dspark.log" | cut -c1-160)" | tee -a "$OUT/guard.log"
  grep -E '^\[dspark-q38-gate\]' "$OUT/$arm-dspark.log" | sed -E 's/ prime_s=.*//' > "$OUT/$arm-dspark-digests.txt"
  grep -E '^ *tokens: |self-consistency' "$OUT/$arm-rungen-4106.log" "$OUT/$arm-runspec-4106.log" \
    | sed -E 's|^[^:]*/[^-]*-||' > "$OUT/$arm-streams.txt"
done
echo "GB10-EXACT-DONE rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
exit $rc
