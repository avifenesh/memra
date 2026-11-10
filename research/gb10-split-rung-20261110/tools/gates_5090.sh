#!/usr/bin/env bash
# Rig: the deep rung's RTX 5090 gate cells, one gpulock lease per cell, untimed pass/fail. Each
# cell reaches the 5090's rung (256 keys past 65,536 with Qwen3.8-27B's 4 KV heads): kernel-check
# with its rows cases to 262k, and the 27B's run-spec, spec-serve-gate and argmax margin (eager and
# batch) on prompts that straddle 65,536. Ornith-1.5-35B-A3B does not fit a deep cache on this card.
# Each cell starts only after no landing lease or ticket has shown for 5 minutes (landing_quiet.sh).
# usage: gates_5090.sh HEAD_BIN_DIR SRC_DIR OUTDIR [CELL...]
set -u
HB=$1; SRC=$2; OUT=$3; shift 3
M=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
P=$HOME/.cache/memra-split-rung/prompts
mkdir -p "$OUT" "$OUT/kt/release"
for b in "$HB"/*; do ln -sf "$b" "$OUT/kt/release/"; done
declare -A TTL CMD
ORDER=()
add() { TTL[$1]=$2; CMD[$1]=$3; ORDER+=("$1"); }
add kc 25 "env MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 $HB/kernel-check"
add rs-27b-65516 40 "env MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P/deep-65516.txt $HB/run-spec $M"
add ssg-27b-65516 30 "env MEMRA_PROMPT_FILE=$P/deep-65516.txt $HB/spec-serve-gate $M --steps 8 --ts 1,2,4"
add am-27b-65542-e 75 "env CARGO_TARGET_DIR=$OUT/kt bash $SRC/tools/argmax-margin-gate.sh $M --prompt $P/deep-65542.txt --logdir $OUT/am-27b-65542-e"
add am-27b-65542-b 75 "env CARGO_TARGET_DIR=$OUT/kt bash $SRC/tools/argmax-margin-gate.sh $M --prompt $P/deep-65542.txt --decode batch --logdir $OUT/am-27b-65542-b"
L=$OUT/driver.log
cells=("$@"); [ ${#cells[@]} -gt 0 ] || cells=("${ORDER[@]}")
for c in "${cells[@]}"; do
  [ -n "${CMD[$c]:-}" ] || { echo "$(date -Is) unknown cell $c" >> "$L"; continue; }
  echo "$(date -Is) $c start" >> "$L"
  waitfor --until "bash $(dirname "$0")/landing_quiet.sh" --every 20s --timeout 180m --quiet >> "$L" 2>&1
  gpulock run --ttl "${TTL[$c]}m" --wait 300m --for "memra deep split rung: 5090 gate cell $c (untimed pass/fail, one cell per lease)" -- \
    bash -c "${CMD[$c]}" > "$OUT/$c.log" 2>&1
  echo "$(date -Is) $c rc=$?" >> "$L"
done
echo "$(date -Is) all 5090 gate cells done" >> "$L"
