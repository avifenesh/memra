#!/usr/bin/env bash
# tp2-battery engine-probe launcher (window: tp2-box agent, cards 0,1 [+2 for PP-3 arm],
# port NONE — engine-level; the served calibration boot uses serve.sh).
# One binary, env-selected arms (comparability): this script owns the arm env tables so
# every boot of an arm is byte-identical env. Scoped stop is unnecessary: the probe is a
# foreground process of this shell; a window abort kills the shell's process group only.
set -uo pipefail
OUT=${OUT:-/root/out-tp2}
BIN=${BIN:-/root/memra-tp2/target/release/glm5-tp2-box-probe}
MODEL=/root/models/glm53-nvfp4
mkdir -p "$OUT/logs"

arm=$1; mode=$2; prompts=$3; run=$4; shift 4   # extra env as "$@"

# scrub inherited MEMRA_* (flip-battery discipline), then the arm table
unsets=()
while IFS='=' read -r k _; do case "$k" in MEMRA_*|BOXP_*) unsets+=(-u "$k");; esac; done < <(env)

common=(NVIDIA_TF32_OVERRIDE=0 MEMRA_BF16_MMV=1)
case "$arm" in
  pp3)    armenv=(CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 \
                  MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 \
                  MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16) ;;
  tp2)    armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 \
                  MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16) ;;
  tp2red) armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 \
                  MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16 MEMRA_GLM5_TP_GATE_RED=swap-wo) ;;
  plain1) armenv=(CUDA_VISIBLE_DEVICES=0 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 \
                  MEMRA_ST_PINNED=1) ;;
  *) echo "unknown arm $arm (pp3|tp2|tp2red|plain1)"; exit 2 ;;
esac

name="$arm-$mode-$run"
log="$OUT/logs/probe-$name.log"
echo "[tp2b] arm=$arm mode=$mode prompts=$prompts run=$run extras=$* bin=$BIN" | tee "$log"
sha256sum "$BIN" | tee -a "$log"
env "${unsets[@]}" "${common[@]}" "${armenv[@]}" BOXP_MODE="$mode" "$@" \
  "$BIN" "$MODEL" "$prompts" "$OUT/$name" 2>>"$log" | tee -a "$log"
rc=${PIPESTATUS[0]}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log"
echo "[tp2b] rc=$rc arm=$arm mode=$mode run=$run" | tee -a "$log"
exit $rc
