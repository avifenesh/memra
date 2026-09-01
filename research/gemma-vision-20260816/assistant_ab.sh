#!/usr/bin/env bash
# Official gemma4-assistant MTP vs DSpark dflash — interleaved A/B (lane/gemma-assistant)
# Per rep the arms run back-to-back on the same card (interleaved A/B law); each gate run
# carries its own plain baseline. Pairing law measured on 5090: assistant heads must match
# their trunk's weight lineage (QAT head on QAT trunk 0.573 vs official head 0.344 there).
set -uo pipefail
EV=/data/memra/evidence/gemma-assistant-ab
mkdir -p "$EV"
G=/data/memra/memra-assistant/target/release/gemma-gate
T_Q4=/data/memra/models/gemma4-31b/gemma-4-31B_q4_0-it.gguf
T_NV=/data/memra/models/gemma4-31b/gemma-4-31B-it-NVFP4mix.gguf
DSPARK=/data/memra/models/gemma4-31b/dspark
MTP_QAT=/data/memra/models/gemma4-31b/gemma-4-31B-it-Q8_0-MTP.gguf
MTP_OFF_F16=/data/memra/models/gemma4-31b/gemma-4-31B-it-official-F16-MTP.gguf
MTP_OFF_Q8=/data/memra/models/gemma4-31b/gemma-4-31B-it-official-Q8_0-MTP.gguf
# private ranks copy — fresh .learned sidecar, never touches the drafter lane's file
R=$EV/ranks-32768.txt
[ -f "$R" ] || cp /data/memra/evidence/gemma-drafter/gemma31b-ranks-32768.gguf.txt "$R"
DEV=${DEV:-0}
# receipt-exact prose prompt (accept5.sh); code prompt = same chat frame, code-class content
PROSE="2 105 2364 107 155122 1217 14820 3927 4146 236764 607 614 2591 236761 106 107 105 4368 107"
CODE="2 105 2364 107 6974 496 17856 1292 600 130450 1156 19372 12809 15852 236764 1299 8082 1061 990 532 2557 16783 236761 106 107 105 4368 107"

run_dflash() { # $1 trunk  $2 ngen  $3... ids
  local t=$1 n=$2; shift 2
  env -u MEMRA_DRAFT -u MEMRA_GEMMA_DRAFT_RANKS CUDA_VISIBLE_DEVICES=$DEV \
    MEMRA_SPEC_DFLASH=$DSPARK MEMRA_SPEC_STATS=1 MEMRA_NGEN=$n \
    "$G" "$t" $@ 2>&1 | grep -E "acceptance|plain:" | tail -2
}
run_mtp() { # $1 trunk  $2 ngen  $3 drafter  $4 ranks 0|1  $5... ids
  local t=$1 n=$2 d=$3 rk=$4; shift 4
  local envs=(CUDA_VISIBLE_DEVICES=$DEV MEMRA_SPEC=5 MEMRA_DRAFT=$d MEMRA_SPEC_STATS=1 MEMRA_NGEN=$n)
  [ "$rk" = 1 ] && envs=("${envs[@]}" MEMRA_GEMMA_DRAFT_RANKS=$R MEMRA_GEMMA_TRIM_ADAPT=512)
  env -u MEMRA_SPEC_DFLASH "${envs[@]}" "$G" "$t" $@ 2>&1 \
    | grep -E "accept-rate|plain:" | tail -2
}

q4_battery() { # $1 class  $2 ngen  $3... ids
  local cls=$1 n=$2; shift 2
  for rep in 1 2 3 4 5; do
    echo "=== q4/$cls rep $rep ==="
    echo "--- dflash(dspark) ---";        run_dflash "$T_Q4" "$n" $@
    echo "--- mtp qat-Q8 K=5 ---";        run_mtp "$T_Q4" "$n" "$MTP_QAT" 0 $@
    echo "--- mtp official-F16 K=5 ---";  run_mtp "$T_Q4" "$n" "$MTP_OFF_F16" 0 $@
  done
}
nv_battery() { # $1 class  $2 ngen  $3... ids
  local cls=$1 n=$2; shift 2
  for rep in 1 2 3 4 5; do
    echo "=== nv/$cls rep $rep ==="
    echo "--- dflash(dspark) ---";              run_dflash "$T_NV" "$n" $@
    echo "--- mtp official-F16 K=5 ---";        run_mtp "$T_NV" "$n" "$MTP_OFF_F16" 0 $@
    echo "--- mtp official-Q8+ranks K=5 ---";   run_mtp "$T_NV" "$n" "$MTP_OFF_Q8" 1 $@
  done
}

case "${1:-all}" in
  q4)
    q4_battery prose 128 $PROSE 2>&1 | tee "$EV/ab-q4-prose.log"
    q4_battery code  256 $CODE  2>&1 | tee "$EV/ab-q4-code.log"
    ;;
  nv)
    nv_battery prose 128 $PROSE 2>&1 | tee "$EV/ab-nv-prose.log"
    nv_battery code  256 $CODE  2>&1 | tee "$EV/ab-nv-code.log"
    ;;
  all)
    "$0" nv
    "$0" q4
    ;;
esac
echo AB-DONE
