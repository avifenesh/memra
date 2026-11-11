#!/usr/bin/env bash
# Rig: lever 5's cells at the new main (11f3c0920b) on one GB10 unit after the rebase, one
# sparklock lease per cell through the lane's unit_cell.sh (a landing / release waiter goes first
# at every cell boundary). Cells:
#   build   spark_build_11f3.sh: B' = main 11f3c0920b, W' = rebased head code d117bf8607 (servers)
#   ssB ssW the short sampled warm == cold pair case on B' and on W' (main's parked pair)
#   c01B c02W   one B'/W' cell of the 131k campaign (campaign bt131k-11f3)
# usage: l5_11f3.sh UNIT CELL...
set -u
UNIT=$1; shift
S=$HOME/.cache/memra-specpub
L=$S/logs/driver-$UNIT.log
log() { echo "$(date -Is) l5-11f3: $*" >> "$L"; }
cd "$S" || exit 1
R='$HOME/.cache/memra-specpub'
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
DR27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf'
ENV27="MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_MTP_DRAFT=$DR27"
SAMPLED="_TEMP=1.0 _SEED=1234 _TRACE=1"
B=base-11f3c0920b; W=head-d117bf8607
pair() { # pair NAME ARM_DIR
  bash tools/unit_cell.sh "$UNIT" "$1" 14 "cd box && CELL_PORT=18193 PAIR_MODEL=$M27 timeout 12m python3 specpub_pairs.py $R/cells/$1/out \"$1=$R/bins/$2/memra-server|deep38|64,64,64|$ENV27 $SAMPLED\""
}
for c in "$@"; do
  case $c in
    build) bash tools/unit_cell.sh "$UNIT" build-11f3 60 "bash box/spark_build_11f3.sh"; rc=$?; log "build-11f3 rc=$rc"; [ $rc = 0 ] || exit 1 ;;
    ssB) pair pairs-short-sampled-B11f3 "$B"; log "pairs-short-sampled-B11f3 rc=$?" ;;
    ssW) pair pairs-short-sampled-W11f3 "$W"; log "pairs-short-sampled-W11f3 rc=$?" ;;
    c01B) bash tools/unit_cell.sh "$UNIT" bt131k-11f3-01-B 15 "bash box/specpub_cell.sh bt131k-11f3 01-B $B $W 131k none W=$W && grep -q 'SPECPUB-CELL-DONE 01-B' runs/bt131k-11f3/campaign.log"; log "bt131k-11f3-01-B rc=$?" ;;
    c02W) bash tools/unit_cell.sh "$UNIT" bt131k-11f3-02-W 15 "bash box/specpub_cell.sh bt131k-11f3 02-W $B $W 131k none W=$W && grep -q 'SPECPUB-CELL-DONE 02-W' runs/bt131k-11f3/campaign.log"; log "bt131k-11f3-02-W rc=$?" ;;
    *) log "unknown cell $c" ;;
  esac
done
log "done: $*"
