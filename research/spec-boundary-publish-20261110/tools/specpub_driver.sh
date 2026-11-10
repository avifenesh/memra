#!/usr/bin/env bash
# Rig: the spec boundary publication's GB10 cells on one unit, one sparklock lease per cell
# (unit_cell.sh; a landing waiter goes first at every cell boundary). Steps, each a list of cells:
#   build      spark_build.sh: both arms' sm_121a server and test binaries (CPU only)
#   unit       unit_gpu_cells.sh: the GPU unit cells on both arms' test binaries
#   bt131k, bt260k   the served B/T, B T T B B T, one fresh server per cell (specpub_cell.sh);
#              a failed cell stops the step
#   pairs-*    warm == cold pair cells (specpub_pairs.py) on each arm: deep (about 131k tokens)
#              and short (about 1k), greedy in the served shape and sampled (temperature 1.0,
#              seed 1234) with the published-entry digest trace
#   nsys       one Nsight Systems cell per arm at 131k (specpub_nsys.sh)
#   build-w    spark_build_arm.sh: arm W's server and test binary (W_DIR, arms/worker.w.rs), the
#              base's test binary with W's tests and the failed-copy cell's red arm
#   unit-w     unit_gpu_cells.sh on those, the failed-copy cell and its red arm included
#   bt131k-w   five more cells of the 131k campaign with arm W: W B W B W
#   bt260k-w   the 260k campaign, B against W: B W W B B W
#   nsys-w     one Nsight Systems cell at 131k for B and for W
#   build-u    spark_build_arm.sh: the third arm's server (U_DIR, from arms/worker.u.rs)
#   bt131k-u   seven more cells of the 131k campaign with the third arm: U B T U T B U
#   bt260k3    the 260k campaign with three arms: B T U U T B B U T
#   nsys3      one Nsight Systems cell per arm at 131k, three arms
#   hitgate    tools/spec-on-cache-hit-gate.sh qwen on each arm, on the gate's own small
#              artifact (Qwen3.5-9B NVFP4 with its MTP head)
# YIELD_ANY=1 (a unit another lane has first call on) hands the unit to any waiter that is not one
# of this lane's cells, not only to a landing waiter.
# U_DIR names the third arm's directory under bins/ (the steps that end in -u or 3). ARMS lists
# the arms of the pairs and hitgate steps (default "T B"); an arm other than B or T runs the
# binary under bins/$W_DIR.
# NOT_AFTER=HH:MM is the unit's handover time (unit_cell.sh): the driver stops at the first cell
# that cannot end before it.
# usage: [YIELD_ANY=1] [NOT_AFTER=HH:MM] [ARMS="..."] [W_DIR=name] [U_DIR=name] specpub_driver.sh UNIT BASE10 HEAD10 STEP...
set -u
UNIT=$1; B10=$2; H10=$3; B=base-$B10; T=head-$H10; shift 3
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-specpub/logs/driver-$UNIT.log
R='$HOME/.cache/memra-specpub'
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
M9='$HOME/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf'
DR27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf'
ENV27="MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_MTP_DRAFT=$DR27"
log() { echo "$(date -Is) $*" >> "$L"; }
cell() { # exit 3 from unit_cell.sh: the unit's window (NOT_AFTER) is closed, the driver stops
  bash "$H/unit_cell.sh" "$UNIT" "$1" "$2" "$3"; local rc=$?; log "$1 rc=$rc"
  [ $rc = 3 ] && { log "window closed at $1: driver stops"; exit 3; }
  return $rc
}
pair() { # pair NAME ARM_DIR SHAPE TURNS SAMPLING TTL
  cell "$1" "$6" "cd box && CELL_PORT=18193 PAIR_MODEL=$M27 timeout $(( $6 - 2 ))m python3 specpub_pairs.py $R/cells/$1/out \"$1=$R/bins/$2/memra-server|$3|$4|$ENV27 $5\""
}
UD=${U_DIR:-}
WD=${W_DIR:-}; W10=${WD#head-}
bt() { # bt SET TTL [CELL...]
  local set=$1 ttl=$2; shift 2
  [ $# -gt 0 ] || set -- 01-B 02-T 03-T 04-B 05-B 06-T
  for c in "$@"; do
    cell "bt$set-$c" "$ttl" "bash box/specpub_cell.sh bt$set $c $B $T $set ${UD:-none} ${WD:+W=$WD} && grep -q 'SPECPUB-CELL-DONE $c' runs/bt$set/campaign.log" || { log "bt$set: $c failed, step stopped"; return 1; }
  done
}
SAMPLED="_TEMP=1.0 _SEED=1234 _TRACE=1"
armdir() { case $1 in B) echo "$B" ;; T) echo "$T" ;; *) echo "${W_DIR:?W_DIR names arm $1}" ;; esac; }
pairs() { # pairs NAME SHAPE TURNS SAMPLING TTL: one case per arm of ARMS (default T B)
  for arm in ${ARMS:-T B}; do pair "pairs-$1-$arm" "$(armdir "$arm")" "$2" "$3" "$4" "$5"; done
}
for step in "$@"; do
  log "step $step"
  case $step in
    build) cell build 40 "bash box/spark_build.sh $B10 $H10; grep -q BUILD-DONE logs/build.log" ;;
    unit) cell unit 12 "bash box/unit_gpu_cells.sh $B $T" ;;
    bt131k) bt 131k 15 ;;
    bt260k) bt 260k 20 ;;
    pairs-deep-greedy) pairs deep-greedy deep4840 64,64,64 "" 24 ;;
    pairs-deep-sampled) pairs deep-sampled deep4840 64,64 "$SAMPLED" 18 ;;
    pairs-short-greedy) pairs short-greedy deep38 64,64,64 "" 14 ;;
    pairs-short-sampled) pairs short-sampled deep38 64,64,64 "$SAMPLED" 14 ;;
    build-w) # the final head's server and test binary, the base's and the red arm's test binaries
      cell build-w 30 "bash box/spark_build_arm.sh $WD worker.w.rs server test && bash box/spark_build_arm.sh basetests-$W10 worker.red.rs test && bash box/spark_build_arm.sh unwritten-$W10 worker.unwritten.rs test && cp arms/worker.w.rs src/crates/memra-server/src/worker.rs" ;;
    unit-w) cell unit-w 12 "bash box/unit_gpu_cells.sh basetests-$W10 $WD unwritten-$W10" ;;
    bt131k-w) bt 131k 15 14-W 15-B 16-W 17-B 18-W ;;
    bt260k-w) bt 260k 20 01-B 02-W 03-W 04-B 05-B 06-W ;;
    nsys-w) cell nsys131k-B 22 "bash box/specpub_nsys.sh nsys131k-B $B 131072"; cell nsys131k-W 22 "bash box/specpub_nsys.sh nsys131k-W $WD 131072" ;;
    build-u) cell build-u 15 "bash box/spark_build_arm.sh $UD worker.u.rs; grep -q BUILD-DONE logs/build-$UD.log" ;;
    bt131k-u) bt 131k 15 07-U 08-B 09-T 10-U 11-T 12-B 13-U ;;
    bt260k3) bt 260k 20 01-B 02-T 03-U 04-U 05-T 06-B 07-B 08-U 09-T ;;
    nsys3) cell nsys131k-B 22 "bash box/specpub_nsys.sh nsys131k-B $B 131072"; cell nsys131k-T 22 "bash box/specpub_nsys.sh nsys131k-T $T 131072"; cell nsys131k-U 22 "bash box/specpub_nsys.sh nsys131k-U $UD 131072" ;;
    nsys) cell nsys131k-B 22 "bash box/specpub_nsys.sh nsys131k-B $B 131072"; cell nsys131k-T 22 "bash box/specpub_nsys.sh nsys131k-T $T 131072" ;;
    hitgate)
      for arm in ${ARMS:-T B}; do
        d=$(armdir "$arm")
        cell "hitgate-$arm" 20 "cd src && MEMRA_GPU_LOCK=$R/cells/hitgate-$arm/gpu.lock timeout 18m bash tools/spec-on-cache-hit-gate.sh qwen $M9 $R/bins/$d/memra-server $R/cells/hitgate-$arm/ev"
      done ;;
    *) log "unknown step $step" ;;
  esac
done
log "driver done: $*"
