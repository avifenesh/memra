#!/usr/bin/env bash
# The admit-w-release cell's halves in order, started by run.sh in its own process group (setsid) so the guard's
# deadline and thermal stop reach every process of the cell and nothing else.
# usage: cell.sh <part>...   part 1 = DAY45 2.2 (chain tree 0ac7f6413, binary S45), part 2 = DAY46 2.2 (chain tree
# 43dbdb29f, binary S46). env (from run.sh): ROOT, OUT, CARD, MODEL, CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES,
# RIG_LOCK_HELD=1, CELL_LOCK=none, IDLE_WAIT_S, READER_CARD, DRY_RUN.
set -uo pipefail
ROOT=${ROOT:?}; OUT=${OUT:?}
rc_all=0
for p in "$@"; do
  case $p in
    1) R=$OUT/day45 WT=$ROOT/d45 BIN=$ROOT/bins/s45/memra-server \
         bash "$ROOT/d45/research/spill-b-20260919/pro-single-day45/chain-b.sh"; rc=$? ;;
    2) R=$OUT/day46 WT=$ROOT/d46 BIN=$ROOT/bins/s46/memra-server \
         bash "$ROOT/d46/research/spill-b-20260919/pro-single-day46c/chain.sh"; rc=$? ;;
    *) echo "unknown part $p"; rc=2 ;;
  esac
  echo "$(date -u +%FT%TZ) part $p rc=$rc"
  [ "$rc" = 0 ] || rc_all=$rc
done
exit "$rc_all"
