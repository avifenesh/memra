#!/usr/bin/env bash
# GB10 debug lease: the peer-pull rows-16 failures of the tp gate. 2 ranks on peer-pull, layers 2 and 24: c1 x3,
# c3 = c1 + debug arms: base, V41DBG_LOCK (host calls serialized), V41DBG_SYNC (stream sync after each CUTLASS
# launch), V41DBG_SFZERO (zero-filled activation scale buffers), three rounds interleaved.
# Done-file out/dbg-pp/runs.txt, marker DBG-PP3-DONE.
set -u
R=$HOME/.cache/dsv41-cutlass
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1 MEMRA_TP_TRANSPORT=peer-pull
O=$R/out/dbg-pp3; mkdir -p $O; : > $O/runs.txt
run() { tag=$1; bin=$2; shift 2
  timeout 6m env "$@" $bin/dsv41-gate tp $C $D/captures-v2/needle --layers 2,24 --devices 0,0 --decode 2 --token-map $TM \
    --out $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag rc $? $(grep -o '[0-9]* checks, [0-9]* failed' $O/$tag.log | tail -1) $(date +%T)" >> $O/runs.txt
  grep -a '^FAIL' $O/$tag.log | head -6 >> $O/runs.txt; }
run c3base-0 $R/bins/c3 X=1
run c3lock-0 $R/bins/c3 V41DBG_LOCK=1
run c3sync-0 $R/bins/c3 V41DBG_SYNC=1
run c3sfzero-0 $R/bins/c3 V41DBG_SFZERO=1
run c3base-1 $R/bins/c3 X=1
run c3lock-1 $R/bins/c3 V41DBG_LOCK=1
run c3sync-1 $R/bins/c3 V41DBG_SYNC=1
run c3sfzero-1 $R/bins/c3 V41DBG_SFZERO=1
run c3base-2 $R/bins/c3 X=1
run c3lock-2 $R/bins/c3 V41DBG_LOCK=1
run c3sync-2 $R/bins/c3 V41DBG_SYNC=1
run c3sfzero-2 $R/bins/c3 V41DBG_SFZERO=1
echo "DBG-PP3-DONE $(date +%T)" >> $O/runs.txt
