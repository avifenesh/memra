#!/usr/bin/env bash
# Rig: lever 1's warm == cold pair cells on a GB10 unit (research/spec-pool-grid-20261109's
# spec_pool_pair_cells.py with the unit's model path and the served route's masked MTP draft), a
# subset of six cases, one case (one warm boot, two cold boots) per sparklock lease. Untimed
# byte-equality cells. usage: pairs_gb10_driver.sh UNIT HEAD10
set -u
UNIT=$1; H=$2
D=$HOME/.cache/memra-gb10-r4
L=$D/pairs-gb10-driver-$UNIT.log
R='$HOME/.cache/memra-gb10-r4'
BIN="$R/bins/eost-$H/memra-server"
DR='MEMRA_MTP_DRAFT=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf MEMRA_PREFIX_STABLE_BOUNDARY=1'
CASES=(
  "chat-short-open|chat-short|open,open,open|_EXPECT=grid"
  "chat-short-open-t1|chat-short|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=grid"
  "chat-exact-open|chat-exact|open,open,open|_EXPECT=grid"
  "chat-exact-open-t1|chat-exact|open,open,open|_TEMP=1.0 _SEED=1234 _EXPECT=grid"
  "raw-long-nopc|raw-long|open,open,open|MEMRA_PREFIX_CACHE_MB=0 _USER=pair-1 _EXPECT=grid"
  "raw-long-nopc-t1|raw-long|open,open,open|MEMRA_PREFIX_CACHE_MB=0 _USER=pair-1 _TEMP=1.0 _SEED=1234 _EXPECT=grid"
)
for c in "${CASES[@]}"; do
  n=${c%%|*}; rest=${c#*|}
  id=$(sparklock acquire --unit "$UNIT" --ttl 14m --for "memra GB10 round 5 lever 1: warm == cold pair cell $n (untimed, one case per lease)" --wait 30m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $n: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $n: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "cd $R/pairs-gb10 && CELL_PORT=18193 timeout 13m python3 spec_pool_pair_cells.py $R/pairs-gb10/out \"$n=$BIN|$rest $DR\"" >> "$D/pairs-gb10-$UNIT.log" 2>&1
  echo "$(date -Is) $n: exit $?" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
done
echo "$(date -Is) all cases done" >> "$L"
