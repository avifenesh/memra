#!/bin/bash
# The single-card residency (routed experts in pinned host memory behind a device LRU cache) against the decode
# lane's receipts, byte for byte: (1) the 13 chat tapes teacher-forced with the cache filling the card, against the
# committed-binary run dec/native-final; (2) five chat tapes greedy with the cache at its floor (one layer's experts,
# constant eviction), against dec/greedy. one card only, one lease per run.
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
D=$ROOT
S=$D/serve
B=$D/target-serve/release/dsv41-run
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
UUID=$GPU_UUID
mkdir -p $S/logs
lease() {
  local name=$1 to=$2; shift 2
  if [ -e $QUIET_FILE ]; then echo "QUIET is set: not starting $name"; exit 3; fi
  if [ -e $ROOT/serve/STOP ]; then echo "STOP is set: not starting $name"; exit 4; fi
  local R=$D/leases/serve-$name-$(date +%Y%m%dT%H%M%S); mkdir -p "$R"
  $GPU_RUN --gpus $UUID --receipt "$R" --box-lock $BOX_LOCK --timeout $to --wait 1800 -- \
    taskset -c $CORES env MEMRA_FULL_PREC=1 $B $C --token-map $TM "$@" > $S/logs/$name.log 2>&1
  echo "$name rc $? $(grep -h 'dsv41-run: [0-9]* forwards' $S/logs/$name.log | tail -n 1)"
}
sha256sum $B
grep -h 'MemAvailable' /proc/meminfo
rm -rf $S/res-forced $S/res-greedy-floor
mkdir -p $S/res-forced $S/res-greedy-floor
lease res-forced 2400 --cases $D/dec/chat-forced.txt --out $S/res-forced --experts-host
n=0; bad=0
for c in capital gsm code copy hebrew gsm_think story hebrew_para code_pal sys_json multiturn needle needle1200; do
  for f in top_ids.i32 top_vals.f32 lse.f64 argmax.i32 full.f32; do
    n=$((n + 1)); cmp -s $D/dec/native-final/$c/$f $S/res-forced/$c/$f || { bad=$((bad + 1)); echo "DIFFERS $c $f"; }
  done
done
echo "res-forced: $n files compared with dec/native-final, $bad differ"
printf '%s\n' capital gsm hebrew sys_json multiturn | while read -r c; do echo "$c $D/dec/ids/$c.prompt.i32"; done > $S/greedy5.txt
lease res-greedy-floor 2400 --cases $S/greedy5.txt --gen 200 --out $S/res-greedy-floor --experts-host --cache-slots 384
n=0; bad=0
for c in capital gsm hebrew sys_json multiturn; do
  for f in gen.i32 top_ids.i32 top_vals.f32 lse.f64 argmax.i32 full.f32; do
    n=$((n + 1)); cmp -s $D/dec/greedy/$c/$f $S/res-greedy-floor/$c/$f || { bad=$((bad + 1)); echo "DIFFERS $c $f"; }
  done
done
echo "res-greedy-floor: $n files compared with dec/greedy, $bad differ"
echo RES-GATE-DONE
