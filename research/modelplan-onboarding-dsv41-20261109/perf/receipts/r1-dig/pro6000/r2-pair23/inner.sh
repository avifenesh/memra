#!/bin/bash
set -u
export MEMRA_FULL_PREC=1 MEMRA_DSV41_TP_ISSUE=parallel
DL=$(( $(date +%s) + 600 - 15 ))
drops() { journalctl -k --since '-3 hours' --no-pager 2>/dev/null | grep drop_caches | awk '{print $3}' | tail -2 | tr '\n' ' '; }
step() {  # NAME EST CMD...
  local name=$1 est=$2; shift 2
  if [ $(( $(date +%s) + est )) -gt $DL ]; then echo "SKIP $name (budget)" >> /home/access/dsv41/receipts-perf/r2-pair23/runs.txt; return 1; fi
  echo "$name start $(date +%H:%M:%S) drops $(drops)" >> /home/access/dsv41/receipts-perf/r2-pair23/runs.txt
  bash /home/access/dsv41/bin/perf_slow.sh run /home/access/dsv41/receipts-perf/r2-pair23/slowdown.txt $name "$@"
  echo "$name rc $? end $(date +%H:%M:%S) drops $(drops)" >> /home/access/dsv41/receipts-perf/r2-pair23/runs.txt
}
gate() {  # NAME TRANSPORT
  if [ $2 = default ]; then unset MEMRA_TP_TRANSPORT; else export MEMRA_TP_TRANSPORT=$2; fi
  step gate-$1 75 /home/access/dsv41/perf-bins/r2/dsv41-gate tp /home/access/dsv41/DeepSeek-V4.1-Flash-1017 /home/access/dsv41/captures-v2/needle --layers 1,2,14,20,21,24 --devices 0,1     --decode 2 --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32 --out /home/access/dsv41/receipts-perf/r2-pair23/gate-$1.json > /home/access/dsv41/receipts-perf/r2-pair23/gate-$1.log 2>&1
  echo "gate-$1: $(grep -o '[0-9]* checks, [0-9]* failed' /home/access/dsv41/receipts-perf/r2-pair23/gate-$1.log | tail -1) $(grep -ac '^FAIL' /home/access/dsv41/receipts-perf/r2-pair23/gate-$1.log) FAIL lines" >> /home/access/dsv41/receipts-perf/r2-pair23/runs.txt
  grep -a "stress/" /home/access/dsv41/receipts-perf/r2-pair23/gate-$1.log | cut -c1-200 >> /home/access/dsv41/receipts-perf/r2-pair23/runs.txt
}
if [ 1 != 0 ]; then
  gate default default
  gate peer-pull peer-pull
fi
export MEMRA_TP_TRANSPORT=device-push
i=0
for a in A:kpc B:r2 B:r2 A:kpc B:r2 A:kpc A:kpc B:r2 ; do
  i=$((i+1)); arm=${a%%:*}; b=${a#*:}
  step run$i-$arm 70 /home/access/dsv41/perf-bins/$b/dsv41-tpbench /home/access/dsv41/DeepSeek-V4.1-Flash-1017 --token-map /home/access/dsv41/oracle/engram_constants.token_map.i32 --devices 0,1 --out /home/access/dsv41/receipts-perf/r2-pair23/runs/run$i-$arm     --layers 20 --resident 20 --prefill /home/access/dsv41/receipts-perf/r2-pair23/prompts/rows4.i32,/home/access/dsv41/receipts-perf/r2-pair23/prompts/rows8.i32     --decode /home/access/dsv41/dec/ids/capital.prompt.i32 --decode-steps 24 --rounds 12 > /home/access/dsv41/receipts-perf/r2-pair23/runs/run$i-$arm.log 2>&1
done
