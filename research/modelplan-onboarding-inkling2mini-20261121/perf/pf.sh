#!/bin/bash
# usage: pf.sh NAME BIN0 BIN1 [BIN2]: prefill of 512, 2048, 8192 tokens (chunk 512), arms in rotating order, 3 takes each
N=$1; shift; BINS=("$@"); [ "$N" = pf6 ] && BINS+=(/home/avifenesh/.cache/inkling2mini/perf/s7-bin)
O=/home/avifenesh/.cache/inkling2mini/perf/$N; mkdir -p $O
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
i=0; for b in "${BINS[@]}"; do echo "arm$i $b"; cat $b/SHA256SUMS; i=$((i+1)); done > $O/binaries.sha256
bash /home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; echo CLAMPCHECK_RC=$?
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > $O/smi.csv 2>&1 &
SMI=$!
n=${#BINS[@]}
for PL in 512 2048 8192; do
  CTX=$(( (PL + 64 + 1023) / 1024 * 1024 ))
  for i in 1 2 3; do
    order=$(seq 0 $((n-1))); [ $((i % 2)) -eq 0 ] && order=$(seq $((n-1)) -1 0)
    for k in $order; do
      B=${BINS[$k]}
      echo "MARK pf$PL-arm$k-$i start $(date +%T.%N)"
      $B/ink2-gpu $M --class declared --tape $T/p3_logscale.exact_f32 --prompt-len $PL --context $CTX --gen 2 --out $O/pf$PL-arm$k-$i.json > $O/pf$PL-arm$k-$i.log 2>&1; echo "RC $PL arm$k $i=$?"
      echo "MARK pf$PL-arm$k-$i end $(date +%T.%N)"
    done
  done
done
kill $SMI
