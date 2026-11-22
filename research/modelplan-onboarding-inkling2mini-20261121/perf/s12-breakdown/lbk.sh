#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
B=$P/a22-bin
O=$P/prof-bk; mkdir -p $O
for cls in nvfp4-rig shipped; do
  nsys profile -f true -o $O/$cls-c1 --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --gen 32 --out $O/$cls-c1.json > $O/$cls-c1.log 2>&1; echo "NSYS $cls c1=$?"
  nsys export --type sqlite -f true -o $O/$cls-c1.sqlite $O/$cls-c1.nsys-rep > /dev/null 2>&1
  nsys profile -f true -o $O/$cls-4k --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $cls --tape $T/p3_logscale.exact_f32 --prompt-len 4096 --context 5120 --gen 16 --out $O/$cls-4k.json > $O/$cls-4k.log 2>&1; echo "NSYS $cls 4k=$?"
  nsys export --type sqlite -f true -o $O/$cls-4k.sqlite $O/$cls-4k.nsys-rep > /dev/null 2>&1
done
