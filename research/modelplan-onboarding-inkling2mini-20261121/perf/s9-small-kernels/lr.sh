#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini; V=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
$P/a9-bin/ink2-gpu-gate > $P/a9-gate.txt 2>&1; echo GATE_RC=$?; tail -1 $P/a9-gate.txt; grep -E 'FAIL|f32t' $P/a9-gate.txt | cut -c1-400
mkdir -p $P/gr-id
for arm in base new; do
  B=$P/a6-bin; [ $arm = new ] && B=$P/a9-bin
  $B/ink2-gpu $M --class nvfp4-rig --tape $V/p3.exact_f64.chunk1024.as_f32 --chunk 1024 --context 8448 --score --out $P/gr-id/id-rig-p3-$arm.json > $P/gr-id/id-rig-p3-$arm.log 2>&1; echo "RC p3 $arm=$?"
done
bash $P/cls2.sh gr-rig4k nvfp4-rig $P/a6-bin "" $P/a9-bin "" 4096 > $P/gr-rig4k.log 2>&1; echo RC_rig4k=$?
O=$P/prof-32k; mkdir -p $O
nsys profile -f true -o $O/rig --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $P/a9-bin/ink2-gpu $M --class nvfp4-rig --tape $T/p3_logscale.exact_f32 --prompt-len 32768 --context 33792 --gen 6 --out $O/rig.json > $O/rig.log 2>&1; echo NSYS=$?
