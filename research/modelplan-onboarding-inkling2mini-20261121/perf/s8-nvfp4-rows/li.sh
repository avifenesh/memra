#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
waitfor --file $P/a2-bin/SHA256SUMS --timeout 20m; echo WAIT=$?
$P/a2-bin/ink2-gpu-gate > $P/a2-gate.txt 2>&1; echo GATE_RC=$?; tail -1 $P/a2-gate.txt; grep -E 'FAIL|prec-adopted' $P/a2-gate.txt
bash $P/cls2.sh gi-rig nvfp4-rig $P/a1-bin "" $P/a2-bin "" > $P/gi-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh gi-shipped shipped $P/a1-bin "" $P/a2-bin "" 0 notime > $P/gi-shipped.log 2>&1; echo RC_shipped=$?
O=$P/prof-4k; mkdir -p $O
for cls in nvfp4-rig shipped; do
  nsys profile -f true -o $O/$cls --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $P/a2-bin/ink2-gpu $M --class $cls --tape $T/p3_logscale.exact_f32 --prompt-len 4096 --context 5120 --gen 8 --out $O/$cls.json > $O/$cls.log 2>&1; echo NSYS_$cls=$?
  nsys stats -f csv -r cuda_gpu_kern_sum -o $O/$cls $O/$cls.nsys-rep > /dev/null 2>&1
done
