#!/bin/bash
# combined lease: nsys of step 1, then the step 2 checks against step 1
O=/home/avifenesh/.cache/inkling2mini/perf/prof-s1
P=/home/avifenesh/.cache/inkling2mini/perf
mkdir -p $O
bash /home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; echo CLAMP_RC=$?
nsys profile -f true -o $O/run --trace=cuda --sample=none --cpuctxsw=none $P/s1-bin/ink2-gpu /home/avifenesh/models/inkling2-mini --class declared --tape /home/avifenesh/.cache/inkling2mini/oracle/tapes/p1_chat.exact_f32 --gen 8 --out $O/run.json > $O/run.log 2>&1; echo RC=$?
nsys stats -f csv -r cuda_gpu_kern_sum -o $O/run $O/run.nsys-rep > /dev/null 2>&1; echo STATS=$?
bash $P/step.sh s2 $P/s2-bin $P/s1-bin > $P/s2.log 2>&1; echo STEP2_RC=$?
