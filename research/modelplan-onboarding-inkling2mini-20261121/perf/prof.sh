#!/bin/bash
# usage: prof.sh NAME BIN [CLASS] [extra runner args]: nsys of 8 decode tokens from p1
N=$1; BIN=$2; CLS=${3:-declared}; shift 3
O=/home/avifenesh/.cache/inkling2mini/perf/prof-$N
mkdir -p $O
bash /home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; echo CLAMP_RC=$?
nsys profile -f true -o $O/run --trace=cuda --sample=none --cpuctxsw=none $BIN/ink2-gpu /home/avifenesh/models/inkling2-mini --class $CLS --tape /home/avifenesh/.cache/inkling2mini/oracle/tapes/p1_chat.exact_f32 --gen 8 --out $O/run.json "$@" > $O/run.log 2>&1; echo RC=$?
nsys stats -f csv -r cuda_gpu_kern_sum,cuda_gpu_trace -o $O/run $O/run.nsys-rep > /dev/null 2>&1; echo STATS=$?
ls $O
