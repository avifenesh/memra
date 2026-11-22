#!/bin/bash
# usage: kprof.sh NAME BIN [CLASS]  (nsys graph-node profile of 32 decode tokens, per-kernel breakdown)
P=/home/avifenesh/.cache/inkling2mini/perf; T=/home/avifenesh/.cache/inkling2mini/oracle/tapes; M=/home/avifenesh/models/inkling2-mini
N=$1; B=$2; CLS=${3:-nvfp4-rig}; O=$P/$N; mkdir -p $O
nsys profile -f true -o $O/$CLS-c1 --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none $B/ink2-gpu $M --class $CLS --tape $T/p1_chat.exact_f32 --gen 32 --out $O/$CLS-c1.json > $O/$CLS-c1.log 2>&1; echo NSYS=$?
nsys export --type sqlite -f true -o $O/$CLS-c1.sqlite $O/$CLS-c1.nsys-rep > /dev/null 2>&1
PB=$O/ python3 $P/bk.py $CLS-c1 14 > $O/bk-$CLS-c1.txt; cat $O/bk-$CLS-c1.txt
