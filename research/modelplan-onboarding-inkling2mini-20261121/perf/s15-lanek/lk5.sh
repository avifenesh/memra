#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
$P/a33-bin/ink2-actq-gate > $P/lk5-actq-gate.txt 2>&1; echo ACTQ_GATE_RC=$?; tail -1 $P/lk5-actq-gate.txt
bash $P/idab.sh lk5 $P/a33-bin $P/a32-bin ab
bash $P/kprof.sh lk5-prof $P/a33-bin
