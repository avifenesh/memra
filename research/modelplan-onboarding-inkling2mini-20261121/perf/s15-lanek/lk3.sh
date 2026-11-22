#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
$P/a31-bin/ink2-actq-gate > $P/lk3-actq-gate.txt 2>&1; echo ACTQ_GATE_RC=$?; tail -2 $P/lk3-actq-gate.txt; grep -i fail $P/lk3-actq-gate.txt | head -5
bash $P/idab.sh lk3 $P/a31-bin $P/a30-bin ab
bash $P/kprof.sh lk3-prof $P/a31-bin
