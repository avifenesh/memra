#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
bash $P/idab.sh lk10 $P/a39-bin $P/a36-bin ab
bash $P/kprof.sh lk10-prof $P/a39-bin
python3 $P/bygrid.py $P/lk10-prof/nvfp4-rig-c1.sqlite gemv4
python3 $P/slotseq.py $P/lk10-prof/nvfp4-rig-c1.sqlite 20 10 | tail -22
