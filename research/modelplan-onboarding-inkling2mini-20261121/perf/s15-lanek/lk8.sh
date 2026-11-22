#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
bash $P/idab.sh lk8 $P/a37-bin $P/a36-bin ab
bash $P/kprof.sh lk8-prof $P/a37-bin
python3 $P/slotseq.py $P/lk8-prof/nvfp4-rig-c1.sqlite 20 10
python3 $P/bygrid.py $P/lk8-prof/nvfp4-rig-c1.sqlite gemv4
