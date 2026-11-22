#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
bash $P/idab.sh lk7 $P/a36-bin $P/a33-bin
bash $P/kprof.sh lk7-prof $P/a36-bin
python3 $P/slotseq.py $P/lk7-prof/nvfp4-rig-c1.sqlite 20 10
