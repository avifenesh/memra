#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
bash $P/idab.sh lk2 $P/a30-bin $P/a28-bin ab
bash $P/kprof.sh lk2-prof $P/a30-bin
