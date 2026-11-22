#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
bash $P/idab.sh lk4 $P/a32-bin $P/a31-bin ab
bash $P/kprof.sh lk4-prof $P/a32-bin
