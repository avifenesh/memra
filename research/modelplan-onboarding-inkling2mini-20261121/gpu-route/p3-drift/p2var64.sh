#!/bin/bash
set -uo pipefail
O=$HOME/.cache/inkling2mini/oracle
C=$HOME/.cache/inkling2mini/ckpt
cd "$O"
PY="venv/bin/python -I"
K="0:64,480:544,1024:1088,-96:0"
nice -n 10 $PY ink2_oracle.py capture --ckpt "$C" --work work/nonexpert --prompt prompts/p2_window.json --greedy 1 --threads 16 --resid-keep $K --logits-keep $K --arm exact --dtype f64 --chunk 512 --out tapes-var/p2.exact_f64.chunk512 > logs/var_p2.exact_f64.chunk512.log 2>&1 && echo OK || { echo CAPTURE_FAILED; tail -n 20 logs/var_p2.exact_f64.chunk512.log; }
echo P2VAR64_DONE
