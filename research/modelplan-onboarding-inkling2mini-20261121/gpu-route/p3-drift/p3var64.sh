#!/bin/bash
set -uo pipefail
O=$HOME/.cache/inkling2mini/oracle
C=$HOME/.cache/inkling2mini/ckpt
cd "$O"
PY="venv/bin/python -I"
W3="0:16,8128:0"
nice -n 10 $PY ink2_oracle.py capture --ckpt "$C" --work work/nonexpert --prompt prompts/p3_logscale.json --greedy 1 --threads 16 --resid-keep $W3 --logits-keep $W3 --arm exact --dtype f64 --chunk 1024 --out tapes-var/p3.exact_f64.chunk1024 > logs/var_p3.exact_f64.chunk1024.log 2>&1 && echo OK || { echo CAPTURE_FAILED; tail -n 20 logs/var_p3.exact_f64.chunk1024.log; }
echo P3VAR64_DONE
