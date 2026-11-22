#!/bin/bash
set -uo pipefail
O=$HOME/.cache/inkling2mini/oracle
C=$HOME/.cache/inkling2mini/ckpt
cd "$O"
PY="venv/bin/python -I"
W3="0:16,8128:0"
mkdir -p tapes-var logs
cap() { name=$1; shift; nice -n 10 $PY ink2_oracle.py capture --ckpt "$C" --work work/nonexpert --prompt prompts/p3_logscale.json --greedy 1 --threads 12 --resid-keep $W3 --logits-keep $W3 --out tapes-var/$name "$@" > logs/var_$name.log 2>&1 && echo OK $name || { echo CAPTURE_FAILED $name; tail -n 20 logs/var_$name.log; }; }
( cap p3.exact_f32.chunk2048 --arm exact --chunk 2048 ) &
( cap p3.exact_f64.chunk2800 --arm exact --dtype f64 --chunk 2800 ) &
wait
echo P3VAR_DONE
