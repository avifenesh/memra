#!/bin/bash
# Fourth lease: the reference check's red arm on p1, then memra model verify checkpoint and verify serve.
set -uo pipefail
L=$HOME/.cache/inkling2mini
R=$L/target-serve/release
C=$HOME/models/inkling2-mini
T=$L/oracle
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
V=$CP/verify
mkdir -p $V
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $CP/logs/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 3 $CP/logs/$name.log; return 0; }
step p1-moe-ref-red $R/ink2-gpu $C --class shipped --tape $T/tapes/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --red moe-ref-index-order --out $CP/p1-moe-ref-red.json
step verify-checkpoint $R/memra model verify checkpoint $C --against inkling2 --out $V --oracle $CP/probe/lane-e-cpu-actq.tsv --native-runner $R/ink2-gpu
step verify-serve env MEMRA_CTX=8192 $R/memra model verify serve $C --against inkling2 --out $V --native-runner $R/memra-server
echo RUN_GPU_D_DONE
