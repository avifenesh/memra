#!/bin/bash
# The qualification pin: verify checkpoint and verify serve with the binaries of b948b9f57a
# (serve-frozen4: lane P's decode merged and the attention scratch fix).
set -uo pipefail
L=$HOME/.cache/inkling2mini
R=$L/serve-frozen4
C=$HOME/models/inkling2-mini
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
V=$CP/verify-b948b9f57a
mkdir -p $V
cp $R/SHA256SUMS $R/COMMIT $V/
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $V/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 3 $V/$name.log; return 0; }
step verify-checkpoint $R/memra model verify checkpoint $C --against inkling2 --out $V --oracle $CP/probe/lane-e-cpu-actq.tsv --native-runner $R/ink2-gpu
step verify-serve env MEMRA_CTX=8192 $R/memra model verify serve $C --against inkling2 --out $V --native-runner $R/memra-server
gzip -9 $V/native-oracle.tsv
echo RUN_GPU_E_DONE
