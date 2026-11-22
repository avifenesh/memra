#!/bin/bash
# verify checkpoint and verify serve for the nvfp4-rig class on the band twice its source spread.
set -uo pipefail
L=$HOME/.cache/inkling2mini
R=${RIG_BIN:-$L/serve-frozen7}
C=$HOME/models/inkling2-mini
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
V=$CP/verify-rig
mkdir -p $V
cp $R/SHA256SUMS $R/COMMIT $V/
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $V/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 3 $V/$name.log; return 0; }
export INK2_GPU=$R/ink2-gpu
step verify-checkpoint $R/memra model verify checkpoint $C --against inkling2 --out $V --oracle $CP/probe/lane-e-cpu-nvfp4-rig.tsv --native-runner $CP/ink2-gpu-rig.sh
step verify-serve env MEMRA_CTX=8192 MEMRA_INK2_CLASS=nvfp4-rig $R/memra model verify serve $C --against inkling2 --out $V --native-runner $R/memra-server
gzip -9 $V/native-oracle.tsv 2>/dev/null
# the serve route's functional checks with nvfp4-rig as the boot's class
MEMRA_INK2_CLASS=nvfp4-rig MEMRA_REWRITE_BUNDLE=$V MEMRA_MODELS=ink2=$C MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $R/memra-server > $V/server-rig.log 2>&1 &
SRV=$!
if waitfor --http http://127.0.0.1:18341/readyz --timeout 5m; then
  mkdir -p $V/serve-check
  SERVE_CHECK_CLASS=nvfp4-rig python3 research/modelplan-onboarding-inkling2mini-20261121/serve/serve_check.py http://127.0.0.1:18341 lane-h-local ink2 $V/serve-check --levels 1,2 --n 4 --max-tokens 64 > $V/serve-check.log 2>&1; echo "RC serve-check=$?"; grep -E 'FAIL|ALL PASS' $V/serve-check.log
else
  echo "SERVER NOT READY"; tail -n 5 $V/server-rig.log
fi
kill $SRV; waitfor --gone $SRV --timeout 2m
echo RUN_GPU_RIGVERIFY_DONE
