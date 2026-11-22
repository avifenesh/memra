#!/bin/bash
# land/inkling2mini-1: verify checkpoint and verify serve for both served classes, and the served
# c=1/2/4 of both classes under clampcheck, on the landing's own binaries (land-frozen3). Run from the
# serve worktree (receipts land in its research tree).
L=/home/avifenesh/.cache/inkling2mini
B=$L/land-frozen3
M=/home/avifenesh/models/inkling2-mini
C=$L/ceiling/clampcheck.sh
R=/home/avifenesh/.cache/inkling2mini/wt-serve/research/modelplan-onboarding-inkling2mini-20261121
CP=$R/checkpoint-parity
O=$R/serve/land-verify
mkdir -p $O; cd $O
cat $B/COMMIT $B/SHA256SUMS
bash $C; echo "CLAMPCHECK_RC=$?"
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > smi.csv 2>&1 &
SMI=$!
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; tail -n 2 $tag.log; }
mkdir -p shipped-verify nvfp4-rig-verify
cp $B/SHA256SUMS $B/COMMIT shipped-verify/; cp $B/SHA256SUMS $B/COMMIT nvfp4-rig-verify/
run verify-checkpoint-shipped $B/memra model verify checkpoint $M --against inkling2 --out $O/shipped-verify --oracle $CP/probe/lane-e-cpu-actq.tsv --native-runner $B/ink2-gpu
run verify-serve-shipped env MEMRA_CTX=8192 MEMRA_INK2_CLASS=shipped $B/memra model verify serve $M --against inkling2 --out $O/shipped-verify --native-runner $B/memra-server
export INK2_GPU=$B/ink2-gpu
run verify-checkpoint-rig $B/memra model verify checkpoint $M --against inkling2 --out $O/nvfp4-rig-verify --oracle $CP/probe/lane-e-cpu-nvfp4-rig.tsv --native-runner $CP/ink2-gpu-rig.sh
run verify-serve-rig env MEMRA_CTX=8192 MEMRA_INK2_CLASS=nvfp4-rig $B/memra model verify serve $M --against inkling2 --out $O/nvfp4-rig-verify --native-runner $B/memra-server
gzip -9 $O/*-verify/native-oracle.tsv 2>/dev/null
for cls in shipped nvfp4-rig; do
  MEMRA_INK2_CLASS=$cls MEMRA_REWRITE_BUNDLE=$O/shipped-verify MEMRA_MODELS=ink2=$M MEMRA_ADDR=127.0.0.1:18341 MEMRA_API_KEY=lane-h-local MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 $B/memra-server > server-$cls.log 2>&1 &
  SRV=$!
  if waitfor --http http://127.0.0.1:18341/readyz --timeout 5m; then
    mkdir -p $O/$cls
    SERVE_CHECK_CLASS=$cls python3 $R/serve/serve_check.py http://127.0.0.1:18341 lane-h-local ink2 $O/$cls --levels 1,2,4 --n 8 --max-tokens 128 > serve-check-$cls.log 2>&1; echo "RC serve-check-$cls=$?"
    grep -E 'FAIL|ALL PASS' serve-check-$cls.log
  else
    echo "SERVER NOT READY ($cls)"; tail -n 5 server-$cls.log
  fi
  kill $SRV; waitfor --gone $SRV --timeout 2m
done
kill $SMI
echo LAND_VERIFY_DONE
