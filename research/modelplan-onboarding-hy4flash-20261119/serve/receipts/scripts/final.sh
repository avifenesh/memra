#!/usr/bin/env bash
# GB10 box: the record's chain at one commit: build, verify checkpoint + serve, route checks, the
# route's greedy ids, the serve gates (default chunk, with readings; chunk 500), each GPU job alone
set -u
D=$HOME/hy4flash/pr9
L=$HOME/hy4flash/logs/pr9-final.log
S=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
echo "FINAL START $(date '+%F %T') $(cat $D/src/HEAD.txt 2>/dev/null)" >> $L
bash $D/build.sh > $HOME/hy4flash/logs/pr9-build.log 2>&1
grep BUILD_EXIT $HOME/hy4flash/logs/pr9-build.log >> $L
grep -q BUILD_EXIT=0 $HOME/hy4flash/logs/pr9-build.log || exit 1
bash $D/verify.sh final; tail -n 9 $HOME/hy4flash/logs/pr9-verify-final.log >> $L
bash $D/route-check.sh; grep -E 'SHA' $HOME/hy4flash/logs/pr9-route-check.log >> $L
bash $D/gen-chain.sh; grep -E 'GEN |ARGMAX' $HOME/hy4flash/logs/pr9-gen.log | tail -n 3 >> $L
bash $S/serve_gate.sh final both; echo "SERVE-GATE final rc=$?" >> $L
bash $S/serve_gate.sh chunk500 gate MEMRA_HY4F_PREFILL_CHUNK=500; echo "SERVE-GATE chunk500 rc=$?" >> $L
echo "FINAL_DONE $(date '+%F %T')" >> $L
