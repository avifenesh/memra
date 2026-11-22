#!/usr/bin/env bash
# GB10 box, batch lane: the baseline readings on land/31 (bins-base): drafts off and the draft
# gate, conc_bench at c=1,2,4,8 (long + chat), then nsys of a c=4 decode, drafts off and gate.
set -u
D=$HOME/hy4flash/batch
cd $D
B=$D/bins-base/memra-server
echo "CHAIN-BASE START $(date '+%F %T')"
bash ./serve.sh base-off $B MEMRA_HY4F_DRAFT=0; echo "base-off rc=$? $(date '+%T')"
bash ./serve.sh base-gate $B; echo "base-gate rc=$? $(date '+%T')"
bash ./prof-serve.sh base-off-c4 $B 4 10 MEMRA_HY4F_DRAFT=0; echo "prof off rc=$? $(date '+%T')"
bash ./prof-serve.sh base-gate-c4 $B 4 10; echo "prof gate rc=$? $(date '+%T')"
echo "CHAIN-BASE-DONE $(date '+%F %T')"
