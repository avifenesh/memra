#!/bin/bash
# Laptop driver, 2026-11-13: after the four-card block ends (WIN4-HELD-DONE), sync + build the CUTLASS decode fix
# (a4de892880) on wrx, then pause ext35 on cards 2+3, run the two-card confirm (k3 vs cfix, gates first), resume ext35.
set -u
SHA=a4de892880
D=/home/access/dsv41
O=/home/avifenesh/.cache/dsv41-memra/win1113
log() { echo "$(date +%T) $*"; }
log "waiting for the block end"
waitfor --timeout 120m --every 30s --ssh wrx "grep -q WIN4-HELD-DONE $D/spec/logs/win4-arm.log" > /dev/null 2>&1 || { log "no block end"; exit 1; }
log "block done; sync $SHA"
bash /home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix/wrx/sync_cfix.sh $SHA > $O/cfix-sync.log 2>&1 || { log "sync failed"; tail -n 20 $O/cfix-sync.log; exit 1; }
tail -n 3 $O/cfix-sync.log
ssh wrx "(bash $D/bin/cfix_build.sh $SHA) > $D/cfix-build.out 2>&1 < /dev/null & echo started"
waitfor --timeout 45m --every 30s --ssh wrx "grep -q CFIX-BUILD-DONE $D/cfix-build.out" > /dev/null 2>&1 || { log "no build end in 45 min"; exit 1; }
b=$(ssh wrx "grep -h CFIX-BUILD-DONE $D/cfix-build.out")
log "$b"
case "$b" in *"rc 0 "*) ;; *) ssh wrx "tail -n 30 $D/cfix-build.out"; exit 1 ;; esac
ssh wrx "bash $D/spec/scripts/ext35_23.sh stop"
ssh wrx "(cd $D && K3_CARDS=23 K3_DEV=0,1 K3_LAYERS=20 K3_ARM_A=k3 K3_ARM_B=cfix K3_TAG=cfix-two K3_KGATE=1 K3_GATE_T='default peer-pull host-canonical' K3_ORDER='A B B A' bash bin/perf_cfix_four.sh 720) > $D/cfix-two.out 2>&1 < /dev/null & echo started"
waitfor --timeout 25m --every 30s --ssh wrx "grep -q 'CFIX-DONE cfix-two' $D/cfix-two.out" > /dev/null 2>&1 || log "no CFIX-DONE in 25 min"
ssh wrx "tail -n 60 $D/cfix-two.out" > $O/cfix-two.tail
ssh wrx "bash $D/spec/scripts/ext35_23.sh go"
log "CFIX-CHAIN-END"
