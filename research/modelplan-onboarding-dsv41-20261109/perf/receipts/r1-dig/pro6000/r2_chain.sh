#!/bin/bash
# Laptop driver, 2026-11-13: after the CUTLASS chain ends (CFIX-CHAIN-END), build r2 (7c964f8205) on wrx (no build
# overlaps a timed cell), then its byte gate vs kpc on card 2, then kpc vs r2 step timing on cards 2+3 (20 layers).
set -u
D=/home/access/dsv41
O=/home/avifenesh/.cache/dsv41-memra/win1113
log() { echo "$(date +%T) $*"; }
log "waiting for CFIX-CHAIN-END"
waitfor --timeout 150m --every 30s --grep 'CFIX-CHAIN-END' "$O/cfix_chain.out" > /dev/null 2>&1 || { log "no CFIX-CHAIN-END"; exit 1; }
log "cfix chain done; ext35 keeps cards 2+3 busy during the build"
ssh wrx "bash $D/spec/scripts/ext35_23.sh go"
ssh wrx "(cd $D && nohup bash bin/wrx_build.sh r2-build bash bin/build_one.sh r2) > $D/receipts-perf-build-r2.out 2>&1 < /dev/null & echo started"
waitfor --timeout 30m --every 30s --ssh wrx "grep -q 'BUILD-DONE r2' $D/receipts-perf-build-r2.out" > /dev/null 2>&1 || { log "no r2 build end"; exit 1; }
b=$(ssh wrx "grep -h 'BUILD-DONE r2' $D/receipts-perf-build-r2.out")
log "$b"
case "$b" in *"rc 0"*) ;; *) ssh wrx "tail -n 30 $D/receipts-perf-build-r2.out"; exit 1 ;; esac
ssh wrx "bash $D/spec/scripts/ext35_23.sh stop"
log "2a byte gate vs kpc on card 2"
ssh wrx "cd $D && bash bin/perf_runc.sh r2-bytes/lease 600 2 bash bin/r2_bytes.sh > receipts-perf-r2-bytes.out 2>&1; cat receipts-perf/r2-bytes/bytes.txt" > $O/r2-bytes.txt 2>&1
cat $O/r2-bytes.txt
log "2b step timing kpc vs r2 on cards 2+3"
ssh wrx "cd $D && K3_ARM_A=kpc K3_ARM_B=r2 K3_TAG=r2-pair23 K3_CARDS=23 K3_DEV=0,1 K3_LAYERS=20 K3_ORDER='A B B A B A A B' bash bin/perf_k3_four.sh 600 > receipts-perf-r2-pair23.out 2>&1; tail -n 40 receipts-perf-r2-pair23.out" > $O/r2-pair23.tail 2>&1
tail -n 20 $O/r2-pair23.tail
ssh wrx "bash $D/spec/scripts/ext35_23.sh go"
log "R2-CHAIN-END"
