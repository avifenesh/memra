#!/usr/bin/env bash
# The order-gate tapes of one group (G = chat | long), teacher-forced through the whole trunk with the route log, and
# the files compared byte for byte with the CUTLASS build's (f4) tapes. usage: tapes.sh TAG G
# Done-file out/tapes-$TAG/$G-runs.txt, marker TAPES-$G-DONE.
set -u
TAG=$1; G=$2
R=$HOME/.cache/dsv41-cfix
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
REF=$HOME/.cache/dsv41-cutlass/out/tapes-f4/$G
export MEMRA_FULL_PREC=1
B=$R/bins/$TAG
O=$R/out/tapes-$TAG; mkdir -p $O/$G
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
sed "s|/home/access/dsv41/dec/ids|$D/dec/ids|g" $D/dec/$G-forced.txt > $O/$G-forced.txt
echo "$G start $(date +%T), $(avail) GB available" > $O/$G-runs.txt
timeout 29m $B/dsv41-run $C --token-map $TM --cases $O/$G-forced.txt --out $O/$G --routes > $O/$G.log 2>&1
echo "$G rc $? $(grep -h 'dsv41-run: [0-9]* forwards\|route log' $O/$G.log | tail -n 2 | tr '\n' ' ') end $(date +%T), $(avail) GB available" >> $O/$G-runs.txt
(cd $O/$G && find . -type f | sort | xargs sha256sum) > $O/$G.sha256
if [ -d $REF ]; then (cd $REF && find . -type f | sort | xargs sha256sum) > $O/$G-ref.sha256
elif [ -f $R/tapes-ref-$G.sha256 ]; then cp $R/tapes-ref-$G.sha256 $O/$G-ref.sha256; fi
if [ -s $O/$G-ref.sha256 ]; then
  n=$(wc -l < $O/$G.sha256); nr=$(wc -l < $O/$G-ref.sha256)
  bad=$(diff <(awk '{print $1, $2}' $O/$G.sha256) <(awk '{print $1, $2}' $O/$G-ref.sha256) | grep -c '^[<>]')
  echo "$G tapes: $n files, ref $nr files, $bad diff lines against f4 ($([ $bad = 0 ] && [ $n = $nr ] && echo BYTE-IDENTICAL || echo DIFFER))" >> $O/$G-runs.txt
fi
echo "TAPES-$G-DONE $(date +%T)" >> $O/$G-runs.txt
