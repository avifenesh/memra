#!/usr/bin/env bash
# GB10 (under one sparklock lease): one arm of the r1 bisect on the order gate's chat group (13 vendor tapes, teacher-
# forced, dsv41-run --cases, one card, layers streamed), WITHOUT the route log. ARM = the MEMRA_R1X_SEG value
# (router | comp | all = unset, r1's program). Memory guard: stop under 12 GB available. Out: out/r1x-tapes/ARM/chat.
# usage: r1x_tapes.sh ARM
set -u
ARM=$1
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/r1x-tapes/$ARM
mkdir -p $O/chat
: > $O/runs.txt
export MEMRA_FULL_PREC=1
if [ "$ARM" = all ]; then unset MEMRA_R1X_SEG; else export MEMRA_R1X_SEG=$ARM; fi
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
# once, in the first arm's lease: the fp32 FMA microbench (kpc / r1 / r2 forms; same source as the 5090 runs) on
# this GB10, 121a, weights L2-resident (1 copy) and rotated (48 copies), ~3 min before the tapes
BD=$R/bench-gb10
if [ -s $BD/ffma_bench.cu ] && [ ! -e $BD/DONE ]; then
  (cd $BD && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_121a -o ffma_bench ffma_bench.cu > build.log 2>&1 \
    && ./ffma_bench 15 1 > l1.txt 2>&1 && ./ffma_bench 15 48 > l48.txt 2>&1; echo "bench rc $?" > DONE)
  echo "gb10 bench $(cat $BD/DONE) $(date +%T)" >> $O/runs.txt
fi
sed "s|/home/access/dsv41/dec/ids|$D/dec/ids|g" $D/dec/chat-forced.txt > $O/chat-forced.txt
echo "chat start $(date +%T), $(avail) GB available, MEMRA_R1X_SEG=${MEMRA_R1X_SEG-unset}" >> $O/runs.txt
$R/bins/r1x/dsv41-run $C --token-map $TM --cases $O/chat-forced.txt --out $O/chat > $O/chat.log 2>&1 &
P=$!
while kill -0 $P 2> /dev/null; do
  a=$(avail)
  if [ $a -lt 12 ]; then echo "STOPPED: $a GB available" >> $O/chat.log; kill $P; break; fi
  sleep 10
done
wait $P
echo "chat rc $? $(grep -h 'dsv41-run: [0-9]* forwards' $O/chat.log | tail -n 1) end $(date +%T)" >> $O/runs.txt
(cd $O/chat && find . -type f | sort | xargs sha256sum) > $O/chat.sha256
echo "TAPES-DONE $ARM $(date +%T)" >> $O/runs.txt
