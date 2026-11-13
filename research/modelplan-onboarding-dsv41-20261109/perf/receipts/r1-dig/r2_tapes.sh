#!/usr/bin/env bash
# GB10 (under one sparklock lease): r2 on one group of the order gate's vendor tapes (teacher-forced, dsv41-run
# --cases, one card, layers streamed), with the route log as the current route's tapes were run. Chat: byte check
# against k3's chat tapes on this unit (out/tapes-k3/chat, = the measurement box's k3 run). Out: out/r2-tapes/GROUP.
# usage: r2_tapes.sh chat|long
set -u
G=$1
R=$HOME/.cache/dsv41-speed
D=$R/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
O=$R/out/r2-tapes
mkdir -p $O/$G
: > $O/runs-$G.txt
export MEMRA_FULL_PREC=1
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
sed "s|/home/access/dsv41/dec/ids|$D/dec/ids|g" $D/dec/$G-forced.txt > $O/$G-forced.txt
echo "$G start $(date +%T), $(avail) GB available" >> $O/runs-$G.txt
$R/bins/r2/dsv41-run $C --token-map $TM --cases $O/$G-forced.txt --out $O/$G --routes > $O/$G.log 2>&1 &
P=$!
while kill -0 $P 2> /dev/null; do
  a=$(avail)
  if [ $a -lt 12 ]; then echo "STOPPED: $a GB available" >> $O/$G.log; kill $P; break; fi
  sleep 10
done
wait $P
echo "$G rc $? $(grep -h 'dsv41-run: [0-9]* forwards\|route log' $O/$G.log | tail -n 2 | tr '\n' ' ') end $(date +%T)" >> $O/runs-$G.txt
(cd $O/$G && find . -type f | sort | xargs sha256sum) > $O/$G.sha256
if [ "$G" = chat ]; then
  n=0; bad=0
  for f in $(cd $R/out/tapes-k3/chat && find . -type f | sort); do
    n=$((n+1)); cmp -s $R/out/tapes-k3/chat/$f $O/chat/$f || { bad=$((bad+1)); echo "DIFF $f" >> $O/runs-$G.txt; }
  done
  echo "chat against k3's chat tapes: $n files compared, $bad differ" >> $O/runs-$G.txt
fi
echo "R2-TAPES-DONE $G $(date +%T)" >> $O/runs-$G.txt
