#!/bin/bash
# memra#1324: the same traced block as receipts-perf/t1324 (40 layers, four ranks, device-push, rows 4 9 10, 6 rounds,
# one nsys trace a row count) for arm B, through bin/perf_cfix_trace.sh, every box lock held with flock -n (a held
# lock aborts), hard budget 300 s; locks released when the traces are on disk, then the CPU-only exports and the
# per-kernel and per-family tables against t1324's base (route2) and main (main25) traces.
# usage: moe_trace.sh TAG ARM_B
set -u
TAG=$1 B=$2
D=/home/access/dsv41
EV=/mnt/Data/work/evals
S=$D/spec
R=$D/receipts-perf/$TAG
T0=$D/receipts-perf/t1324
cd $D || exit 1
say() { echo "$(date +%T) $*"; }
[ -x perf-bins/$B/dsv41-tpbench ] || { say "missing perf-bins/$B"; exit 2; }
[ -e /mnt/Data/work/memra-wrx-ds/QUIET ] && { say "QUIET set: not run"; exit 3; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$apps" ] || { say "ABORT: compute apps on the box: $apps"; exit 3; }
for i in 0 1 2 3; do
  exec {fd}>>"$EV/gpu$i.lock"
  if ! flock -n "$fd"; then say "ABORT: gpu$i.lock is held by someone else; nothing run"; exit 2; fi
  eval "FD$i=$fd"
done
export TP4_HELD=1
say "TRACE-START $TAG B=$B: all four box locks held; load $(cut -d' ' -f1-3 /proc/loadavg)"
mkdir -p $R
env TR_BINS="$B" TR_ROWS="4 9 10" TR_TAG=$TAG bash bin/perf_cfix_trace.sh 300 > $R/trace.out 2>&1 &
tp=$!
until grep -q TROWS-CARDS-DONE $R/trace.out 2>/dev/null; do
  kill -0 $tp 2>/dev/null || break
  sleep 5
done
for i in 0 1 2 3; do eval "fd=\$FD$i"; flock -u "$fd"; eval "exec $fd>&-"; done
touch $EV/dsv41.moe1324-trace-done
say "all four box locks released"
wait $tp
say "trace script rc $?"
for n in 4 9 10; do
  args="route2=$T0/route2-rows$n/trace.sqlite main25=$T0/main25-rows$n/trace.sqlite $B=$R/$B-rows$n/trace.sqlite"
  taskset -c 28-31,60-63 python3 $S/scripts/kern_cmp.py 6 40 $args > $R/kern-rows$n.md 2>&1
  taskset -c 28-31,60-63 python3 $S/scripts/kern_fam.py 6 40 $args > $R/fam-rows$n.md 2>&1
done
say MOE-TRACE-DONE
