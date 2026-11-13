#!/bin/bash
# Coordinator 2026-11-13: the ext35 order gate's engine runs (ext35_gate_inner.sh) on ONE card under memra-gpu-run with
# that card's box lock, as dec/ext35/ext35_launch.sh: a hard stop (memra-gpu-run --timeout ends at STOP; a watchdog
# TERMs the lease at STOP + 10 s, KILLs it at STOP + 30 s, then KILLs any process carrying this run's EXT35G_MARK), the
# owner's thermal stop on the card (win_guard.sh), cores CORES (taskset). No timing: correctness tapes only.
# usage: setsid bash ext35_gate.sh CARD STOP_HH:MM CORES JOBS TAG
#        JOBS: BUILD:GROUP,... (GROUP ext35 | chat | long | long257) -> receipts-perf/ext35-gate/{launch,jobs,guard}-TAG.*
set -u
D=/home/access/dsv41
G=$D/receipts-perf/ext35-gate
CARD=$1
END=$(date -d "today $2" +%s)
CPUS=$3
JOBS=$4
TAG=$5
L=$G/launch-$TAG.log
mkdir -p $G
say() { echo "$(date +%H:%M:%S) $*" | tee -a $L; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$CARD '$1==i {print $2}')
TO=$(( END - $(date +%s) ))
[ $TO -gt 120 ] || { say "less than 2 min to the stop: not started"; exit 2; }
export EXT35G_MARK=ext35g-$$-$(date +%s)
PG=$(ps -o pgid= $$ | tr -d ' ')
bash $D/bin/win_guard.sh $$ "$PG" "$CARD" $G/guard-$TAG.log &
GP=$!
R=$D/leases/ext35g-$TAG-$(date +%Y%m%dT%H%M%S)
say "start $TAG on GPU$CARD ($UUID), cores $CPUS, jobs $JOBS, lease timeout $TO s (stop $(date -d @$END +%T)), mark $EXT35G_MARK; $(free -g | awk '/Mem:/ {print $7}') GB available"
$D/bin/memra-gpu-run --gpus "$UUID" --receipt "$R" --box-lock /mnt/Data/work/evals/gpu$CARD.lock --wait 60 --timeout "$TO" -- \
  env JOBS=$JOBS TAG=$TAG taskset -c $CPUS bash $D/bin/ext35_gate_inner.sh > $G/lease-$TAG.log 2>&1 &
MP=$!
while kill -0 $MP 2>/dev/null && [ "$(date +%s)" -lt $(( END + 10 )) ]; do sleep 5; done
if kill -0 $MP 2>/dev/null; then
  say "hard stop: TERM to the lease"; kill -TERM $MP
  for _ in $(seq 20); do kill -0 $MP 2>/dev/null || break; sleep 1; done
  kill -0 $MP 2>/dev/null && { say "lease still alive: KILL"; kill -KILL $MP; }
fi
wait $MP 2>/dev/null; say "lease ended rc $?; lease.json exit $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("exit_code"), d.get("timed_out", ""), d.get("started_at", ""), d.get("ended_at", ""))' $R/lease.json 2>/dev/null)"
for p in $(grep -l -s -a "EXT35G_MARK=$EXT35G_MARK" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3); do
  [ "$p" = $$ ] && continue
  [ "$p" = "$GP" ] && continue
  c=$(ps -o comm= -p "$p" 2>/dev/null); [ -n "$c" ] || continue
  say "KILL leftover $p ($c)"; kill -KILL "$p" 2>/dev/null
done
kill $GP 2>/dev/null
sleep 3
say "GPU$CARD compute processes now: $(nvidia-smi -i $CARD --query-compute-apps=pid,process_name --format=csv,noheader | grep -c .); thermal: $(tail -n 2 $G/guard-$TAG.log 2>/dev/null | tr '\n' ' ')"
say "EXT35G-END $TAG"
