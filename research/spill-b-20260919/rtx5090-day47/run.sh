#!/usr/bin/env bash
# DAY47 local runs (1.3): the health gate's arms g and h on the 5090 (the 9B), twice. The release build runs first,
# outside the card's lock; each run waits for an idle card (lock free, no compute app), holds the lock, then yields.
# Never a signal to anything this lane did not start.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day47
cd "$WT" || exit 1
echo "$(date -u +%FT%TZ) WP-B DAY47 release build (nice 19, CPUQuota=600%)" >> research/spill-b-20260919/cpu-concurrency.log
systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19 cargo build --release -p memra-server \
  > "$D/build.log" 2>&1 || { echo "$(date -u +%FT%TZ) build failed" >> "$D/run.log"; exit 2; }
for run in r1 r2; do
  deadline=$((SECONDS + 14400))
  until flock -n /tmp/memra-5090.lock true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    [ $SECONDS -ge $deadline ] && { echo "$(date -u +%FT%TZ) $run: card not idle after 14400 s; not run" >> "$D/run.log"; exit 3; }
    sleep 30
  done
  echo "$(date -u +%FT%TZ) $run start HEAD=$(git rev-parse HEAD)" >> "$D/run.log"
  HFG_ARMS=g,h HFG_OUT="$D/$run" flock -w 600 /tmp/memra-5090.lock tools/health-fault-gate.sh > "$D/$run.log" 2>&1
  echo "$(date -u +%FT%TZ) $run rc=$? $(tail -1 "$D/$run.log")" >> "$D/run.log"
  sleep 240
done
