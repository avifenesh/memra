#!/usr/bin/env bash
# DAY49 local runs (1.3, addendum A): the 5090 (the 9B). The builds run first outside the card's lock (the gate's release
# binary and target/day49 from 02dbdfa40); the gate's arm i twice, then the serving shape (burst 8 x 6,144, no warm,
# MEMRA_STEP_OOM_FAULT=1, off and on in both orders). Every GPU step waits for an idle card and yields after. Never a
# signal to anything this lane did not start.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day49
cd "$WT" || exit 1
echo "$(date -u +%FT%TZ) WP-B DAY49 builds (nice 19, CPUQuota=600%)" >> research/spill-b-20260919/cpu-concurrency.log
W="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19"
$W cargo build --release -p memra-server > "$D/gate-build.log" 2>&1 || { echo "$(date -u +%FT%TZ) gate build failed" >> "$D/run.log"; exit 2; }
[ -s target/day49/SHA256SUMS ] || { export WT; TARGET=$WT/target WRAP="$W" bash research/spill-b-20260919/build-arms.sh \
  "$WT/target/day49" "02dbdfa408a5d0215248c0ec2849cd599ca64083" tip > "$D/build.out" 2>&1 || { echo "$(date -u +%FT%TZ) arm build failed" >> "$D/run.log"; exit 2; }; }
idle() { flock -n /tmp/memra-5090.lock true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
for run in g1 g2; do
  deadline=$((SECONDS + 14400))
  until idle; do [ $SECONDS -ge $deadline ] && { echo "$(date -u +%FT%TZ) $run: not idle; not run" >> "$D/run.log"; exit 3; }; sleep 30; done
  echo "$(date -u +%FT%TZ) gate $run start HEAD=$(git rev-parse HEAD)" >> "$D/run.log"
  HFG_ARMS=i HFG_OUT="$D/$run" flock -w 600 /tmp/memra-5090.lock tools/health-fault-gate.sh > "$D/$run.log" 2>&1
  echo "$(date -u +%FT%TZ) gate $run rc=$? $(tail -1 "$D/$run.log")" >> "$D/run.log"
  sleep 240
done
BIN=$WT/target/day49/tip/memra-server YIELD_S=240 CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64" \
  bash research/spill-b-20260919/day49-run.sh "$D" O1-off:off O1-on:on O2-on:on O2-off:off
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$D/run.log"
