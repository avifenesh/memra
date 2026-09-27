#!/usr/bin/env bash
# WP-B local 5090 queue (2026-09-25, after the target-card sitting): once queue-b (pid argument) has exited and
# build-arms.sh has written target/b2/SHA256SUMS: DAY37 addendum F's r4 boots that the first batch never ran (then the
# corrected reader over the r4 root), DAY38 addendum D, DAY39 addendum B (then the admission gate on v3 and the
# readers). Every GPU step waits for an idle rig and holds /tmp/memra-5090.lock alone. Reads /proc only; never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-c.log"; }
pid=${1:?queue-b pid}
log "queue-c start: waiting for pid $pid and target/b2"
while [ -d "/proc/$pid" ]; do sleep 60; done
until [ -s target/b2/SHA256SUMS ]; do
  command grep -q "failed\|does not apply" /tmp/wpb-build-b2.out 2>/dev/null && { log "b2 builds failed: $(tail -1 /tmp/wpb-build-b2.out)"; exit 1; }
  sleep 60
done
cp target/b2/SHA256SUMS "$D/rtx5090-b2-binaries.sha256"
for a in v3 red38 v2 v1; do cp "target/b2/$a/source.patches" "$D/rtx5090-b2-$a.patches"; done
log "b2 present: $(tr '\n' ' ' < target/b2/SHA256SUMS | cut -c1-300)"
# 1. DAY37 r4, the boots the first batch did not run (addendum F), same binaries.
R4=$D/rtx5090-day37/r4
LANE_BIN=$WT/target/day37/r4/memra-server MAIN_BIN=$WT/target/day37/main/memra-server bash "$D/rtx5090-day37/boots-f.sh" "$R4" \
  burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff \
  fault-ensure:vmm-ensurefault:g2 fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2
log "day 37 r4 missed boots rc=$?"
python3 "$D/day37-read.py" rtx5090 "$R4" gates-r4 > "$R4/read.log" 2>&1
log "day 37 r4 reader rc=$?"
# 2. DAY38 addendum D.
R38=$D/rtx5090-day38d; mkdir -p "$R38"
BIN=$WT/target/b2/v3/memra-server RED_BIN=$WT/target/b2/red38/memra-server bash "$D/day38d-run.sh" "$R38" \
  main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off \
  fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red \
  vmm-off:vmm-off vmm-on:vmm-on
log "day 38D boots rc=$?"
python3 "$D/day38d-read.py" rtx5090 "$R38" > "$R38/read.log" 2>&1
# 3. DAY39 addendum B.
R39=$D/rtx5090-day39b; mkdir -p "$R39"
BINS=$WT/target/b2 bash "$D/day39-run.sh" "$R39" v2-G2:v2:G2 v2-L64:v2:L64 v3-G2-r1:v3:G2 v3-L64-r1:v3:L64 \
  v2-off:v2:off v3-off:v3:off v3-G2-r2:v3:G2 v3-L64-r2:v3:L64 v1-G2:v1:G2 v1-L64:v1:L64
log "day 39B boots rc=$?"
deadline=$((SECONDS + 7200))
until flock -n /tmp/memra-5090.lock true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  [ $SECONDS -ge $deadline ] && { log "gate: rig not idle after 7200 s; not run"; break; }; sleep 30
done
flock -w 7200 /tmp/memra-5090.lock tools/admit-mem-burst-gate.sh "$MODEL" "$WT/target/b2/v3/memra-server" "$R39/gate" > "$R39/gate.log" 2>&1
log "day 39B gate rc=$? $(tail -1 "$R39/gate.log")"
B=$R39/boots
python3 "$D/day33-compare.py" --card rtx5090 red:G2:$B/v2-G2 red:L64:$B/v2-L64 green:G2:$B/v3-G2-r1 green:L64:$B/v3-L64-r1 \
  red:off:$B/v2-off green:off:$B/v3-off green:G2:$B/v3-G2-r2 green:L64:$B/v3-L64-r2 > "$R39/SUMMARY.txt" 2>&1
python3 "$D/day39-read.py" rtx5090 $B/v1-G2 $B/v1-L64 $B/v2-G2 $B/v2-L64 $B/v3-G2-r1 $B/v3-L64-r1 $B/v3-G2-r2 $B/v3-L64-r2 \
  $B/v2-off $B/v3-off > "$R39/READINGS.txt" 2>&1
log "queue-c done"
