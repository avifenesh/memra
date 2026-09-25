#!/usr/bin/env bash
# WP-B local 5090 queue (2026-09-25, while the card awaits its reset): waits until the card reads healthy (a numeric
# temperature from nvidia-smi; `[GPU requires reset]` since 01:25Z, Xid 119 then 154), then runs every unrun local cell in
# decide-by order: DAY37 r4 (the stream pairs from stream-O1-2, the seven boots addendum F names, the reader), the
# registered DAY38 and DAY39 halves, DAY38 addendum D, DAY39 addendum B (and the admission gate on v3), DAY41 (the probe,
# RX and FX on both routes both orders, offprev). Every GPU step waits for an idle rig and holds /tmp/memra-5090.lock
# alone. It only reads the card's state; never a reset, never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-d.log"; }
healthy() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | command grep -qE '^[0-9]+$'; }
log "queue-d start: waiting for the card's health"
deadline=$((SECONDS + 96 * 3600))
until healthy; do [ $SECONDS -ge $deadline ] && { log "card not healthy after 96 h; nothing run"; exit 3; }; sleep 120; done
log "card healthy: $(nvidia-smi --query-gpu=name,temperature.gpu,power.draw --format=csv,noheader)"
# 1. DAY37 r4 (O1, decide-by 2026-10-04).
R4=$D/rtx5090-day37/r4
[ -d "$R4/boots/stream-O1-2-pooled" ] && mv "$R4/boots/stream-O1-2-pooled" "$R4/boots/stream-O1-2-pooled.gpu-reset-rc5"
args=(stream-O1-2-pooled:pooled:stream stream-O1-2-vmm:vmm:stream)
for k in 3 4 5; do args+=("stream-O1-$k-pooled:pooled:stream" "stream-O1-$k-vmm:vmm:stream"); done
for k in 1 2 3 4 5; do args+=("stream-O2-$k-vmm:vmm:stream" "stream-O2-$k-pooled:pooled:stream"); done
args+=(burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff
       fault-ensure:vmm-ensurefault:g2 fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2)
LANE_BIN=$WT/target/day37/r4/memra-server MAIN_BIN=$WT/target/day37/main/memra-server bash "$D/rtx5090-day37/boots-f.sh" "$R4" "${args[@]}"
log "day 37 r4 boots rc=$?"
python3 "$D/day37-read.py" rtx5090 "$R4" gates-r4 > "$R4/read.log" 2>&1
# 2. DAY38 registered, then addendum D (O2, decide-by 2026-10-06).
bash "$D/rtx5090-day38/chain.sh"; log "day 38 registered chain rc=$?"
R38=$D/rtx5090-day38d; mkdir -p "$R38"
BIN=$WT/target/b2/v3/memra-server RED_BIN=$WT/target/b2/red38/memra-server bash "$D/day38d-run.sh" "$R38" \
  main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off \
  fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red vmm-off:vmm-off vmm-on:vmm-on
log "day 38D boots rc=$?"
python3 "$D/day38d-read.py" rtx5090 "$R38" > "$R38/read.log" 2>&1
# 3. DAY39 registered, then addendum B (O5, before O3; decide-by of the door 2026-10-07).
bash "$D/rtx5090-day39/chain.sh"; log "day 39 registered chain rc=$?"
R39=$D/rtx5090-day39b; mkdir -p "$R39"
BINS=$WT/target/b2 bash "$D/day39-run.sh" "$R39" v2-G2:v2:G2 v2-L64:v2:L64 v3-G2-r1:v3:G2 v3-L64-r1:v3:L64 \
  v2-off:v2:off v3-off:v3:off v3-G2-r2:v3:G2 v3-L64-r2:v3:L64 v1-G2:v1:G2 v1-L64:v1:L64
log "day 39B boots rc=$?"
flock -w 7200 /tmp/memra-5090.lock tools/admit-mem-burst-gate.sh "$MODEL" "$WT/target/b2/v3/memra-server" "$R39/gate" > "$R39/gate.log" 2>&1
log "day 39B gate rc=$? $(tail -1 "$R39/gate.log")"
B=$R39/boots
python3 "$D/day33-compare.py" --card rtx5090 red:G2:$B/v2-G2 red:L64:$B/v2-L64 green:G2:$B/v3-G2-r1 green:L64:$B/v3-L64-r1 \
  red:off:$B/v2-off green:off:$B/v3-off green:G2:$B/v3-G2-r2 green:L64:$B/v3-L64-r2 > "$R39/SUMMARY.txt" 2>&1
python3 "$D/day39-read.py" rtx5090 $B/v1-G2 $B/v1-L64 $B/v2-G2 $B/v2-L64 $B/v3-G2-r1 $B/v3-L64-r1 $B/v3-G2-r2 $B/v3-L64-r2 \
  $B/v2-off $B/v3-off > "$R39/READINGS.txt" 2>&1
# 4. DAY41 (O11, decide-by 2026-10-09).
R41=$D/rtx5090-day41; mkdir -p "$R41"
bash "$D/day41-probe.sh" "$R41/probe" "$WT/target/day41/concat-prime-probe" "$MODEL" > "$R41/probe.log" 2>&1
log "day 41 probe rc=$?"
for route in plain spec; do
  BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server bash "$D/day41-run.sh" "$R41" \
    "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" "rx-$route-O2-rewind:rewind:$route:RX" \
    "rx-$route-O2-keep:keep:$route:RX" "fx-$route-keep:keep:$route:FX" "fx-$route-rewind:rewind:$route:FX"
  log "day 41 $route boots rc=$?"
done
BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server bash "$D/day41-run.sh" "$R41" offprev:offprev:plain:RX6
python3 "$D/day41-read.py" rtx5090 "$R41" > "$R41/read.log" 2>&1
log "queue-d done"
