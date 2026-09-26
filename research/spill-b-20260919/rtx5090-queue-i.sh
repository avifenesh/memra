#!/usr/bin/env bash
# WP-B local 5090 queue-i (2026-09-26, after the rig reboot at 07:28Z ended queue-e to queue-h): the unrun local cells in
# decide-by order. DAY39 registered as a whole order under a new name (rtx5090-day39r; rtx5090-day39 holds the two boots
# that ran before the reboot), DAY39 addendum B, DAY40, DAY41 (the probe, RX and FX, offprev), DAY42 in the e shape,
# DAY41 addenda B and C (builds target/day41b), DAY43 (builds target/day43). Binaries live under the worktree's target/.
# Every GPU step waits for an idle rig and holds /tmp/memra-5090.lock alone (lanes A and F share the card). Builds run at
# nice 19 under a 600% CPU quota and are logged in cpu-concurrency.log. Never a reset, never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-i.log"; }
build() { # <out dir> <sha> <arm specs...>
  echo "$(date -u +%FT%TZ) WP-B queue-i build $1 (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19" \
    bash "$D/build-arms.sh" "$@"
}
log "queue-i start"
# 1. DAY39 registered (O5), whole order.
bash "$D/rtx5090-day39r/chain.sh"; log "day 39 registered (rerun) chain rc=$?"
# 2. DAY39 addendum B.
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
# 3. DAY40 (O3).
mkdir -p "$D/rtx5090-day40"; bash "$D/rtx5090-day40/chain.sh"; log "day 40 chain rc=$?"
# 4. DAY41 (O11, the original arm).
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
# 5. DAY42 in the e shape (O12).
mkdir -p "$D/rtx5090-day42e"; bash "$D/rtx5090-day42/chain.sh"; log "day 42e chain rc=$?"
# 6. DAY41 addenda B and C (O11, the revised arm).
mkdir -p "$D/rtx5090-day41b"
[ -s target/day41b/SHA256SUMS ] || { build "$WT/target/day41b" 23c296b583d0207413d4a2f3347882e729b4b29d tip offprev:day41b-nodoor.patch \
  > "$D/rtx5090-day41b/build.out" 2>&1; log "day 41b build rc=$?"; }
bash "$D/rtx5090-day41b/chain.sh"; log "day 41b chain rc=$?"
# 7. DAY43 (O13).
mkdir -p "$D/rtx5090-day43"
[ -s target/day43/SHA256SUMS ] || { build "$WT/target/day43" 87d9e00d16ea4436c48d561d99926c0d1677c0dd tip offprev:day43-nodoor.patch \
  > "$D/rtx5090-day43/build.out" 2>&1; log "day 43 build rc=$?"; }
bash "$D/rtx5090-day43/chain.sh"; log "day 43 chain rc=$?"
log "queue-i done"
