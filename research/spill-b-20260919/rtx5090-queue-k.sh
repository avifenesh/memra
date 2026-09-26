#!/usr/bin/env bash
# WP-B local 5090 queue-k (2026-09-26): replaces queue-i's unrun items and queue-j, and YIELDS the card between cells
# (lead's card-sharing request: lanes A and F poll the lock in 120 s windows, so each cell here ends with YIELD_S=240 s
# of the lock free before the next idle check). A registered order that is one hold (DAY40's collector) stays whole and
# yields between orders. After the DAY39 rerun chain (pid argument) has exited: DAY37 addendum G's repro, DAY39B, DAY40,
# DAY41, DAY42e, DAY41B, DAY43, DAY44, DAY45. Builds run at nice 19 under 600%. Reads /proc only; never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
export YIELD_S=240
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-k.log"; }
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
wait_idle() { # <what>
  local deadline=$((SECONDS + 14400))
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "$1: rig not idle after 14400 s; skipped"; return 1; }
    sleep 30
  done
}
gate() { # <out dir> <binary>: the admit-mem burst gate at its defaults, then the yield
  wait_idle "gate $1" || return 1
  env -u MEMRA_KV_ALLOCATOR flock -w 600 /tmp/memra-5090.lock tools/admit-mem-burst-gate.sh "$MODEL" "$2" "$1" \
    > "$1.gate.log" 2>&1
  local rc=$?
  sleep "$YIELD_S"
  return $rc
}
build() { # <out dir> <sha> <arm specs...>
  echo "$(date -u +%FT%TZ) WP-B queue-k build $1 (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19" \
    bash "$D/build-arms.sh" "$@"
}
pid=${1:?the DAY39 rerun chain pid}
log "queue-k start: waiting for the DAY39 rerun chain pid $pid (YIELD_S=$YIELD_S)"
while [ -d "/proc/$pid" ]; do sleep 60; done
sleep "$YIELD_S"
# 1. DAY37 addendum G: the repro of 2.7's A1 red.
RG=$D/rtx5090-day37/r4-repro
for spec in r4-1:target/day37/r4/memra-server v3-1:target/b2/v3/memra-server r4-2:target/day37/r4/memra-server; do
  name=${spec%%:*}; bin=${spec#*:}
  echo "$(date -u +%FT%TZ) $name start bin=$(sha256sum "$bin" | cut -c1-16)" >> "$RG/run.log"
  gate "$RG/$name" "$bin"
  echo "$(date -u +%FT%TZ) $name rc=$? $(tail -1 "$RG/$name.gate.log")" >> "$RG/run.log"
done
log "day 37 addendum G repro done"
# 2. DAY39 addendum B.
R39=$D/rtx5090-day39b; mkdir -p "$R39"
BINS=$WT/target/b2 bash "$D/day39-run.sh" "$R39" v2-G2:v2:G2 v2-L64:v2:L64 v3-G2-r1:v3:G2 v3-L64-r1:v3:L64 \
  v2-off:v2:off v3-off:v3:off v3-G2-r2:v3:G2 v3-L64-r2:v3:L64 v1-G2:v1:G2 v1-L64:v1:L64
log "day 39B boots rc=$?"
gate "$R39/gate" "$WT/target/b2/v3/memra-server"; log "day 39B gate rc=$? $(tail -1 "$R39/gate.gate.log")"
B=$R39/boots
python3 "$D/day33-compare.py" --card rtx5090 red:G2:$B/v2-G2 red:L64:$B/v2-L64 green:G2:$B/v3-G2-r1 green:L64:$B/v3-L64-r1 \
  red:off:$B/v2-off green:off:$B/v3-off green:G2:$B/v3-G2-r2 green:L64:$B/v3-L64-r2 > "$R39/SUMMARY.txt" 2>&1
python3 "$D/day39-read.py" rtx5090 $B/v1-G2 $B/v1-L64 $B/v2-G2 $B/v2-L64 $B/v3-G2-r1 $B/v3-L64-r1 $B/v3-G2-r2 $B/v3-L64-r2 \
  $B/v2-off $B/v3-off > "$R39/READINGS.txt" 2>&1
# 3. DAY40 (one collector hold per order; the yield between orders).
mkdir -p "$D/rtx5090-day40"; bash "$D/rtx5090-day40/chain.sh"; log "day 40 chain rc=$?"; sleep "$YIELD_S"
# 4. DAY41 (the original arm).
R41=$D/rtx5090-day41; mkdir -p "$R41"
RIG_LOCK=/tmp/memra-5090.lock bash "$D/day41-probe.sh" "$R41/probe" "$WT/target/day41/concat-prime-probe" "$MODEL" > "$R41/probe.log" 2>&1
log "day 41 probe rc=$?"
for route in plain spec; do
  BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server bash "$D/day41-run.sh" "$R41" \
    "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" "rx-$route-O2-rewind:rewind:$route:RX" \
    "rx-$route-O2-keep:keep:$route:RX" "fx-$route-keep:keep:$route:FX" "fx-$route-rewind:rewind:$route:FX"
  log "day 41 $route boots rc=$?"
done
BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server bash "$D/day41-run.sh" "$R41" offprev:offprev:plain:RX6
python3 "$D/day41-read.py" rtx5090 "$R41" > "$R41/read.log" 2>&1
# 5. DAY42 in the e shape.
mkdir -p "$D/rtx5090-day42e"; bash "$D/rtx5090-day42/chain.sh"; log "day 42e chain rc=$?"
# 6. DAY41 addenda B and C.
mkdir -p "$D/rtx5090-day41b"
[ -s target/day41b/SHA256SUMS ] || { build "$WT/target/day41b" 23c296b583d0207413d4a2f3347882e729b4b29d tip offprev:day41b-nodoor.patch \
  > "$D/rtx5090-day41b/build.out" 2>&1; log "day 41b build rc=$?"; }
bash "$D/rtx5090-day41b/chain.sh"; log "day 41b chain rc=$?"
# 7. DAY43.
mkdir -p "$D/rtx5090-day43"
[ -s target/day43/SHA256SUMS ] || { build "$WT/target/day43" 87d9e00d16ea4436c48d561d99926c0d1677c0dd tip offprev:day43-nodoor.patch \
  > "$D/rtx5090-day43/build.out" 2>&1; log "day 43 build rc=$?"; }
bash "$D/rtx5090-day43/chain.sh"; log "day 43 chain rc=$?"
# 8. DAY44.
mkdir -p "$D/rtx5090-day44"; bash "$D/rtx5090-day44/chain.sh"; log "day 44 chain rc=$?"
# 9. DAY45.
mkdir -p "$D/rtx5090-day45"; bash "$D/rtx5090-day45/chain.sh"; log "day 45 chain rc=$?"
log "queue-k done"
