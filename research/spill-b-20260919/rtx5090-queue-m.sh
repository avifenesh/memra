#!/usr/bin/env bash
# WP-B local 5090 queue-m (2026-09-26): takes over queue-k's unrun items. queue-k (pid 2946742) lost DAY39B's boots
# after v2-G2: day39-run.sh exits 3 on the first boot whose idle wait runs out (7200 s while another lane's cells held
# the card back to back), and every later spec of that call went unrun. queue-k was stopped by the lane in its next
# idle wait (its log says so). Same cells, specs, binaries, env, receipt paths and readers as queue-k and the chains it
# called (rtx5090-day40/chain.sh, rtx5090-day42/chain.sh, rtx5090-day41b, day43, day44 and day45 chain.sh). What
# changes is only how a spec that did not run is handled:
#   - a spec whose `boot <name> start` line is in its runner's run.log ran, and is never asked again;
#   - a runner call that exits 3 (an idle wait ran out, nothing started for that spec) is asked again with the specs
#     that have not started, up to eight times;
#   - the admission gate takes /tmp/memra-5090.lock in this runner (fd 9) after the idle check and runs under it; a
#     lock not taken within 60 s goes back to the idle wait (queue-k's `flock -w 600` wrapper recorded a timeout as a run);
#   - DAY40's collector (LOCK_NB) is asked again after `Resource temporarily unavailable`, as chain-r5's gates are;
#   - the DAY41 probe is asked again for a cell whose log holds only the lock wait's `exit=1`.
# YIELD_S=240 after every unit, as queue-k. Builds at nice 19 under 600%. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
LOCK=/tmp/memra-5090.lock
export YIELD_S=240
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-m.log"; }
idle() {
  flock -n "$LOCK" true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
wait_idle() { # <what>: 0 when idle, 3 after 28800 s
  local deadline=$((SECONDS + 28800))
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "$1: card not idle after 28800 s"; return 3; }
    sleep 30
  done
}
hold() { # <what>: the lock on fd 9 after an idle wait
  while :; do
    wait_idle "$1" || return 3
    exec 9>"$LOCK"
    flock -w 60 9 && return 0
    exec 9>&-
    log "$1: lock taken by another runner inside the window; back to the idle wait"
  done
}
started() { command grep -q "boot $2 start" "$1/run.log" 2>/dev/null; }
ask() { # <runner> <R> <spec>...: the specs that have not started; again after an exit 3, up to eight times
  local runner=$1 R=$2; shift 2
  local try spec rc left
  for try in 1 2 3 4 5 6 7 8; do
    left=(); for spec in "$@"; do started "$R" "${spec%%:*}" || left+=("$spec"); done
    [ ${#left[@]} = 0 ] && return 0
    [ $try = 1 ] || log "$(basename "$runner") ${left[*]}: asked again (try $try) after an idle-wait timeout"
    bash "$D/$runner" "$R" "${left[@]}"; rc=$?
    [ $rc = 3 ] || return $rc
  done
  log "$(basename "$runner") ${left[*]}: not run after eight idle waits"; return 3
}
gate() { # <out prefix> <binary>: the admit-mem burst gate at its defaults, under the lock, then the yield
  [ -s "$1.gate.log" ] && command grep -q "ADMIT-MEM BURST GATE" "$1.gate.log" && { log "gate $1: done already"; return 0; }
  hold "gate $1" || return 3
  env -u MEMRA_KV_ALLOCATOR tools/admit-mem-burst-gate.sh "$MODEL" "$2" "$1" > "$1.gate.log" 2>&1
  local rc=$?
  exec 9>&-
  sleep "$YIELD_S"
  return $rc
}
build() { # <out dir> <sha> <arm specs...>
  echo "$(date -u +%FT%TZ) WP-B queue-m build $1 (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19" \
    bash "$D/build-arms.sh" "$@"
}
log "queue-m start HEAD=$(git rev-parse HEAD) (YIELD_S=$YIELD_S)"

# 2. DAY39 addendum B (queue-k ran v2-G2 at 15:20Z).
R39=$D/rtx5090-day39b
( export BINS=$WT/target/b2
  ask day39-run.sh "$R39" v2-G2:v2:G2 v2-L64:v2:L64 v3-G2-r1:v3:G2 v3-L64-r1:v3:L64 \
    v2-off:v2:off v3-off:v3:off v3-G2-r2:v3:G2 v3-L64-r2:v3:L64 v1-G2:v1:G2 v1-L64:v1:L64 )
log "day 39B boots rc=$?"
gate "$R39/gate" "$WT/target/b2/v3/memra-server"; log "day 39B gate rc=$? $(tail -1 "$R39/gate.gate.log")"
B=$R39/boots
python3 "$D/day33-compare.py" --card rtx5090 red:G2:$B/v2-G2 red:L64:$B/v2-L64 green:G2:$B/v3-G2-r1 green:L64:$B/v3-L64-r1 \
  red:off:$B/v2-off green:off:$B/v3-off green:G2:$B/v3-G2-r2 green:L64:$B/v3-L64-r2 > "$R39/SUMMARY.txt" 2>&1
python3 "$D/day39-read.py" rtx5090 $B/v1-G2 $B/v1-L64 $B/v2-G2 $B/v2-L64 $B/v3-G2-r1 $B/v3-L64-r1 $B/v3-G2-r2 $B/v3-L64-r2 \
  $B/v2-off $B/v3-off > "$R39/READINGS.txt" 2>&1

# 3. DAY40 (rtx5090-day40/chain.sh's steps: one collector hold per order, the yield between orders).
R=$D/rtx5090-day40; BIN=$WT/target/day40/tip/memra-server; mkdir -p "$R"
c40() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
sha256sum "$BIN" > "$R/binary.sha256"; cp "$WT/target/day40/tip/source.commit" "$R/build-source.txt"
c40 "chain start HEAD=$(git rev-parse HEAD) bin=$(cut -c1-16 "$R/binary.sha256") built_from=$(cat "$R/build-source.txt") (queue-m)"
for order in O1 O2; do
  command grep -q "order $order: collector exit=0" "$R/chain.log" && continue
  for attempt in $(seq 0 90); do
    wait_idle "day 40 order $order" || break
    c40 "order $order: collector start (attempt $attempt)"
    rm -rf "$R/collector-$order"
    env R="$R" RIG_LOCK=$LOCK BIN="$BIN" WT="$WT" MEMRA_CTX=65536 BURST=32 \
      python3 tools/tier-battery.py --rig rtx5090 --timeout 12600 --out "$R/collector-$order" --external-lock --execute \
      bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ "$order" > "$R/collector-$order.log" 2>&1
    rc=$?
    c40 "order $order: collector exit=$rc"
    command grep -q 'Resource temporarily unavailable' "$R/collector-$order.log" || break
    sleep 120
  done
  sleep "$YIELD_S"
done
B=research/spill-b-20260919/rtx5090-day40/boots
( cd "$WT" && python3 "$D/day34-compare.py" --card rtx5090 --served-ctx 65536 --model-ctx 262144 --registry-value 32768 \
    --survey context O1:$B/O1-off O1:$B/O1-on2048 O1:$B/O1-on8192 O1:$B/O1-on32768 O2:$B/O2-off O2:$B/O2-on2048 \
    O2:$B/O2-on8192 O2:$B/O2-on32768 > "$R/SUMMARY.txt" 2>&1 )
python3 "$D/day31-faults.py" "$R"/boots/O*/ > "$R/FAULTS.txt" 2>&1
python3 "$D/day40-read.py" rtx5090 "$R/boots" > "$R/READINGS.txt" 2>&1
c40 "chain done"; log "day 40 done"

# 4. DAY41 (the original arm): the probe, then the boots.
R41=$D/rtx5090-day41; mkdir -p "$R41"
for try in 1 2 3 4 5 6 7 8; do
  pending=0; executed_fail=0
  for L in 6144 30720; do for K in 32 256; do
    c=$R41/probe/primepath-L$L-K$K.log
    if [ ! -s "$c" ] || [ "$(cat "$c")" = "exit=1" ]; then pending=1
    elif ! command grep -q "^verdict rewind" "$c"; then executed_fail=1; fi
  done; done
  [ $pending = 0 ] && break
  # The probe re-runs any cell without a verdict, so a cell that ran and failed stops the asking (never a rerun).
  [ $executed_fail = 1 ] && { log "day 41 probe: a cell ran without a verdict; the pending cells are not asked again"; break; }
  for L in 6144 30720; do for K in 32 256; do
    c=$R41/probe/primepath-L$L-K$K.log; [ -s "$c" ] && [ "$(cat "$c")" = "exit=1" ] && rm -f "$c"
  done; done
  wait_idle "day 41 probe" || break
  RIG_LOCK=$LOCK bash "$D/day41-probe.sh" "$R41/probe" "$WT/target/day41/concat-prime-probe" "$MODEL" >> "$R41/probe.log" 2>&1
  log "day 41 probe rc=$? (try $try)"
done
for route in plain spec; do
  ( export BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server
    ask day41-run.sh "$R41" "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" \
      "rx-$route-O2-rewind:rewind:$route:RX" "rx-$route-O2-keep:keep:$route:RX" "fx-$route-keep:keep:$route:FX" \
      "fx-$route-rewind:rewind:$route:FX" )
  log "day 41 $route boots rc=$?"
done
( export BIN=$WT/target/day41/tip/memra-server PREV_BIN=$WT/target/day41/offprev/memra-server
  ask day41-run.sh "$R41" offprev:offprev:plain:RX6 )
python3 "$D/day41-read.py" rtx5090 "$R41" > "$R41/read.log" 2>&1
log "day 41 done"

# 5. DAY42 in the e shape (rtx5090-day42/chain.sh's steps).
R=$D/rtx5090-day42e; mkdir -p "$R"
cp target/day42e/SHA256SUMS "$R/binaries.sha256"; cp target/day42e/tip/source.commit "$R/tip.source"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256") (queue-m)" >> "$R/chain.log"
( export BIN=$WT/target/day42e/tip/memra-server HOST_MB=8192 BURST=32 WARM=16 WARM_TOKENS=4096 WARM_MAX_TOKENS=16 BURST_MAX_CTX=34816
  ask day42-run.sh "$R" ontick-O1:ontick offtick-O1:offtick offtick-O2:offtick ontick-O2:ontick \
    ontick-nocontracts:ontick-nocontracts fault-d2h-delay:fault-d2h-delay fault-d2h-source-flip:fault-d2h-source-flip \
    fault-sources-helper-gone:fault-sources-helper-gone )
python3 "$D/day42-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"; log "day 42e done"

# 6. DAY41 addenda B and C (rtx5090-day41b/chain.sh's steps).
R=$D/rtx5090-day41b; mkdir -p "$R"
[ -s target/day41b/SHA256SUMS ] || { build "$WT/target/day41b" 23c296b583d0207413d4a2f3347882e729b4b29d tip offprev:day41b-nodoor.patch \
  > "$R/build.out" 2>&1; log "day 41b build rc=$?"; }
cp target/day41b/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256") (queue-m)" >> "$R/chain.log"
for route in plain spec; do
  ( export BIN=$WT/target/day41b/tip/memra-server PREV_BIN=$WT/target/day41b/offprev/memra-server
    ask day41-run.sh "$R" "rx-$route-O1-keep:keep:$route:RX" "rx-$route-O1-rewind:rewind:$route:RX" \
      "rx-$route-O2-rewind:rewind:$route:RX" "rx-$route-O2-keep:keep:$route:RX" )
  echo "$(date -u +%FT%TZ) $route boots rc=$?" >> "$R/chain.log"
done
( export BIN=$WT/target/day41b/tip/memra-server PREV_BIN=$WT/target/day41b/offprev/memra-server
  ask day41-run.sh "$R" offprev:offprev:plain:RX6 )
python3 "$D/day41-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"; log "day 41b done"

# 7. DAY43 (rtx5090-day43/chain.sh's steps).
R=$D/rtx5090-day43; mkdir -p "$R"
[ -s target/day43/SHA256SUMS ] || { build "$WT/target/day43" 87d9e00d16ea4436c48d561d99926c0d1677c0dd tip offprev:day43-nodoor.patch \
  > "$R/build.out" 2>&1; log "day 43 build rc=$?"; }
cp target/day43/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256") (queue-m)" >> "$R/chain.log"
( export BIN=$WT/target/day43/tip/memra-server PREV_BIN=$WT/target/day43/offprev/memra-server
  ask day43-run.sh "$R" rx-spec-O1-unset:unset:spec:RX rx-spec-O1-clamp:clamp:spec:RX rx-spec-O2-clamp:clamp:spec:RX \
    rx-spec-O2-unset:unset:spec:RX offprev:offprev:spec:RX )
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$R/chain.log"
python3 "$D/day43-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"; log "day 43 done"

# 8. DAY44 (rtx5090-day44/chain.sh's steps).
R=$D/rtx5090-day44; mkdir -p "$R"
cp target/day44/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256") (queue-m)" >> "$R/chain.log"
for route in plain spec; do
  for shape in rx rxg; do
    S=$( [ "$shape" = rx ] && echo RX || echo RXg )
    ( export BIN=$WT/target/day44/tip/memra-server PREV_BIN=$WT/target/day44/offprev/memra-server
      ask day44-run.sh "$R" "rx-$route-$shape-O1-keep:keep:$route:$S" "rx-$route-$shape-O1-exact:exact:$route:$S" \
        "rx-$route-$shape-O2-exact:exact:$route:$S" "rx-$route-$shape-O2-keep:keep:$route:$S" )
    echo "$(date -u +%FT%TZ) $route $shape boots rc=$?" >> "$R/chain.log"
  done
done
( export BIN=$WT/target/day44/tip/memra-server PREV_BIN=$WT/target/day44/offprev/memra-server
  ask day44-run.sh "$R" offprev:offprev:plain:RX6 fault-plain-rxg6:fault:plain:RXg6 )
python3 "$D/day44-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"; log "day 44 done"

# 9. DAY45 (rtx5090-day45/chain.sh's steps, addendum B's cell).
R=$D/rtx5090-day45; mkdir -p "$R"
cp target/day45/SHA256SUMS "$R/binaries.sha256"
echo "$(date -u +%FT%TZ) chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256") (queue-m)" >> "$R/chain.log"
( export BIN=$WT/target/day45/tip/memra-server CLIENT_EXTRA="--burst 32 --length 6144 --max-tokens 64 --wave2-delay-s 10" DOOR_MEMORY=1
  ask day45-run.sh "$R" O1-off:off O1-on:on O2-on:on O2-off:off )
echo "$(date -u +%FT%TZ) boots rc=$?" >> "$R/chain.log"
python3 "$D/day45-read.py" rtx5090 "$R" > "$R/read.log" 2>&1
echo "$(date -u +%FT%TZ) chain done" >> "$R/chain.log"; log "day 45 done"
log "queue-m done"
