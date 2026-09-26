#!/usr/bin/env bash
# WP-A day 61 (DAY61.md design W) on the local RTX 5090 Laptop GPU, fixed before it runs. ONE bounded hold of
# /tmp/memra-5090.lock (60 x 120 s behind the other lanes; then no compute app and >= 20000 MiB free, bounded 15 x 60
# s; another project's process that lands on the card is waited out, never touched). Under the hold: (a) the host and
# pinned identity cells on w (green) and red (must fail), the census; the identity gate default and plain door OFF
# and ON and the contract fault gate default and plain on w (--external-lock 9); then (b), (d): base against w by
# demote and promote, o1 = base-demote w-demote base-promote w-promote x5, o2 the reverse x5 (40 boots), door ON,
# MEMRA_MAX_SESSIONS=4, MEMRA_PREFIX_CACHE_MB=256, MEMRA_KV_HOST_MB=8192, MEMRA_SERVE_SPEC=0; each boot's start
# temperature, SM clock and host load; 250 ms telemetry; then w-reading.py --card 5090. Executed-not-qualified.
# usage: card-run.sh <out_root> <model.gguf>   (the binaries from build-local.sh under <out_root>/bins)
set -uo pipefail
ROOT=$1; MODEL=$2
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
mkdir -p "$ROOT/tier/ab" "$ROOT/unit" "$ROOT/gates"
grep -q "^rc=0$" "$ROOT/build.log" 2>/dev/null || { echo "build receipt missing"; exit 2; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
log "model sha256 (outside the hold): $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 60); do
  if flock -n 9; then held=1; break; fi
  log "lock busy, attempt $attempt of 60, waiting 120 s"; sleep 120
done
[ "$held" = 1 ] || { log "NOT RUN: the 5090 lock stayed busy"; exit 2; }
free_ok=0
for attempt in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 | head -1)
  if [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] 2>/dev/null; then free_ok=1; break; fi
  log "card not idle under the hold (apps=[${apps//$'\n'/; }] free=${free} MiB), attempt $attempt of 15"; sleep 60
done
[ "$free_ok" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
log "hold taken"
U=$ROOT/unit
cell() { # name exe filter
  CUDA_VISIBLE_DEVICES=0 "$2" --include-ignored --exact --nocapture --test-threads=1 "$3" > "$U/$1.log" 2>&1
  local rc=$?; grep -q '^running 1 test$' "$U/$1.log" || { echo "RAN NO TEST: the filter matched nothing" >> "$U/$1.log"; rc=97; }
  echo "$rc" > "$U/$1.exit"; log "$1 rc=$rc $(grep -h '^test result' "$U/$1.log")"
}
T=tier_transfer::tests::day61_the_streamed_checksum_is_the_checksum
cell a1-green "$ROOT/bins/w/memra-engine-tests" $T
cell a2-green "$ROOT/bins/w/memra-engine-tests" ${T}_on_pinned_memory
cell a1-red "$ROOT/bins/red/memra-engine-tests" $T
cell a2-red "$ROOT/bins/red/memra-engine-tests" ${T}_on_pinned_memory
"$ROOT/bins/w/memra-engine-tests" day61_the_pinned_views native_cells_own > "$U/censuses.log" 2>&1; echo "$?" > "$U/censuses.exit"
g1=$(cat "$U/a1-green.exit"); g2=$(cat "$U/a2-green.exit"); r1=$(cat "$U/a1-red.exit"); r2=$(cat "$U/a2-red.exit")
m1=$(grep -c 'day61 red arm' "$U/a1-red.log"); m2=$(grep -c 'day61 red arm' "$U/a2-red.log")
echo "UNIT a1-green=$g1 a2-green=$g2 a1-red=$r1 (marker $m1) a2-red=$r2 (marker $m2) censuses=$(cat "$U/censuses.exit")" | tee -a "$U/run.log"
BIN=$ROOT/bins/w/memra-server
gate() { local name=$1; shift; "$@" > "$ROOT/gates/$name.log" 2>&1; local rc=$?; echo "$rc" > "$ROOT/gates/$name.exit"; log "$name rc=$rc"; }
C=MEMRA_HOSTGATE_CACHE_MB=256
gate identity-default-off env $C tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/identity-default-off"
gate identity-default-on  env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/identity-default-on"
gate identity-plain-off   env $C MEMRA_SERVE_SPEC=0 tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/identity-plain-off"
gate identity-plain-on    env $C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/identity-plain-on"
gate contract-fault       env $C tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/contract-fault"
gate contract-fault-plain env $C MEMRA_SERVE_SPEC=0 tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN" "$ROOT/gates/contract-fault-plain"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 250 > "$ROOT/card-250ms.csv" 2>&1 &
SAMPLER=$!
PORT=${MEMRA_GATE_PORT:-18161}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 log $2 bin
  memra_port_guard w-5090 "$PORT" MEMRA_GATE_PORT || return 1
  env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 \
    MEMRA_KV_HOST_CONTRACTS=1 "$2" > "$1" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
stop() {
  [[ -n $SERVER_PID ]] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
finish() { stop; kill "$SAMPLER" 2>/dev/null || true; }
trap finish EXIT
i=0
for order in o1 o2; do
  seq1="base:demote w:demote base:promote w:promote"; [ $order = o2 ] && seq1="w:promote base:promote w:demote base:demote"
  for am in $seq1 $seq1 $seq1 $seq1 $seq1; do
    arm=${am%%:*}; mode=${am#*:}; B=$ROOT/bins/$arm/memra-server
    i=$((i+1)); D=$ROOT/tier/ab/$order/$(printf 'b%02d-%s-%s' "$i" "$arm" "$mode"); mkdir -p "$D"
    echo "arm=$arm mode=$mode order=$order bin=$(sha256sum "$B" | cut -c1-16)" > "$D/BOOT.txt"
    echo "start temperature.gpu,clocks.sm,power.draw: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>&1)" >> "$D/BOOT.txt"
    echo "host load at start: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$D/BOOT.txt"
    echo "compute apps at start: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')" >> "$D/BOOT.txt"
    if ! boot "$D/server.log" "$B"; then log "$order $arm $mode boot NOT READY"; echo boot-failed >> "$D/BOOT.txt"; stop; continue; fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode "$mode" --n 5 --server-log "$D/server.log" \
      --out "$D/$mode" --tag "w-5090-$arm-$mode" > "$D/$mode.log" 2>&1
    log "$order $arm $mode rc=$? $(grep -h 'STALL rule' "$D/$mode.log" | cut -c1-110)"
    stop
    { echo "== $order/$(basename "$D")"; python3 research/spill-a-20260919/stall_cell.py --replay "$D/$mode/receipt.json"; } \
      >> "$ROOT/tier/ab/replays.log" 2>&1
  done
done
trap - EXIT
kill "$SAMPLER" 2>/dev/null || true
flock -u 9
python3 research/spill-a-20260919/w-reading.py --card 5090 "$ROOT" > "$ROOT/reading-w.log" 2>&1
log "done; reading rc=$? $(tail -1 "$ROOT/reading-w.log")"
