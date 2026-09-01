#!/usr/bin/env bash
# tp2-battery served PP-3 CALIBRATION boot (instrument-offset receipt vs the 35.41 baseline).
# Derived from flip-battery serve.sh: same recipe env, same nonce/RESIDENT gates, plain arm
# only (no spec doors in this window; TP is engine-level — the worker refuses MEMRA_GLM5_TP
# by design and this script never sets it).
# stop() is pidfile-scoped: /proc exe + MEMRA_ADDR check, never pkill (basename-trap law).
set -uo pipefail
OUT=/root/out-tp2
PIDFILE=$OUT/server.pid
BIN=/root/memra-tp2/target/release/memra-server
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[tp2b] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[tp2b] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[tp2b] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[tp2b] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[tp2b] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[tp2b] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[tp2b] stopped"
}

start() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log"
  local nonce; nonce=$(cat /proc/sys/kernel/random/uuid)
  : > "$log"
  local unsets=()
  while IFS='=' read -r k _; do case "$k" in MEMRA_*) unsets+=(-u "$k");; esac; done < <(env)
  env "${unsets[@]}" \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 \
    MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 \
    MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 \
    MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=600000 \
    "$@" TP2B_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[tp2b] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[tp2b] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      local fail=0
      tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "TP2B_NONCE=$nonce" \
        || { echo "[tp2b] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
      local nres; nres=$(grep -c "RESIDENT" "$log")
      [ "$nres" -ge 3 ] || { echo "[tp2b] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
      grep -q "\[glm5-spec\]" "$log" && { echo "[tp2b] GATE FAIL: plain arm has [glm5-spec] lines"; fail=1; }
      grep -q "\[glm5-tp" "$log" && { echo "[tp2b] GATE FAIL: served boot has TP lines (must be impossible)"; fail=1; }
      [ "$fail" -eq 0 ] && echo "[tp2b] GATES GREEN boot=$name" || echo "[tp2b] GATES RED boot=$name"
      return "$fail"
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[tp2b] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[tp2b] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[tp2b] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
