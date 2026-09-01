#!/usr/bin/env bash
# spec-battery scoped serve (window: spec-battery agent, cards 0/1/2, port 18400).
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 AND our
# nonce before any signal. NEVER touches the card-3 co-tenant (memra-server-card3, 18500).
set -uo pipefail
OUT=/root/out-specbat
PIDFILE=$OUT/server.pid
BIN=/root/memra/target/release/memra-server
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[sb] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[sb] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[sb] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[sb] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[sb] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[sb] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[sb] stopped"
}

start() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log"
  local nonce; nonce=$(cat /proc/sys/kernel/random/uuid)
  : > "$log"
  # scrub any inherited MEMRA_* then apply the pinned recipe + arm extras ("$@")
  local unsets=()
  while IFS='=' read -r k _; do case "$k" in MEMRA_*) unsets+=(-u "$k");; esac; done < <(env)
  env "${unsets[@]}" \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 \
    MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 \
    MEMRA_MLA_TC_PREFILL=1 \
    MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 \
    MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=600000 \
    "$@" SPECBAT_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[sb] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      echo "[sb] READY after $((SECONDS-t0))s"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[sb] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[sb] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[sb] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  # arm identity: the LISTENING process carries our nonce (A/B arm-identity law)
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "SPECBAT_NONCE=$nonce" \
    || { echo "[sb] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[sb] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local spec_env=""; for a in "$@"; do case "$a" in MEMRA_GLM5_SPEC=1) spec_env=1;; esac; done
  local trim_env=""; for a in "$@"; do case "$a" in MEMRA_FRSPEC_TRIM=*) trim_env=1;; esac; done
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[sb] GATE FAIL: no ARMED line"; fail=1; }
    if [ -n "$trim_env" ]; then
      grep -q "draft head TRIMMED to" "$log" || { echo "[sb] GATE FAIL: trim set but no TRIMMED line"; fail=1; }
    else
      grep -q "draft head FULL target vocab" "$log" || { echo "[sb] GATE FAIL: no FULL-vocab line"; fail=1; }
    fi
  else
    [ "$nspec" -eq 0 ] || { echo "[sb] GATE FAIL: OFF arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  {
    echo "boot=$name pid=$pid nonce=$nonce"
    echo "resident_lines=$nres glm5spec_lines=$nspec spec_env=${spec_env:-0} trim_env=${trim_env:-0}"
    grep -m3 "RESIDENT" "$log"
    grep -m2 "\[glm5-spec\]" "$log" || true
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[sb] GATES GREEN boot=$name" || echo "[sb] GATES RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
