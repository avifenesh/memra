#!/usr/bin/env bash
# hbatch-battery scoped serve (window: hbatch-battery agent, cards 0/1/2, port 18400).
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 before any
# signal. NEVER pkill, NEVER basename matching (gate-stop-pkill-basename-trap law).
# Arms differ ONLY in the extra flags passed as "$@" (arm-condition consistency):
#   OFF arm: no extras (MEMRA_HYPER_BATCH unset = today's default)
#   ON  arm: MEMRA_HYPER_BATCH=1
# MEMRA_MAX_SESSIONS=16 on BOTH arms (covers the c<=12 ladder; deliberate deviation from the
# 3way window's 4 — receipted in the boot identity file).
set -uo pipefail
OUT=/root/out-hbatch
PIDFILE=$OUT/server.pid
BIN=/root/memra/target/release/memra-server
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[hb] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[hb] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[hb] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[hb] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[hb] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[hb] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[hb] stopped"
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
    MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 \
    MEMRA_MAX_SESSIONS=16 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=600000 \
    "$@" HBATCH_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$* max_sessions=16" > "$OUT/logs/boot-$name.identity"
  echo "[hb] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[hb] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[hb] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[hb] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[hb] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  # arm identity: the LISTENING process carries our nonce (A/B arm-identity law:
  # health-200 proves a listener, never WHICH server)
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "HBATCH_NONCE=$nonce" \
    || { echo "[hb] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[hb] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local hb_env=""
  for a in "$@"; do case "$a" in MEMRA_HYPER_BATCH=1) hb_env=1;; esac; done
  # THE BATCH WALK'S ENGAGEMENT LINE (worker carve-out, worker.rs):
  #   ON  = "[worker] <model>: BATCHED DECODE (mHC hyper arm, opt-in via MEMRA_HYPER_BATCH=1..."
  #   OFF = "[worker] <model>: EAGER-ONLY serving (hyper-connections residual — no batched decode arm)..."
  local n_on; n_on=$(grep -c "BATCHED DECODE (mHC hyper arm" "$log")
  local n_off; n_off=$(grep -c "EAGER-ONLY serving" "$log")
  if [ -n "$hb_env" ]; then
    [ "$n_on" -ge 1 ] || { echo "[hb] GATE FAIL: ON arm has no 'BATCHED DECODE (mHC hyper arm' line"; fail=1; }
    [ "$n_off" -eq 0 ] || { echo "[hb] GATE FAIL: ON arm has $n_off EAGER-ONLY lines"; fail=1; }
  else
    [ "$n_off" -ge 1 ] || { echo "[hb] GATE FAIL: OFF arm has no EAGER-ONLY line"; fail=1; }
    [ "$n_on" -eq 0 ] || { echo "[hb] GATE FAIL: OFF arm has $n_on 'BATCHED DECODE (mHC hyper arm' lines"; fail=1; }
  fi
  # No spec lines on either arm (plain serving; spec flags default OFF)
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  [ "$nspec" -eq 0 ] || { echo "[hb] GATE FAIL: $nspec [glm5-spec] lines on a plain arm"; fail=1; }
  {
    echo "boot=$name pid=$pid nonce=$nonce hyper_batch_env=${hb_env:-0}"
    echo "resident_lines=$nres batched_on_lines=$n_on eager_only_lines=$n_off glm5spec_lines=$nspec"
    grep -m3 "RESIDENT" "$log"
    grep -m2 "BATCHED DECODE (mHC hyper arm" "$log" || true
    grep -m2 "EAGER-ONLY serving" "$log" || true
    grep -m2 "decode wave cap" "$log" || true
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram" 2>/dev/null
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[hb] GATES GREEN boot=$name" || echo "[hb] GATES RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
