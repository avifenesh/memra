#!/usr/bin/env bash
# flip-reprice scoped serve (window: flip-reprice agent, cards 0/1/2, port 18400).
# Third run of the flip battery, now against lane/glm5-verify-batch @ c62677352 (the
# batched verify walk, MEMRA_GLM5_VERIFY_BATCH default ON). Derived byte-for-byte from
# flip-battery-20260830/box/serve.sh: the pinned 3-card recipe is UNCHANGED
# (MEMRA_BF16_MMV=1 is load-bearing — it puts KDA projections + lm head on the tcols
# class, LANE.md section 9). Own build clone /root/memra-vb; own out namespace.
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 before
# any signal. NEVER pkill, NEVER basename matching (gate-stop-pkill-basename-trap law).
set -uo pipefail
OUT=/root/out-flip3
PIDFILE=$OUT/server.pid
BIN=/root/memra-vb/target/release/memra-server
DFLASH_SHA8=b33c0347
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[frp] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[frp] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[frp] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[frp] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[frp] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[frp] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[frp] stopped"
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
    MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=600000 \
    "$@" FLIPRP_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[frp] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[frp] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[frp] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[frp] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[frp] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  # arm identity: the LISTENING process carries our nonce (A/B arm-identity law:
  # health-200 proves a listener, never WHICH server)
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "FLIPRP_NONCE=$nonce" \
    || { echo "[frp] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[frp] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local spec_env="" mtp_env="" dfl_env="" vb0_env=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1) spec_env=1;;
    MEMRA_GLM5_MTP=1) mtp_env=1;;
    MEMRA_GLM5_DFLASH=*) dfl_env=1;;
    MEMRA_GLM5_VERIFY_BATCH=0) vb0_env=1;;
  esac; done
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[frp] GATE FAIL: no ARMED line"; fail=1; }
    if [ -n "$dfl_env" ]; then
      grep -q "draft source = dflash2 @ $DFLASH_SHA8" "$log" \
        || { echo "[frp] GATE FAIL: no 'draft source = dflash2 @ $DFLASH_SHA8' line"; fail=1; }
      if [ -z "$mtp_env" ]; then
        grep -q "native MTP head NOT loaded" "$log" \
          || { echo "[frp] GATE FAIL: MTP flag unset but head not reported NOT loaded"; fail=1; }
      fi
    else
      grep -q "\[glm5-spec\] draft source = native-mtp" "$log" \
        || { echo "[frp] GATE FAIL: no native-mtp source line"; fail=1; }
      grep -q "MTP head loaded" "$log" \
        || { echo "[frp] GATE FAIL: no 'MTP head loaded' line"; fail=1; }
    fi
  else
    [ "$nspec" -eq 0 ] || { echo "[frp] GATE FAIL: PLAIN arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  {
    echo "boot=$name pid=$pid nonce=$nonce"
    echo "resident_lines=$nres glm5spec_lines=$nspec spec_env=${spec_env:-0} mtp_env=${mtp_env:-0} dflash_env=${dfl_env:-0} verify_batch_0=${vb0_env:-0}"
    grep -m3 "RESIDENT" "$log"
    grep -m3 "\[glm5-spec\]" "$log" || true
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram" 2>/dev/null
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[frp] GATES GREEN boot=$name" || echo "[frp] GATES RED boot=$name"
  return "$fail"
}

# walk_receipt NAME EXPECT(batched|perrow): the verify-walk engagement receipt. The line
# prints ONCE PER PROCESS at the FIRST verify walk (not at boot), so this is checked
# AFTER the first spec request of a boot. On batched-ON spec arms the BATCHED line must
# exist and the PER-ROW line must NOT; on the =0 rollback arm the reverse.
walk_receipt() {
  local name="$1" expect="$2"
  local log="$OUT/logs/boot-$name.log" fail=0
  local nb np
  nb=$(grep -c "verify walk BATCHED per layer" "$log")
  np=$(grep -c "verify walk PER-ROW" "$log")
  case "$expect" in
    batched)
      [ "$nb" -ge 1 ] || { echo "[frp] WALK RECEIPT FAIL boot=$name: no BATCHED line"; fail=1; }
      [ "$np" -eq 0 ] || { echo "[frp] WALK RECEIPT FAIL boot=$name: PER-ROW line on a batched arm"; fail=1; }
      ;;
    perrow)
      [ "$np" -ge 1 ] || { echo "[frp] WALK RECEIPT FAIL boot=$name: no PER-ROW line"; fail=1; }
      [ "$nb" -eq 0 ] || { echo "[frp] WALK RECEIPT FAIL boot=$name: BATCHED line on the =0 arm"; fail=1; }
      ;;
  esac
  echo "walk_receipt boot=$name expect=$expect batched_lines=$nb perrow_lines=$np fail=$fail" \
    >> "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[frp] WALK RECEIPT GREEN boot=$name ($expect)"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  stop) stop ;;
  walk) shift; walk_receipt "$@" ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh stop | serve.sh walk <name> batched|perrow"; exit 2 ;;
esac
