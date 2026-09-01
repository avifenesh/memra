#!/usr/bin/env bash
# struct-battery scoped serve (window: struct-battery agent, cards 0/1/2, port 18400).
# Derived from mv-battery-20260831/box/serve.sh — the serving env is byte-identical
# (comparability requirement: the 70.458 tok/s winner is the banked baseline on this env;
# this build c7d936536 = ep-place bringup head + the moe-loc merge, doors T/X/K/W now
# DEFAULT ON, doors D/H + instrument S default OFF).
# NOTE (receipt): MEMRA_HYPER_BATCH and MEMRA_GLM5_VERIFY_BATCH are DEFAULT ON at this
# head; both left at default in EVERY arm. Doors T/X/K/W are DEFAULT ON at this head: an
# OFF arm is spelled ONLY by pinning =0 (owner law; unset is an ON arm).
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 before
# any signal. NEVER pkill, NEVER basename matching (gate-stop-pkill-basename-trap law).
# Arms differ ONLY in the extras passed as "$@" (comparability requirement).
set -uo pipefail
OUT=/root/out-struct
PIDFILE=$OUT/server.pid
BIN=/root/memra-struct/target/release/memra-server
DFLASH_SHA8=b33c0347
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
    MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 \
    MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=600000 \
    "$@" SBAT_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[sb] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[sb] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
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
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "SBAT_NONCE=$nonce" \
    || { echo "[sb] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[sb] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local spec_env=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1) spec_env=1;;
  esac; done
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[sb] GATE FAIL: no ARMED line"; fail=1; }
    grep -q "draft source = dflash2 @ $DFLASH_SHA8" "$log" \
      || { echo "[sb] GATE FAIL: no 'draft source = dflash2 @ $DFLASH_SHA8' line"; fail=1; }
    grep -q "native MTP head NOT loaded" "$log" \
      || { echo "[sb] GATE FAIL: head not reported NOT loaded"; fail=1; }
  else
    [ "$nspec" -eq 0 ] || { echo "[sb] GATE FAIL: non-spec arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  {
    echo "boot=$name pid=$pid nonce=$nonce"
    echo "resident_lines=$nres glm5spec_lines=$nspec spec_env=${spec_env:-0}"
    grep -m3 "RESIDENT" "$log"
    grep -m3 "\[glm5-spec\]" "$log" || true
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram" 2>/dev/null
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[sb] GATES GREEN boot=$name" || echo "[sb] GATES RED boot=$name"
  return "$fail"
}

# engage NAME [ENV=VAL ...] — the DOOR/INSTRUMENT ENGAGEMENT gate, run AFTER the first
# request (announces print at first engagement). Expected set derived from the extras the
# boot got, WITH THE DEFAULT-ON DOORS INVERTED vs the mv-battery gate: doors T/X/K/W read
# !=Ok("0") at this head, so on a SPEC boot their announces are DEMANDED unless the arm
# pins =0 (owner law: unset is an ON arm). Door D announces only when =1 (default OFF);
# instrument S emits [moe-vrows-dedup] only when =1 AND door D is =0/unset. Door H has NO
# announce (counter-anchored on the rig, moe-loc LANE §4.1) — its arm identity is the
# boot env receipt. A trace-armed boot (MEMRA_MOE_TRACE/WEIGHT_TRACE) fail-closes door D
# by design, so D's announce is FORBIDDEN there even if the arm set =1 (never done here).
engage() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log" fail=0
  local spec_env="" d_on="" dedup_on="" t_off="" x_off="" k_off="" w_off="" traced=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1)              spec_env=1;;
    MEMRA_MOE_VROWS_DEV_TABLES=1)   d_on=1;;
    MEMRA_MOE_VROWS_DEDUP_STAT=1)   dedup_on=1;;
    MEMRA_BF16_TCOLS_WIDE=0)        t_off=1;;
    MEMRA_BF16_TCOLS_X1=0)          x_off=1;;
    MEMRA_TOPK_SHARDS=0)            k_off=1;;
    MEMRA_GLM5_VERIFY_WS=0)         w_off=1;;
    MEMRA_MOE_TRACE=*|MEMRA_MOE_WEIGHT_TRACE=*) traced=1;;
  esac; done
  [ -n "$traced" ] && d_on=""   # fail-closed conjunct: a trace disarms door D
  chk() { # expected(1/""), pattern, label
    local n; n=$(grep -c -- "$2" "$log")
    if [ -n "$1" ]; then
      [ "$n" -ge 1 ] || { echo "[sb] ENGAGE FAIL: expected '$3' announce, lines=$n"; fail=1; }
    else
      [ "$n" -eq 0 ] || { echo "[sb] ENGAGE FAIL: forbidden '$3' announce, lines=$n"; fail=1; }
    fi
    echo "engage=$3 expected=${1:-0} lines=$n"
  }
  local e_t=1 e_x=1 e_k=1 e_w=1
  [ -n "$t_off" ] && e_t=""; [ -n "$x_off" ] && e_x=""
  [ -n "$k_off" ] && e_k=""; [ -n "$w_off" ] && e_w=""
  if [ -z "$spec_env" ]; then
    # non-spec boot: every spec-scoped announce is structurally absent — forbid all
    e_t=""; e_x=""; e_k=""; e_w=""; d_on=""
  fi
  {
    chk "$e_t" "\[bf16-tcols-wide\] engaged" "bf16-tcols-wide(T,default-on)"
    chk "$e_x" "\[bf16-tcols-x1\] engaged" "bf16-tcols-x1(X,default-on)"
    chk ""     "\[moe-vrows-pack\] engaged" "moe-vrows-pack(M,off-refuted)"
    chk "$e_k" "\[topk-shards\] engaged" "topk-shards(K,default-on)"
    chk "$e_w" "\[glm5-verify-ws\] engaged" "glm5-verify-ws(W,default-on)"
    chk "$d_on" "\[moe-vrows-dev-tables\] engaged" "moe-vrows-dev-tables(D)"
    chk "$dedup_on" "\[moe-vrows-dedup\]" "moe-vrows-dedup(S)"
    if [ -n "$spec_env" ]; then
      chk 1  "verify walk BATCHED per layer" "verify-walk-batched"
      chk "" "verify walk PER-ROW" "verify-walk-per-row"
      chk 1 "\[glm5-vrows\] verify MoE batched across rows" "glm5-vrows"
    fi
    grep -m1 "\[moe-vrows-dev-tables\] engaged" "$log" || true
    grep -m1 "\[moe-vrows-dedup\]" "$log" || true
  } > "$OUT/logs/boot-$name.engage" 2>&1
  cat "$OUT/logs/boot-$name.engage"
  [ "$fail" -eq 0 ] && echo "[sb] ENGAGE GREEN boot=$name" || echo "[sb] ENGAGE RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  engage) shift; engage "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh engage <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
