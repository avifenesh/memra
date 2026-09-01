#!/usr/bin/env bash
# decode-diet scoped serve (window: decode-diet agent, cards 0/1/2, port 18400).
# Derived from flip-battery-20260830/box/serve.sh — the serving env is byte-identical
# (comparability requirement: plain 35.41 tok/s is the banked baseline on this env).
# NOTE (receipt): MEMRA_HYPER_BATCH is DEFAULT ON at this head (28cbc1af6, hbatch-battery
# flip 2026-08-31); we leave it at default in BOTH arms — c=1 rows unaffected per the
# ladder receipts (B=1 cost -0.30%).
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 before
# any signal. NEVER pkill, NEVER basename matching (gate-stop-pkill-basename-trap law).
# Arms differ ONLY in the source flags passed as "$@" (comparability requirement).
set -uo pipefail
OUT=/root/out-diet
PIDFILE=$OUT/server.pid
BIN=/root/memra-diet/target/release/memra-server
DFLASH_SHA8=b33c0347
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[dd] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[dd] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[dd] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[dd] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[dd] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[dd] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[dd] stopped"
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
    "$@" DIETBAT_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[dd] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[dd] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[dd] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[dd] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[dd] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "DIETBAT_NONCE=$nonce" \
    || { echo "[dd] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[dd] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local spec_env="" dfl_env=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1) spec_env=1;;
    MEMRA_GLM5_DFLASH=*) dfl_env=1;;
  esac; done
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[dd] GATE FAIL: no ARMED line"; fail=1; }
    grep -q "draft source = dflash2 @ $DFLASH_SHA8" "$log" \
      || { echo "[dd] GATE FAIL: no 'draft source = dflash2 @ $DFLASH_SHA8' line"; fail=1; }
    grep -q "native MTP head NOT loaded" "$log" \
      || { echo "[dd] GATE FAIL: head not reported NOT loaded"; fail=1; }
  else
    [ "$nspec" -eq 0 ] || { echo "[dd] GATE FAIL: non-spec arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  {
    echo "boot=$name pid=$pid nonce=$nonce"
    echo "resident_lines=$nres glm5spec_lines=$nspec spec_env=${spec_env:-0} dflash_env=${dfl_env:-0}"
    grep -m3 "RESIDENT" "$log"
    grep -m3 "\[glm5-spec\]" "$log" || true
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram" 2>/dev/null
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[dd] GATES GREEN boot=$name" || echo "[dd] GATES RED boot=$name"
  return "$fail"
}

# doors NAME [ENV=VAL ...] — the DOOR ENGAGEMENT gate, run AFTER the first request
# (announces print at first engagement, not at boot). Expected set derived from the
# same extras the boot got: every door env present => its announce DEMANDED; every door
# env absent => its announce FORBIDDEN (zero lines). On the serving recipe
# (MEMRA_BF16_MMV=1) the kda6 door must take the bf16 arm: demand 'engaged arm=bf16',
# forbid the q8 form 'engaged in_f=' (fails-closed evidence). Spec boots additionally
# demand the BATCHED verify-walk line (MEMRA_GLM5_VERIFY_BATCH default ON at this head)
# and forbid the PER-ROW line.
doors() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log" fail=0
  local e_pre="" e_ws="" e_kda="" e_mla="" spec_env="" vrest_phase=""
  for a in "$@"; do case "$a" in
    MEMRA_HC_FUSED_PRE=1) e_pre=1;;
    MEMRA_HC_DECODE_WS=1) e_ws=1;;
    MEMRA_KDA_FUSED_PROJ=1) e_kda=1;;
    MEMRA_MLA_DECODE_SPLIT=1) e_mla=1;;
    MEMRA_GLM5_SPEC=1) spec_env=1;;
    DIET_PHASE=vrest) vrest_phase=1;;
  esac; done
  chk() { # expected(1/""), pattern, label
    local n; n=$(grep -c -- "$2" "$log")
    if [ -n "$1" ]; then
      [ "$n" -ge 1 ] || { echo "[dd] DOOR FAIL: expected '$3' announce, lines=$n"; fail=1; }
    else
      [ "$n" -eq 0 ] || { echo "[dd] DOOR FAIL: forbidden '$3' announce, lines=$n"; fail=1; }
    fi
    echo "door=$3 expected=${1:-0} lines=$n"
  }
  {
    chk "$e_pre" "\[hc-fused-pre\] engaged" "hc-fused-pre"
    if [ -n "$spec_env" ] && [ -n "$e_ws" ]; then
      # SPEC boots decode through the t=K+1 verify walk, never the t=1
      # hyper_range_decode walk that owns this door — the announce is structurally
      # absent (window finding, receipted 2026-08-31): record, do not demand.
      echo "door=hc-decode-ws expected=optional-on-spec lines=$(grep -c -- '\[hc-decode-ws\] engaged' "$log")"
    else
      chk "$e_ws" "\[hc-decode-ws\] engaged" "hc-decode-ws"
    fi
    chk "$e_kda" "\[kda-fused6\] engaged arm=bf16" "kda-fused6-bf16"
    chk ""       "\[kda-fused6\] engaged in_f=" "kda-fused6-q8(forbidden-on-this-recipe)"
    chk "$e_mla" "\[mla-decode-split\] engaged" "mla-decode-split"
    if [ -n "$spec_env" ]; then
      chk 1  "verify walk BATCHED per layer" "verify-walk-batched"
      chk "" "verify walk PER-ROW" "verify-walk-per-row"
      if [ -n "$vrest_phase" ]; then
        # vrest head (a3fc59aaf): the batched-walk line ends "moe=pairs rows-call where
        # qualified" and the first spec round announces the pairs dispatch (carry list,
        # vrest-20260831/LANE.md above §5).
        chk 1 "moe=pairs rows-call" "verify-walk-moe-pairs"
        chk 1 "\[glm5-vrows\] verify MoE batched across rows" "glm5-vrows"
      fi
    fi
    grep -m1 "\[hc-fused-pre\] engaged" "$log" || true
    grep -m1 "\[hc-decode-ws\] engaged" "$log" || true
    grep -m1 "\[kda-fused6\] engaged arm=bf16" "$log" || true
    grep -m1 "\[mla-decode-split\] engaged" "$log" || true
  } > "$OUT/logs/boot-$name.doors" 2>&1
  cat "$OUT/logs/boot-$name.doors"
  [ "$fail" -eq 0 ] && echo "[dd] DOORS GREEN boot=$name" || echo "[dd] DOORS RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  doors) shift; doors "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh doors <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
