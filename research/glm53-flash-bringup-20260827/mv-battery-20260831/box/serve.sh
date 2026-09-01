#!/usr/bin/env bash
# mv-doors scoped serve (window: mv-doors agent, cards 0/1/2, port 18400).
# Derived from diet-battery-20260831/box/serve.sh — the serving env is byte-identical
# (comparability requirement: SHIP 62.43 tok/s is the banked baseline on this env, vrest
# head; this build 146b13c33 = consol-db + the five matvec doors, all default OFF).
# NOTE (receipt): MEMRA_HYPER_BATCH and MEMRA_GLM5_VERIFY_BATCH are DEFAULT ON at this
# head; both left at default in EVERY arm (c=1 rows unaffected per the hbatch ladder;
# the BATCHED walk is the ship program the doors were priced against).
# stop() is pidfile-scoped: verifies /proc/pid/exe AND MEMRA_ADDR=127.0.0.1:18400 before
# any signal. NEVER pkill, NEVER basename matching (gate-stop-pkill-basename-trap law).
# Arms differ ONLY in the extras passed as "$@" (comparability requirement).
set -uo pipefail
OUT=/root/out-mv
PIDFILE=$OUT/server.pid
BIN=/root/memra-mv/target/release/memra-server
DFLASH_SHA8=b33c0347
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[mv] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[mv] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[mv] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=127.0.0.1:18400" \
    || { echo "[mv] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[mv] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[mv] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 3; fi
  rm -f "$PIDFILE"
  echo "[mv] stopped"
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
    "$@" MVBAT_NONCE="$nonce" MEMRA_ADDR=127.0.0.1:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "nonce=$nonce pid=$pid boot=$name extras=$*" > "$OUT/logs/boot-$name.identity"
  echo "[mv] launched pid=$pid boot=$name nonce=$nonce"
  local t0=$SECONDS
  for _ in $(seq 1 600); do
    if curl -s -m 2 http://127.0.0.1:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[mv] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[mv] BOOT DIED"; tail -15 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[mv] BOOT FAILED"; tail -15 "$log"; return 1; }
    sleep 2
  done
  echo "[mv] NOT READY after $((SECONDS-t0))s"; tail -20 "$log"; return 1
}

gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "MVBAT_NONCE=$nonce" \
    || { echo "[mv] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  local nres; nres=$(grep -c "RESIDENT" "$log")
  [ "$nres" -ge 3 ] || { echo "[mv] GATE FAIL: RESIDENT lines=$nres (<3)"; fail=1; }
  local spec_env=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1) spec_env=1;;
  esac; done
  local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[mv] GATE FAIL: no ARMED line"; fail=1; }
    grep -q "draft source = dflash2 @ $DFLASH_SHA8" "$log" \
      || { echo "[mv] GATE FAIL: no 'draft source = dflash2 @ $DFLASH_SHA8' line"; fail=1; }
    grep -q "native MTP head NOT loaded" "$log" \
      || { echo "[mv] GATE FAIL: head not reported NOT loaded"; fail=1; }
  else
    [ "$nspec" -eq 0 ] || { echo "[mv] GATE FAIL: non-spec arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  {
    echo "boot=$name pid=$pid nonce=$nonce"
    echo "resident_lines=$nres glm5spec_lines=$nspec spec_env=${spec_env:-0}"
    grep -m3 "RESIDENT" "$log"
    grep -m3 "\[glm5-spec\]" "$log" || true
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram" 2>/dev/null
  } > "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[mv] GATES GREEN boot=$name" || echo "[mv] GATES RED boot=$name"
  return "$fail"
}

# doors NAME [ENV=VAL ...] — the DOOR ENGAGEMENT gate, run AFTER the first request
# (announces print at first engagement, not at boot). Expected set derived from the same
# extras the boot got: every matvec-door env present => its announce DEMANDED; absent =>
# FORBIDDEN (zero lines). Engagement scope (LANE.md/FLAGS.md, stated): ALL FIVE doors
# engage only on SPEC boots on this recipe — T needs the drafter block head (t=15), X the
# verify-walk tcols route (t=K+1 in 2..8), M/W the verify-rows MoE pair + walk, K the
# DFlash2 candidate selector (n_cols>=16384). A plain boot leaves every announce
# structurally absent, so plain+doors arms are gated announce-FORBIDDEN only.
# Spec boots additionally demand the BATCHED verify-walk line + the vrest-head vrows pair
# announce and forbid the PER-ROW line (verify-batch default ON at this head).
doors() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log" fail=0
  local e_t="" e_x="" e_m="" e_k="" e_w="" spec_env=""
  for a in "$@"; do case "$a" in
    MEMRA_BF16_TCOLS_WIDE=1) e_t=1;;
    MEMRA_BF16_TCOLS_X1=1)   e_x=1;;
    MEMRA_MOE_VROWS_PACK=1)  e_m=1;;
    MEMRA_TOPK_SHARDS=1)     e_k=1;;
    MEMRA_GLM5_VERIFY_WS=1)  e_w=1;;
    MEMRA_GLM5_SPEC=1)       spec_env=1;;
  esac; done
  chk() { # expected(1/""), pattern, label
    local n; n=$(grep -c -- "$2" "$log")
    if [ -n "$1" ]; then
      [ "$n" -ge 1 ] || { echo "[mv] DOOR FAIL: expected '$3' announce, lines=$n"; fail=1; }
    else
      [ "$n" -eq 0 ] || { echo "[mv] DOOR FAIL: forbidden '$3' announce, lines=$n"; fail=1; }
    fi
    echo "door=$3 expected=${1:-0} lines=$n"
  }
  if [ -z "$spec_env" ]; then
    # non-spec boot: every door announce is structurally absent — forbid all five
    e_t=""; e_x=""; e_m=""; e_k=""; e_w=""
  fi
  {
    chk "$e_t" "\[bf16-tcols-wide\] engaged" "bf16-tcols-wide(T)"
    chk "$e_x" "\[bf16-tcols-x1\] engaged" "bf16-tcols-x1(X)"
    chk "$e_m" "\[moe-vrows-pack\] engaged" "moe-vrows-pack(M)"
    chk "$e_k" "\[topk-shards\] engaged" "topk-shards(K)"
    chk "$e_w" "\[glm5-verify-ws\] engaged" "glm5-verify-ws(W)"
    if [ -n "$spec_env" ]; then
      chk 1  "verify walk BATCHED per layer" "verify-walk-batched"
      chk "" "verify walk PER-ROW" "verify-walk-per-row"
      chk 1 "moe=pairs rows-call" "verify-walk-moe-pairs"
      chk 1 "\[glm5-vrows\] verify MoE batched across rows" "glm5-vrows"
    fi
    grep -m1 "\[bf16-tcols-wide\] engaged" "$log" || true
    grep -m1 "\[bf16-tcols-x1\] engaged" "$log" || true
    grep -m1 "\[moe-vrows-pack\] engaged" "$log" || true
    grep -m1 "\[topk-shards\] engaged" "$log" || true
    grep -m1 "\[glm5-verify-ws\] engaged" "$log" || true
  } > "$OUT/logs/boot-$name.doors" 2>&1
  cat "$OUT/logs/boot-$name.doors"
  [ "$fail" -eq 0 ] && echo "[mv] DOORS GREEN boot=$name" || echo "[mv] DOORS RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start) shift; stop && start "$@" ;;
  doors) shift; doors "$@" ;;
  stop) stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | serve.sh doors <name> [ENV=VAL ...] | serve.sh stop"; exit 2 ;;
esac
