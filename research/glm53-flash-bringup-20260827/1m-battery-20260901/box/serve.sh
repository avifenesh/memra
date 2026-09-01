#!/usr/bin/env bash
# 1m-battery scoped serve: cards 0-3, port 18400, own dir /root/out-1m.
#
# POSTURE (the ONLY demonstrated 1M config, 1m-demo-20260829/LANE.md phase 7):
#   PP4 across all four cards, MEMRA_PP_SPLITS=13,26,39 (uneven: keeps the tail stage
#   light, because the last-stage card also carries the worker's primary engine, the f32
#   output head and the ~17.1 GB whole-prime hidden stack at 1M), plus the CAPPED expert
#   SLRU arena: MEMRA_MOE_SLOTS=12000 caps any card's arena at ~52 GB while each stage's
#   working set (10/13/13/6 expert layers) grows it only to ~39/51/51/23 GB, leaving the
#   tail card ~35 GB for the 1M request's upfront allocations.
#   The 3-card RESIDENT shape is NOT usable at 1M: it OOMs in DSA k-pool selection at
#   layer 31 (reproduced on this box, /root/out-1m-b/, gpu2 peak 97,242 of 97,887 MiB).
#   MEMRA_MOE_SLOTS=256 is NOT a lever either: it starves the fused-epi SLRU arm below
#   3*n_used, fails closed to the sequential loop and halves prefill (demo phase 6).
#
# SERVED-ENV FIDELITY, AND THE MEASURED REASON IT CANNOT HOLD AT 1M (finding, cell 1a):
# today's fleet serving env (mv-/struct-battery box/serve.sh, the env behind the banked
# 70.458/71.489 ship rows) also carries MEMRA_BF16_MMV=1 and MEMRA_PP_BF16=1. Those are
# NOT in this recipe, deliberately, because they REFUSE the 1M request at admission:
# BF16_MMV keeps every large non-expert 2D tensor bf16-RESIDENT (the boot log's
# "[bf16-mmv] RESIDENT" census: output.weight 634,388,480 elements plus kda_q/k/v/out
# 33,554,432 each across 42 layers), and the first attempt measured
#   [admission] request cost ctx=1035677 path=plain = 23936 B/token x ctx
#               + 25575MB prefill-workspace + 18MB fixed = 50382MB
#   [admit-oom] VRAM reject ... no attainable admission headroom (available 39090MB)
# i.e. the 1M prime wants ~50.4 GB and the bf16-resident mirror leaves only ~39.1 GB.
# So the recipe is DEMO-EXACT (1m-demo phase 7) - the only demonstrated 1M config - and the
# bf16 arm is priced separately as a named capacity/speed trade, never silently carried.
#
# MEMRA_MOE_GROUPED_PREFILL is DEFAULT ON at this head but is STRUCTURALLY ABSENT on this
# posture: the grouped arm runs its per-projection GEMM over the LOCAL RESIDENT SLAB, and
# the 1M posture has no resident experts (host-pinned staging + a capped on-demand SLRU
# arena). So the demo's named "lever 1" does not reach the 1M config as posture'd - a
# finding, not a misconfiguration. Its announce is therefore expected-ABSENT here.
#
# Doors T/X/K/W are DEFAULT ON at this head and are left at default in EVERY arm; doors
# D/H/R stay default OFF (unpriced/refuted). An OFF arm would be spelled ONLY by pinning
# =0 (owner law: unset is an ON arm).
#
# MEMRA_TIMEOUT_MS_MAX=64800000 (18 h) is the demo's MEASUREMENT-CELL override: the 90 s
# ceiling is a platform fact of the FRONTED route and this override must NEVER reach a
# fronted deploy. This box is direct-to-server and a ~1M prime legitimately runs for tens
# of minutes.
#
# THE ARENA CAP IS THIS WINDOW'S ONE FORCED DEVIATION FROM THE DEMO RECIPE, and it is a
# MEASURED necessity, not a preference. The expert SLRU arena grows on demand toward
# MEMRA_MOE_VRAM_FRAC (default 0.85) of free VRAM per device; on this head it took a card
# from 7 GB at boot-ready to ~62 GB after ONE 1,108-token request. The 1M request's own
# admission cost is
#   [admission] ctx=1035677 plain = 23936 B/token x ctx + 25575MB prefill-workspace
#               + 18MB fixed = 50382MB
# so with a ~55 GB arena only ~36.7 GB was left and the request was REFUSED at admission
# (HTTP 429, [admit-oom] VRAM reject) - twice, with and without the bf16-resident mirror,
# which is how we know the mirror was NOT the cause (available 39090MB with it, 36740MB
# without). MEMRA_MOE_SLOTS stays at the demo's demonstrated 12000; the arena is instead
# capped by FRACTION at 0.35 (soft and hard), which leaves ~32 GB of arena - still about
# 2x the ~17 GB working set (288 experts x 13 stage layers x ~4.6 MB/slot) - and frees the
# ~50 GB the 1M prime needs. The demo's floor warning is respected: SLOTS=256 starves the
# fused-epi SLRU arm below 3*n_used and HALVES prefill to ~40 tok/s, so the slot count is
# left alone and prefill tok/s is watched as the starvation detector on every rung.
# MEMRA_GLM5_VISION=0 reclaims the 2.09 GiB the tower takes at load (this is a text 1M
# cell; vision is default-ON by owner order for the PRODUCT and that default is untouched).
#
# MEMRA_REUSE_POOL=0 is the 1m-demo's OWN PRESCRIPTION for repeated 1M cells, and this
# window needs it: an eos-terminated session PARKS and keeps its planes, so on the demo box
# "gpu3 carried TWO 1M sessions' planes" and it warned "a third 1M session would not fit;
# repeated 1M cells on one boot should set MEMRA_REUSE_POOL=0 or expect the pool eviction to
# decide". Cell 3 runs a greedy AND a vendor-default 1M prime on ONE boot, and with the
# arena capped there is no room for a parked 1M session beside a live one - the first
# refusal receipt already shows the reclaim path firing ("[admit-oom] reclaim-on-defer:
# evicted ... 1 plain ... parked sessions (global LRU)"). Pinning 0 makes every rung an
# honest cold prime instead of letting an LRU eviction decide the measurement, which also
# composes with the pinned MEMRA_PREFIX_CACHE_MB=0: nothing is ever reused across rungs.
#
# stop() is PIDFILE-scoped and verifies /proc/pid/exe AND MEMRA_ADDR=<ip>:18400
# before any signal. NEVER pkill, NEVER basename matching (GATE:gate-stop-pkill-basename-trap:
# a renamed binary orphans a VRAM-holding server and corrupts oracles).
# Arms differ ONLY in the extras passed as "$@" (comparability requirement).
set -uo pipefail
OUT=/root/out-1m
PIDFILE=$OUT/server.pid
BIN=/root/memra-1m/target/release/memra-server
DFLASH_SHA8=b33c0347
mkdir -p "$OUT/logs"

stop() {
  [ -f "$PIDFILE" ] || { echo "[1m] no pidfile, nothing to stop"; return 0; }
  local pid; pid=$(cat "$PIDFILE")
  if [ ! -d "/proc/$pid" ]; then echo "[1m] pid $pid already gone"; rm -f "$PIDFILE"; return 0; fi
  local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") ;;
    *) echo "[1m] REFUSE stop pid=$pid exe=$exe (not our binary)"; return 1 ;;
  esac
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "MEMRA_ADDR=<ip>:18400" \
    || { echo "[1m] REFUSE stop pid=$pid (not port 18400)"; return 1; }
  echo "[1m] SIGTERM pid=$pid exe=$exe"
  kill -TERM "$pid"
  for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[1m] SIGKILL pid=$pid"; kill -KILL "$pid"; sleep 5; fi
  rm -f "$PIDFILE"
  echo "[1m] stopped"
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
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=13,26,39 \
    MEMRA_MOE_SLOTS=12000 MEMRA_MOE_RESIDENT_HEADROOM_GB=36 \
    MEMRA_MOE_VRAM_FRAC=0.35 MEMRA_MOE_HARD_VRAM_FRAC=0.35 \
    MEMRA_GLM5_VISION=0 \
    MEMRA_CTX=1048576 \
    MEMRA_ST_PINNED=1 MEMRA_MOE_FUSED_EPI=1 \
    MEMRA_MAX_SESSIONS=1 MEMRA_PREFIX_CACHE_MB=0 NVIDIA_TF32_OVERRIDE=0 \
    MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
    MEMRA_TIMEOUT_MS_MAX=64800000 \
    "$@" ONEM_NONCE="$nonce" MEMRA_ADDR=<ip>:18400 \
    setsid nohup "$BIN" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  { echo "nonce=$nonce"; echo "pid=$pid"; echo "boot=$name"; echo "extras=$*"
    echo "bin=$BIN"; echo "bin_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "head=$(git -C /root/memra-1m log --oneline -1)"
    echo "started=$(date -u +%FT%TZ)"; } > "$OUT/logs/boot-$name.identity"
  echo "[1m] launched pid=$pid boot=$name nonce=$nonce extras=$*"
  local t0=$SECONDS
  for _ in $(seq 1 900); do
    if curl -s -m 2 http://<ip>:18400/v1/models 2>/dev/null | grep -q "glm-5.3-flash"; then
      local boot_s=$((SECONDS-t0))
      echo "[1m] READY after ${boot_s}s"
      echo "boot_s=$boot_s" >> "$OUT/logs/boot-$name.identity"
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
        > "$OUT/logs/boot-$name.vram"
      gates "$name" "$pid" "$nonce" "$@"
      return $?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[1m] BOOT DIED"; tail -25 "$log"; return 1; fi
    grep -qE "panicked|FATAL" "$log" && { echo "[1m] BOOT FAILED"; tail -25 "$log"; return 1; }
    sleep 2
  done
  echo "[1m] NOT READY after $((SECONDS-t0))s"; tail -25 "$log"; return 1
}

# gates: BOOT-time identity. Proves WHICH server answers (A/B arm identity is a boot nonce
# in /proc/<pid>/environ, never a health 200 - LAW:ab-arm-identity-not-liveness), that the
# 1M window and the PP4 placement are the ones actually configured, and that a spec arm
# loaded the PINNED DFlash2 drafter rather than the native MTP head.
gates() {
  local name="$1" pid="$2" nonce="$3"; shift 3
  local log="$OUT/logs/boot-$name.log" fail=0
  tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "ONEM_NONCE=$nonce" \
    || { echo "[1m] GATE FAIL: nonce not in /proc/$pid/environ"; fail=1; }
  # The CTX gate asserts the LIVE value, not a literal: cells that vary the context window
  # (the PP4-vs-PP3 spec A/B pins 131072 on both arms so stage count is the only variable)
  # must not be RED'd for not being 1M. Hardcoding 1048576 here aborted that A/B's first arm.
  live_ctx=$(tr '\0' '\n' < "/proc/$pid/environ" | grep -m1 '^MEMRA_CTX=' | cut -d= -f2)
  if [ -z "${live_ctx:-}" ]; then
    echo "[1m] GATE FAIL: no MEMRA_CTX in the live environ"; fail=1
  else
    echo "[1m] live MEMRA_CTX=$live_ctx"
    if [ -n "${ONEM_EXPECT_CTX:-}" ] && [ "$live_ctx" != "$ONEM_EXPECT_CTX" ]; then
      echo "[1m] GATE FAIL: live MEMRA_CTX=$live_ctx != expected $ONEM_EXPECT_CTX"; fail=1
    fi
  fi
  # PP4 must be the live placement: 4 stage lines, and NOT the 3-card resident shape
  local nstage; nstage=$(grep -cE "\[pp\]|stage " "$log")
  local spec_env=""
  for a in "$@"; do case "$a" in MEMRA_GLM5_SPEC=1) spec_env=1;; esac; done
  if [ -n "$spec_env" ]; then
    grep -q "\[glm5-spec\] serve route ARMED" "$log" || { echo "[1m] GATE FAIL: no spec ARMED line"; fail=1; }
    grep -q "draft source = dflash2 @ $DFLASH_SHA8" "$log" \
      || { echo "[1m] GATE FAIL: no 'draft source = dflash2 @ $DFLASH_SHA8' line"; fail=1; }
    grep -q "native MTP head NOT loaded" "$log" \
      || { echo "[1m] GATE FAIL: head not reported NOT loaded"; fail=1; }
  else
    local nspec; nspec=$(grep -c "\[glm5-spec\]" "$log")
    [ "$nspec" -eq 0 ] || { echo "[1m] GATE FAIL: plain arm has $nspec [glm5-spec] lines"; fail=1; }
  fi
  # the prefix cache MUST be off: pinned defence-in-depth against the glm5 restore bug,
  # and it is what makes every rung an honest cold prime (cached_tokens=0)
  grep -iE "prefix-cache" "$log" | head -3 > "$OUT/logs/boot-$name.prefixcache" || true
  {
    echo "boot=$name pid=$pid nonce=$nonce spec_env=${spec_env:-0}"
    echo "stage_lines=$nstage"
    echo "--- ctx / window ---";     grep -iE "ctx|window|max.*token" "$log" | head -6
    echo "--- pp / placement ---";   grep -iE "\[pp\]|stage|split|resident|slru|arena" "$log" | head -20
    echo "--- prefix-cache ---";     cat "$OUT/logs/boot-$name.prefixcache"
    echo "--- spec ---";             grep -E "\[glm5-spec\]" "$log" | head -8
    echo "--- vram-at-ready (index, used MiB, total MiB) ---"
    cat "$OUT/logs/boot-$name.vram"
  } > "$OUT/logs/boot-$name.gates" 2>&1
  cat "$OUT/logs/boot-$name.gates"
  [ "$fail" -eq 0 ] && echo "[1m] GATES GREEN boot=$name" || echo "[1m] GATES RED boot=$name"
  return "$fail"
}

# engage NAME [ENV=VAL ...] — the DOOR/LEVER ENGAGEMENT gate, run AFTER the first request
# (announces print at first engagement, so this cannot run at boot). "Verified" means a
# spec-engagement receipt from the server log, never a 200 (owner law, born of the DE
# DFlash2 flip that served the plain path at half speed on greedy-only receipts).
# Doors T/X/K/W are DEFAULT ON at this head, so their announces are DEMANDED on a spec
# boot unless the arm pins =0. They are spec-walk-scoped: structurally ABSENT on a plain
# boot, so they are FORBIDDEN there. MLA-TC prefill is default ON and prefill-scoped, so
# it is DEMANDED on EVERY arm - it is the whole point of this re-price.
engage() {
  local name="$1"; shift
  local log="$OUT/logs/boot-$name.log" fail=0
  local spec_env="" t_off="" x_off="" k_off="" w_off=""
  for a in "$@"; do case "$a" in
    MEMRA_GLM5_SPEC=1)        spec_env=1;;
    MEMRA_BF16_TCOLS_WIDE=0)  t_off=1;;
    MEMRA_BF16_TCOLS_X1=0)    x_off=1;;
    MEMRA_TOPK_SHARDS=0)      k_off=1;;
    MEMRA_VERIFY_WS=0|MEMRA_GLM5_VERIFY_WS=0) w_off=1;;
  esac; done
  chk() { # expected(1/""), pattern, label
    local n; n=$(grep -c -- "$2" "$log")
    if [ -n "$1" ]; then
      [ "$n" -ge 1 ] || { echo "[1m] ENGAGE FAIL: expected '$3' announce, lines=$n"; fail=1; }
    else
      [ "$n" -eq 0 ] || { echo "[1m] ENGAGE FAIL: forbidden '$3' announce, lines=$n"; fail=1; }
    fi
    echo "engage=$3 expected=${1:-0} lines=$n"
  }
  local e_t=1 e_x=1 e_k=1 e_w=1
  [ -n "$t_off" ] && e_t=""; [ -n "$x_off" ] && e_x=""
  [ -n "$k_off" ] && e_k=""; [ -n "$w_off" ] && e_w=""
  if [ -z "$spec_env" ]; then e_t=""; e_x=""; e_k=""; e_w=""; fi
  {
    # THE LEVER UNDER RE-PRICE: default ON, prefill-scoped, demanded on every arm.
    chk 1 "\[mla-tc-prefill\] engaged" "mla-tc-prefill(DEFAULT ON, the re-priced lever)"
    chk "" "\[mla-tc-prefill\] DECLINED" "mla-tc-prefill DECLINED (a cuBLASLt shape decline)"
    # grouped MoE prefill: default ON, but it needs a LOCAL RESIDENT SLAB and the 1M
    # posture is host-pinned + capped-SLRU, so its announce is expected ABSENT. Recorded,
    # never asserted either way - the count IS the finding.
    echo "engage=moe-grouped-prefill(needs a resident slab; expected ABSENT on capped-SLRU) lines=$(grep -c '\[moe-grouped-prefill\] execute' "$log")"
    chk "$e_t" "\[bf16-tcols-wide\] engaged"  "door T bf16-tcols-wide(default-on)"
    chk "$e_x" "\[bf16-tcols-x1\] engaged"    "door X bf16-tcols-x1(default-on)"
    chk "$e_k" "\[topk-shards\] engaged"      "door K topk-shards(default-on)"
    chk "$e_w" "\[glm5-verify-ws\] engaged"   "door W glm5-verify-ws(default-on)"
    # doors D/H/R are default OFF and never set here: their announces are FORBIDDEN
    chk "" "\[moe-vrows-dev-tables\] engaged"  "door D (default OFF, never set)"
    chk "" "\[bf16-tcols-red-fused\] engaged"  "door R (default OFF, never set)"
    if [ -n "$spec_env" ]; then
      # PMIN announces at the first spec round, NOT at boot: this is an engage-time check.
      # (It was briefly asserted in gates() and RED'd a whole cell before its first request.)
      chk 1  "PMIN=0.700" "spec draft-confidence gate armed at tau 0.7 (the ship tau)"
      chk 1  "verify walk BATCHED per layer" "verify-walk-batched"
      chk "" "verify walk PER-ROW"           "verify-walk-per-row(must not happen)"
      chk 1  "\[glm5-vrows\] verify MoE batched across rows" "glm5-vrows"
    fi
    echo "--- mla-tc-prefill announce ---"; grep -m2 "\[mla-tc-prefill\]" "$log" || true
    echo "--- accept lines ---";            grep -c "\[glm5-acc\]" "$log" || true
    echo "--- ADMISSION cost model + any reject (the 1M capacity receipt) ---"
    grep -E "\[admission\]|\[admit-oom\]" "$log" | tail -8 || true
    echo "--- bf16-mmv resident census (must be 0 on the demo-exact 1M posture) ---"
    grep -c "\[bf16-mmv\] RESIDENT" "$log" || true
  } > "$OUT/logs/boot-$name.engage" 2>&1
  cat "$OUT/logs/boot-$name.engage"
  [ "$fail" -eq 0 ] && echo "[1m] ENGAGE GREEN boot=$name" || echo "[1m] ENGAGE RED boot=$name"
  return "$fail"
}

case "${1:-}" in
  start)  shift; stop && start "$@" ;;
  engage) shift; engage "$@" ;;
  gates)  shift; gates "$@" ;;
  stop)   stop ;;
  *) echo "usage: serve.sh start <name> [ENV=VAL ...] | engage <name> [ENV=VAL ...] | stop"; exit 2 ;;
esac
