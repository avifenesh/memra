#!/bin/bash
# Door-free RE-RUN of the defect-7 attribution cell. Same pre-registered plan, same
# drivers, same 48-row structure, same blind protocol; the ONE axis changed is the
# BINARY (memra 3999a92a6, built after the 2026-08-29 bank-v2 door REMOVAL - the original
# lane ran on 8695bdef4a, whose step37 defaults armed the logits-corrupting doors).
# See PLAN-DIFF.md for every mechanical difference and its reason.
#
# ONE boot block: refusal-by-construction receipt (env-only, no GPU) -> boot (door-free,
# environ proven from /proc) -> clean transcript build -> turn-2 stop-inside-think probe
# (n=8 addendum) -> all 48 rows interleaved -> down. Resume-safe: banked turns/rows skip.
set -u
LANE=${LANE:?set LANE to the lane dir}
D7_BIN=${D7_BIN:-/home/ubuntu/degen-rerun/bin/memra-server-3999a92a6}
EXPECT_MD5=${D7_EXPECT_MD5:?set D7_EXPECT_MD5 to the built binary md5 (arm identity binds on md5, not commit subject: build.rs fingerprint staleness means system_fingerprint can lie)}
MODEL=${D7_MODEL:-/data/models/step37-flash-nvfp4}
PORT=18903
LOCK=/tmp/memra-degen-rerun.lock
LOCKWAIT=${D7_LOCKWAIT:-21600}

MD5=$(md5sum "$D7_BIN" | cut -d' ' -f1)
echo "RESULTS HEADER: bin=$D7_BIN md5=$MD5 expect=$EXPECT_MD5 date=$(date -u +%FT%TZ)"
[ "$MD5" = "$EXPECT_MD5" ] || { echo "MD5_MISMATCH"; exit 2; }

# Serving env = the original lane's launch env MINUS the two removed doors. ERA_BASE is
# the byte-verbatim 2026-08-29-era step37 serving env list (carried through
# research/toolchain-ab-20260831 -> research/perf-chain-20260831 harness/launch.sh, mode
# `era-nodoors`); GATES is d7-run.sh's own gate list, unchanged.
ERA_BASE="MEMRA_OPROJ_TAIL=1 MEMRA_DEV1_ROUTER=1 MEMRA_LEN_MIRROR_LAZY=1 MEMRA_ASYNC_CHAIN=8 MEMRA_SHEXP_OVERLAP=1 MEMRA_ROUTES_PRESTAGE=1 MEMRA_MOE_DIRECT=1 MEMRA_OPROJ_DIRECT=1 MEMRA_STEP_TP=0-44@0,1 MEMRA_STEP_TP_NATIVE_P2P=1 MEMRA_STEP_NVFP4_DEV_ROUTES=1 MEMRA_STEP_TP_DECODE_V2=1 MEMRA_STEP_TP_QKV_FUSED=1 MEMRA_BF16_MMV=1 MEMRA_STEP_TP_DEV_ROUTER=1 MEMRA_STEP_TP_DCW=1 MEMRA_RMS_BLOCK=1024 MEMRA_SIG_EXPF_DEV=1 MEMRA_HEAD_SPLIT=1 MEMRA_FA_DCW_MEMSET=0 MEMRA_NO_LOCAL_SHADOW=1 MEMRA_FUSE_ROPE_APPEND=1 MEMRA_SEL_MIRROR=1 MEMRA_FA_COMBINE_S=1 MEMRA_MAX_CTX=262144"
GATES="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1"
CUR_LOG=$LANE/raw/server-d7boot.log
REC=$LANE/receipts
SERVER_PID=""
mkdir -p "$REC" "$LANE/raw"

# --- refusal by construction ---------------------------------------------------------
# The tree is >= 75bf4ce76, so the doors are GONE from the engine and the binary refuses
# to boot for ANY model family when a stale recipe still sets them. Env-only check, runs
# before any model load, so this costs milliseconds and no GPU. Banked as a receipt
# rather than asserted in prose.
refusal_receipt() {
  local R=$REC/door-refusal-by-construction.txt
  {
    echo "date=$(date -u +%FT%TZ)"
    echo "bin=$D7_BIN md5=$MD5"
    echo "-- probe: same recipe + MEMRA_NVFP4_BANK_V2=1 (expect boot refusal) --"
  } > "$R"
  env $ERA_BASE $GATES MEMRA_NVFP4_BANK_V2=1 MEMRA_STEP_GEMM_PRIME_SUFFIX=0 \
    MEMRA_MODELS="step37=$MODEL" MEMRA_ADDR=127.0.0.1:18904 \
    timeout 120 "$D7_BIN" >> "$R" 2>&1
  echo "probe_exit=$?" >> "$R"
  if grep -q "were REMOVED from the engine on 2026-08-29" "$R"; then
    echo "REFUSAL_OK (door recipe refused at boot)" | tee -a "$R"
  else
    echo "REFUSAL_UNPROVEN: binary did not emit the removed-doors refusal" | tee -a "$R"
    return 1
  fi
  {
    echo "-- the measured recipe's own env, doors proven ABSENT --"
    for v in MEMRA_NVFP4_BANK_V2 MEMRA_SEL_DOWN8; do
      case " $ERA_BASE $GATES " in *" $v="*) echo "$v: PRESENT (FAIL)";; *) echo "$v: absent";; esac
    done
  } >> "$R"
}

boot() {
  local T0=$(date +%s)
  if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then
    echo "REFUSE: GPU not empty (another lane may be running)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; return 1
  fi
  local NONCE="d7rr-$(date +%s)-$RANDOM"
  echo "[boot d7boot] door-free log=$CUR_LOG nonce=$NONCE"
  env $ERA_BASE $GATES MEMRA_STEP_GEMM_PRIME_SUFFIX=0 RUST_BACKTRACE=1 \
    BOOT_NONCE="$NONCE" \
    MEMRA_MODELS="step37=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT \
    nohup "$D7_BIN" > "$CUR_LOG" 2>&1 &
  SERVER_PID=$!
  for hw in $(seq 1 540); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$CODE" = "200" ] && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "[boot] SERVER_DIED"; tail -30 "$CUR_LOG"; return 1; }
    sleep 5
  done
  [ "$CODE" = "200" ] || { echo "[boot] BOOT_TIMEOUT"; return 1; }
  echo "[boot] HEALTH=200 boot_seconds=$(( $(date +%s) - T0 ))"
  # Arm identity proven from /proc, not intended (perf-chain boot.sh craft).
  local ENVF=$REC/environ-d7boot.txt
  tr '\0' '\n' < /proc/$SERVER_PID/environ | grep -E '^(MEMRA_|BOOT_NONCE=)' | sort > "$ENVF"
  {
    echo "server_pid=$SERVER_PID"
    echo "exe=$(readlink /proc/$SERVER_PID/exe)"
    echo "bin_md5=$MD5"
    echo "bin_fingerprint_sha12=$(grep -aom1 'memra-[0-9a-f]\{12\}' "$D7_BIN" | cut -d- -f2)"  # bare sha12: the public-boundary live_fingerprint rule forbids the memra-<sha12> token in this repo
    echo "boot_nonce_asked=$NONCE"
    echo "boot_nonce_live=$(sed -n 's/^BOOT_NONCE=//p' "$ENVF")"
    echo "kernel_boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "environ_census=$ENVF ($(wc -l < "$ENVF") MEMRA_/BOOT vars)"
    echo "door_MEMRA_NVFP4_BANK_V2_live=$(grep -c '^MEMRA_NVFP4_BANK_V2=' "$ENVF")"
    echo "door_MEMRA_SEL_DOWN8_live=$(grep -c '^MEMRA_SEL_DOWN8=' "$ENVF")"
    nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.sm --format=csv,noheader
  } > "$REC/boot-d7boot.receipt"
  [ "$(sed -n 's/^BOOT_NONCE=//p' "$ENVF")" = "$NONCE" ] || { echo "IDENTITY_FAIL nonce"; return 1; }
  grep -q '^MEMRA_NVFP4_BANK_V2=' "$ENVF" && { echo "ENV_FAIL: bank-v2 door live"; return 1; }
  grep -q '^MEMRA_SEL_DOWN8=' "$ENVF" && { echo "ENV_FAIL: sel-down8 door live"; return 1; }
  cat "$REC/boot-d7boot.receipt"
  return 0
}

down() {
  local ILL=$(grep -ac ILLEGAL "$CUR_LOG")
  local H87=$(grep -ac '#87' "$CUR_LOG")
  local PAN=$(grep -ac "panicked at" "$CUR_LOG")
  echo "[down] ILLEGAL=$ILL hash87=$H87 panics=$PAN"
  if [ "$ILL" != "0" ] || [ "$H87" != "0" ] || [ "$PAN" != "0" ]; then
    echo "FAULT_FOUND ILLEGAL=$ILL hash87=$H87 panics=$PAN" | tee -a "$LANE/FAULT"
  fi
  kill -TERM $SERVER_PID 2>/dev/null; sleep 12; kill -KILL $SERVER_PID 2>/dev/null; sleep 3
  pgrep -f "$D7_BIN" >/dev/null && echo "[down] SERVER_STILL_UP" || echo "[down] SERVER_GONE"
}

block() {
  refusal_receipt || exit 6
  boot || exit 3
  # Prove uncorrupted text on the incident's own short prompt BEFORE any evaluated row.
  # Captured, then gated: a pipe would swallow the exit status.
  env P=$PORT LOG=$CUR_LOG LANE=$LANE BIN_MD5=$MD5 BOOT_ID=d7boot \
    python3 "$LANE/d7-doorfree-gate.py"
  GRC=$?
  [ $GRC -eq 0 ] || { echo "DOORFREE_GATE aborted the cell (rc=$GRC)"; down; exit 8; }
  env P=$PORT LOG=$CUR_LOG LANE=$LANE BIN_MD5=$MD5 BOOT_ID=d7boot \
    python3 "$LANE/d7-drive.py" clean_transcript || { down; exit 4; }
  env P=$PORT LOG=$CUR_LOG LANE=$LANE BIN_MD5=$MD5 BOOT_ID=d7boot \
    python3 "$LANE/d7-t2probe.py" || { down; exit 7; }
  env P=$PORT LOG=$CUR_LOG LANE=$LANE BIN_MD5=$MD5 BOOT_ID=d7boot \
    python3 "$LANE/d7-drive.py" rows
  local RC=$?
  down
  [ $RC -eq 0 ] || exit 5
}

(
  exec 9>$LOCK
  echo "[lock] waiting (max ${LOCKWAIT}s) $(date -u +%T)"
  flock -w $LOCKWAIT 9 || { echo "LOCK_TIMEOUT"; exit 9; }
  echo "[lock] acquired $(date -u +%T)"
  block
) || exit $?
echo "D7_RERUN_DONE"
