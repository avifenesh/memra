#!/usr/bin/env bash
# Boot one arm with a PID-verified identity receipt. Ported from toolchain-ab-20260831
# harness/boot.sh; adds the MODE argument, banks the LIVE MEMRA_* environ, and asserts a
# per-mode expectation table so the env axis under test is proven from /proc, never merely
# intended.
# Usage: boot.sh <arm-tag> <binary> <mode>
#
# pgrep patterns are ANCHORED on this lane's absolute binary path
# ("^/home/ubuntu/perf-chain/bin/memra-server"), which is what keeps them from
# self-matching the driving shell's own command line.
set -u
ARM=${1:?arm tag}
BIN=${2:?binary}
MODE=${3:?mode}
D=/home/ubuntu/perf-chain
PAT="^/home/ubuntu/perf-chain/bin/memra-server"
LOG=$D/logs/server-$ARM.log
R=$D/receipts/boot-$ARM.receipt
if pgrep -f "$PAT" >/dev/null; then
  echo "REFUSE: a perf-chain server is already up"; exit 1
fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then
  echo "REFUSE: GPU not empty"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; exit 1
fi
S12=$(basename "$BIN" | sed 's/^memra-server-//')
NONCE="pc37-$ARM-$(date +%s)-$RANDOM"
{
  echo "arm=$ARM"
  echo "mode=$MODE"
  echo "boot_nonce=$NONCE"
  # Kernel boot_id: the bench box rebooted mid-cell once (2026-08-31 11:12Z, graceful,
  # not ours). Banking it makes an arm measured across a reboot detectable instead of
  # silently pooled with arms from a different kernel/clock session.
  echo "kernel_boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "bin=$BIN"
  echo "bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "bin_fingerprint=$(grep -aom1 "memra-[0-9a-f]\\{12\\}" "$BIN")"
  echo "built_from=$(sed -n 's/^sha=//p' "$D/receipts/build-$S12.receipt" 2>/dev/null)"
  echo "git_log_1=$(sed -n 's/^git_log_1=//p' "$D/receipts/build-$S12.receipt" 2>/dev/null)"
} > "$R"
nohup "$D/harness/launch.sh" "$BIN" "$NONCE" "$MODE" > "$LOG" 2>&1 < /dev/null &
sleep 3
SPID=$(pgrep -f "$PAT" | head -1)
[ -n "$SPID" ] || { echo "BOOT_FAIL: no server pid"; tail -20 "$LOG"; exit 2; }
UP=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18640/health 2>/dev/null)
  [ "$CODE" = "200" ] && { UP=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "SERVER_DIED during boot"; tail -30 "$LOG"; exit 3; }
  sleep 5
done
[ "$UP" = 1 ] || { echo "BOOT_TIMEOUT"; tail -20 "$LOG"; exit 4; }

# --- PID-verified arm identity + live env census -------------------------------------
ENVF=$D/receipts/environ-$ARM.txt
tr '\0' '\n' < /proc/$SPID/environ | grep -E '^(MEMRA_|PC_MODE=|BOOT_NONCE=)' | sort > "$ENVF"
EXE=$(readlink /proc/$SPID/exe)
ENV_NONCE=$(sed -n 's/^BOOT_NONCE=//p' "$ENVF")
ENV_MODE=$(sed -n 's/^PC_MODE=//p' "$ENVF")
{
  echo "server_pid=$SPID"
  echo "exe=$EXE"
  echo "environ_nonce=$ENV_NONCE"
  echo "environ_mode=$ENV_MODE"
  echo "environ_census=$ENVF ($(wc -l < "$ENVF") MEMRA_/PC vars)"
  nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.sm --format=csv,noheader
} >> "$R"
[ "$ENV_NONCE" = "$NONCE" ] || { echo "IDENTITY_FAIL: environ nonce mismatch"; exit 5; }
[ "$ENV_MODE" = "$MODE" ] || { echo "IDENTITY_FAIL: environ mode $ENV_MODE != $MODE"; exit 5; }
case "$EXE" in *"$(basename "$BIN")"*) ;; *) echo "IDENTITY_FAIL: exe $EXE != $BIN"; exit 5;; esac

# --- per-mode env expectation table, asserted against the LIVE environ ----------------
# "NAME=VALUE" must be present; "NAME!" must be ABSENT.
case "$MODE" in
  era)                EXPECT="MEMRA_NVFP4_BANK_V2=1 MEMRA_SEL_DOWN8=1 MEMRA_STEP_VISION_DIR! MEMRA_SPEC_GRAPH_FILTERED! MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  era-nodoors)        EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR! MEMRA_SPEC_GRAPH_FILTERED! MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  fixed-nofiltered)   EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR! MEMRA_SPEC_GRAPH_FILTERED=0 MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  fixed-nochaingraph) EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR! MEMRA_MTP_CHAIN_GRAPH=0 MEMRA_SPEC_GRAPH_FILTERED! MEMRA_STEP35_DRAFT_DCW!" ;;
  fixed-nodcw)        EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR! MEMRA_STEP35_DRAFT_DCW=0 MEMRA_SPEC_GRAPH_FILTERED! MEMRA_MTP_CHAIN_GRAPH!" ;;
  current)            EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR=/data/models/step37-flash-nvfp4 MEMRA_SPEC_GRAPH_FILTERED! MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  current-novision)   EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR! MEMRA_SPEC_GRAPH_FILTERED! MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  current-nofiltered) EXPECT="MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8! MEMRA_STEP_VISION_DIR=/data/models/step37-flash-nvfp4 MEMRA_SPEC_GRAPH_FILTERED=0 MEMRA_MTP_CHAIN_GRAPH! MEMRA_STEP35_DRAFT_DCW!" ;;
  *) echo "ENV_FAIL: no expectation table for mode $MODE"; exit 6 ;;
esac
for e in $EXPECT; do
  case "$e" in
    *!) N=${e%!}
        grep -q "^$N=" "$ENVF" && { echo "ENV_FAIL: $N must be UNSET for mode $MODE but is $(grep "^$N=" "$ENVF")"; exit 6; } ;;
    *)  grep -qx "$e" "$ENVF" || { echo "ENV_FAIL: mode $MODE expects $e; live environ has $(grep "^${e%%=*}=" "$ENVF" || echo UNSET)"; exit 6; } ;;
  esac
done
echo "ENV_VERIFIED mode=$MODE ($EXPECT)" >> "$R"
echo "BOOT_OK pid=$SPID arm=$ARM mode=$MODE"
cat "$R"
