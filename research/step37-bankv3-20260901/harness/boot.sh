#!/usr/bin/env bash
# Boot one arm with a PID-verified identity receipt. Ported from toolchain-ab-20260831
# harness/boot.sh; adds the MODE argument, banks the LIVE MEMRA_* environ, and asserts a
# per-mode expectation table so the env axis under test is proven from /proc, never merely
# intended.
# Usage: boot.sh <arm-tag> <binary> <mode>
#
# pgrep patterns are ANCHORED on this lane's absolute binary path
# ("^/home/ubuntu/bankv3/lane/bin/memra-server"), which is what keeps them from
# self-matching the driving shell's own command line.
set -u
ARM=${1:?arm tag}
BIN=${2:?binary}
MODE=${3:?mode}
D=/home/ubuntu/bankv3/lane
PAT="^/home/ubuntu/bankv3/lane/bin/memra-server"
LOG=$D/logs/server-$ARM.log
R=$D/receipts/boot-$ARM.receipt
# THE BINARY PATH MUST BE ABSOLUTE, and this is checked BEFORE anything is launched.
# Incident 2026-09-01, this lane, first boot: `boot.sh GOFF bin/memra-server-<sha> gate-off`
# launched fine and then reported "BOOT_FAIL: no server pid", because `$PAT` is ANCHORED on the
# absolute path (deliberately — an unanchored pattern self-matches the driving shell) while
# launch.sh `exec`s whatever it was handed, so argv[0] was relative and pgrep could not see it.
# boot.sh exited; the nohup'd server did not, and a fully loaded step37 sat on 130 GB of VRAM
# with no supervisor and no receipt. That is the gate-stop pkill class: a pattern that stops
# matching orphans a VRAM-holding server and corrupts every arm after it. Refusing a relative
# path costs one line and turns a 4-minute silent orphan into an instant error.
case "$BIN" in
  /*) ;;
  *) echo "REFUSE: binary path must be ABSOLUTE (got '$BIN'); \$PAT is anchored on $PAT so a relative argv[0] is invisible to pgrep and the server would orphan"; exit 1 ;;
esac
case "$BIN" in
  "$D"/bin/*) ;;
  *) echo "REFUSE: binary must live under $D/bin (got '$BIN') or the anchored pattern cannot match it"; exit 1 ;;
esac
[ -x "$BIN" ] || { echo "REFUSE: '$BIN' is not executable"; exit 1; }
if pgrep -f "$PAT" >/dev/null; then
  echo "REFUSE: a bank-v3 server is already up"; exit 1
fi
# A stale flock file is harmless (flock keys on the inode, not on existence) but a HELD lock
# means someone else's server is mid-boot and invisible to pgrep for a few seconds.
if command -v flock >/dev/null && ! flock -n "/tmp/memra-bv3.lock" true 2>/dev/null; then
  echo "REFUSE: /tmp/memra-bv3.lock is held — another launch is in flight"; exit 1
fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then
  echo "REFUSE: GPU not empty"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; exit 1
fi
S12=$(basename "$BIN" | sed 's/^memra-server-//')
NONCE="bv3-$ARM-$(date +%s)-$RANDOM"
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
tr '\0' '\n' < /proc/$SPID/environ | grep -E '^(MEMRA_|BV3_MODE=|BOOT_NONCE=)' | sort > "$ENVF"
EXE=$(readlink /proc/$SPID/exe)
ENV_NONCE=$(sed -n 's/^BOOT_NONCE=//p' "$ENVF")
ENV_MODE=$(sed -n 's/^BV3_MODE=//p' "$ENVF")
{
  echo "server_pid=$SPID"
  echo "exe=$EXE"
  echo "environ_nonce=$ENV_NONCE"
  echo "environ_mode=$ENV_MODE"
  echo "environ_census=$ENVF ($(wc -l < "$ENVF") MEMRA_/BV3 vars)"
  nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.sm --format=csv,noheader
} >> "$R"
[ "$ENV_NONCE" = "$NONCE" ] || { echo "IDENTITY_FAIL: environ nonce mismatch"; exit 5; }
[ "$ENV_MODE" = "$MODE" ] || { echo "IDENTITY_FAIL: environ mode $ENV_MODE != $MODE"; exit 5; }
case "$EXE" in *"$(basename "$BIN")"*) ;; *) echo "IDENTITY_FAIL: exe $EXE != $BIN"; exit 5;; esac

# --- per-mode env expectation table, asserted against the LIVE environ ----------------
# "NAME=VALUE" must be present; "NAME!" must be ABSENT.
case "$MODE" in
  # PRICING ARMS (spec on). Cumulative: each arm adds one door to the one above it, and the
  # doors NOT yet reached are asserted ABSENT — a leaked door from a previous rotation is the
  # failure mode that would silently merge two arms into one number.
  v3-off)        EXPECT="MEMRA_NVFP4_BANK_SM! MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=1" ;;
  v3-sm)         EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=1" ;;
  v3-sm-gu)      EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU=1 MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=1" ;;
  v3-sm-gu-d8)   EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU=1 MEMRA_NVFP4_SEL_DOWN8=1 MEMRA_SERVE_SPEC=1" ;;
  # BYTE/CONTENT GATE ARMS (spec off, so the tape measures the bank and not the scheduler).
  gate-off)      EXPECT="MEMRA_NVFP4_BANK_SM! MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=0" ;;
  gate-sm)       EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=0" ;;
  gate-sm-gu)    EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU=1 MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=0" ;;
  gate-sm-gu-d8) EXPECT="MEMRA_NVFP4_BANK_SM=1 MEMRA_NVFP4_SEL_GU=1 MEMRA_NVFP4_SEL_DOWN8=1 MEMRA_SERVE_SPEC=0" ;;
  # gate-main: the doors are absent because the BINARY has no code for them, not because the
  # env omitted them. Both facts are asserted -- the env here, the binary in launch.sh.
  gate-main)     EXPECT="MEMRA_NVFP4_BANK_SM! MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8! MEMRA_SERVE_SPEC=0" ;;
  *) echo "ENV_FAIL: no expectation table for mode $MODE"; exit 6 ;;
esac
# The retired bundle names are asserted ABSENT in EVERY mode, not just the ones that would
# notice. The server refuses to boot with them, so a violation could not reach a measurement
# -- but a receipt that does not say so leaves a reader guessing which program was armed.
EXPECT="$EXPECT MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8!"
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
