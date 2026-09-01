#!/usr/bin/env bash
# Boot one arm with a PID-verified identity receipt.
# Usage: boot.sh <arm-tag> <binary>
set -u
ARM=${1:?arm tag}
BIN=${2:?binary}
LAUNCHER=${3:-/home/ubuntu/toolchain-ab/launch.sh}
D=/home/ubuntu/toolchain-ab
LOG=$D/logs/server-$ARM.log
R=$D/receipts/boot-$ARM.receipt
if pgrep -f "^/home/ubuntu/toolchain-ab/bin/memra-server" >/dev/null; then
  echo "REFUSE: an ab-lane server is already up"; exit 1
fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then
  echo "REFUSE: GPU not empty"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; exit 1
fi
NONCE="ab37-$ARM-$(date +%s)-$RANDOM"
{
  echo "arm=$ARM"
  echo "boot_nonce=$NONCE"
  echo "bin=$BIN"
  echo "bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "built_from=$(git -C $D/memra log -1 --format=%H)"
  echo "bin_fingerprint=$(grep -aom1 "memra-[0-9a-f]\\{12\\}" "$BIN")"
} > "$R"
nohup "$LAUNCHER" "$BIN" "$NONCE" > "$LOG" 2>&1 < /dev/null &
sleep 3
SPID=$(pgrep -f "^/home/ubuntu/toolchain-ab/bin/memra-server" | head -1)
[ -n "$SPID" ] || { echo "BOOT_FAIL: no server pid"; tail -20 "$LOG"; exit 2; }
UP=0
for i in $(seq 1 240); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18620/health 2>/dev/null)
  [ "$CODE" = "200" ] && { UP=1; break; }
  kill -0 "$SPID" 2>/dev/null || { echo "SERVER_DIED during boot"; tail -30 "$LOG"; exit 3; }
  sleep 5
done
[ "$UP" = 1 ] || { echo "BOOT_TIMEOUT"; tail -20 "$LOG"; exit 4; }
# PID-verified arm identity: exe symlink + this boot's nonce in the live environ.
EXE=$(readlink /proc/$SPID/exe)
ENV_NONCE=$(tr '\0' '\n' < /proc/$SPID/environ | grep '^BOOT_NONCE=' | cut -d= -f2)
{
  echo "server_pid=$SPID"
  echo "exe=$EXE"
  echo "environ_nonce=$ENV_NONCE"
  nvidia-smi --query-gpu=index,memory.used,power.limit,clocks.sm --format=csv,noheader
} >> "$R"
[ "$ENV_NONCE" = "$NONCE" ] || { echo "IDENTITY_FAIL: environ nonce mismatch"; exit 5; }
case "$EXE" in *"$(basename "$BIN")"*) ;; *) echo "IDENTITY_FAIL: exe $EXE != $BIN"; exit 5;; esac
echo "BOOT_OK pid=$SPID arm=$ARM"
cat "$R"
