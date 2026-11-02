#!/usr/bin/env bash
# Rig-side, after chain.sh: the F8F4 served-acceptance A/B, the K-rung arm's c1 cell and an F8F4
# serving arm (warm cold-prefill + c4) on gb10-b, then hand the box to the #1026 agent.
set -u
export BIN=~/memra-run/bin-b014549cb92a
export LEASE=<lease>
A=~/.cache/memra-1025/gb10-b/arms.log
waitfor --file "$A" --grep 'chain done' --timeout 3h --quiet
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) phase3 start" >> "$A"
/usr/bin/ssh -n -o BatchMode=yes gb10-b "cd ~/memra-run && SPARKLOCK_LEASE=$LEASE BIN=$BIN bash ./f8f4-accept.sh" > ~/.cache/memra-1025/gb10-b/f8f4-accept.out 2>&1
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) f8f4-accept rc=$?" >> "$A"
ARMS="gguf-mtpmask-k2" USERS="1" TAGSFX=gb10-b-c1 bash ~/.cache/memra-1025/gb10-b/arms.sh
ARMS="gguf-mtpmask" EXTRA="MEMRA_MMQ_F8F4=1" USERS="4" TAGSFX=gb10-b-f8f4 bash ~/.cache/memra-1025/gb10-b/arms.sh
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) phase3 done" >> "$A"
