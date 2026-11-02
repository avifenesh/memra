#!/usr/bin/env bash
# Rig-side: the #1025 GB10 serving validation queue on gb10-b, one arm after another.
set -u
export BIN=~/memra-run/bin-b014549cb92a
export LEASE=<lease>
A=~/.cache/memra-1025/gb10-b/arms.sh
L=~/.cache/memra-1025/gb10-b/arms.log
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) chain start" >> "$L"
ARMS="gguf-mtpmask" USERS="1 4 8 16 32" QUALITY_ARMS="gguf-mtpmask" bash $A
ARMS="gguf-dflash-noskip" NONSTREAM=1 bash $A
ARMS="gguf-dflash" USERS="1 4 8 16 32" bash $A
ARMS="gguf-mtpmask-nobook" USERS="32" QUALITY_ARMS="gguf-mtpmask-nobook" bash $A
ARMS="gguf-mtpmask-k2" USERS="4 8 16 32" bash $A
ARMS="dflash" USERS="4 8 16 32" bash $A
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) chain done" >> "$L"
