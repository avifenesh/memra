#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one boot of a cell program in box/.
# usage: gb10-boot.sh PROGRAM OUTSUB ARM BOOT [ENV=VALUE ...]
set -u
D=$HOME/.cache/memra-carry
PROG=$1; SUB=$2; ARM=$3; BOOT=$4; shift 4
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export CELL_MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CELL_PORT=18194 BOOT_START=$BOOT
for kv in "$@"; do export "$kv"; done
mkdir -p "$D/run/$SUB"
{ date -Is; echo "boot $BOOT $ARM"; sha256sum "$D/bin/memra-server-$ARM"; uptime; } >> "$D/run/$SUB/guard.log"
python3 "$D/box/$PROG" "$D/run/$SUB" "$D/bin/memra-server-$ARM" "$ARM" >> "$D/run/$SUB/boots.log" 2>&1
rc=$?
echo "boot $BOOT $ARM rc=$rc"
tail -n 14 "$D/run/$SUB/boots.log" | cut -c1-200
exit $rc
