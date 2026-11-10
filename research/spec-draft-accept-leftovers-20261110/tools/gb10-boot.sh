#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one boot of ab_cell.py.
# usage: gb10-boot.sh OUTSUB ARM BOOT [ENV=VALUE ...]
set -u
D=$HOME/.cache/memra-1160
SUB=$1; ARM=$2; BOOT=$3; shift 3
export CELL_MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CELL_PORT=18194 BOOT_START=$BOOT
for kv in "$@"; do export "$kv"; done
mkdir -p "$D/run/$SUB"
python3 "$D/ab_cell.py" "$D/run/$SUB" "$D/bin/memra-server-$ARM" "$ARM" >> "$D/run/$SUB/boots.log" 2>&1
echo "boot $BOOT $ARM rc=$?"
tail -n 12 "$D/run/$SUB/boots.log" | cut -c1-200
