#!/usr/bin/env bash
# Rig: kcl_driver.sh resumed from boot START (the first driver died with its session at boot 7).
# Same arms, same command, one boot per sparklock lease. usage: kcl_resume.sh UNIT START
set -u
UNIT=$1; START=$2
D=$HOME/.cache/memra-gb10-r4
L=$D/kcl-driver-$UNIT.log
R=.cache/memra-gb10-r4
G8='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF'
n=0
for arm in k3 k2 k2 k3 k3 k2 k2 k3 k3 k2 k2 k3; do
  n=$((n + 1))
  [ "$n" -lt "$START" ] && continue
  id=$(sparklock acquire --unit "$UNIT" --ttl 12m --for "memra #1093 cold-long K A/B on GB10, boot $n ($arm), one boot per lease" --wait 120m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) boot $n: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) boot $n $arm: lease $id (resumed driver)" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "cd ~/$R/kcl && CELL_PORT=18193 BOOT_START=$n PROMPT_SET=gb10 CELL_MODEL=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_EXTRA=\"MEMRA_MTP_DRAFT=$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf MEMRA_PREFIX_STABLE_BOUNDARY=1\" timeout 11m python3 cold_long_ab.py ~/$R/kcl/out ~/$R/bins/main-a8a980a017/memra-server $arm" >> "$D/kcl-boots-$UNIT.log" 2>&1
  echo "$(date -Is) boot $n $arm: exit $?" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
done
echo "$(date -Is) all boots done" >> "$L"
