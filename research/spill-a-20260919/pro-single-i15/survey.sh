#!/usr/bin/env bash
# OWED item 15's kernel price on the target card (DAY43.md section 1, a reading): the day-38 survey probe (G's framed
# SHA-256 kernel bitwise against the receipt program on 56 sizes and offsets; priced at 16 x 60 KiB, 32 x 60 KiB,
# 32 x 1 MiB, 32 x 4 MiB, N=5 each; the host's write-combined and cached read rates) in ONE collector hold:
# `tier-battery.py --rig pro-single --external-lock --execute bash survey.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-i15
cd /root/wt-a || exit 1
mkdir -p "$R/survey"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/survey/LOCK.json"
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
echo "hold taken $(date -u +%FT%TZ) apps=[${apps//$'\n'/; }]" > "$R/survey/run.log"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> "$R/survey/run.log" 2>&1
sha256sum "$R/bins/survey/day38-hash-survey" > "$R/survey/binary.sha256"
CUDA_VISIBLE_DEVICES=0 "$R/bins/survey/day38-hash-survey" > "$R/survey/survey.log" 2>&1
echo "rc=$?" >> "$R/survey/run.log"
