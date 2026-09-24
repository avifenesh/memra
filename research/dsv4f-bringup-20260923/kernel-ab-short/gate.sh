#!/usr/bin/env bash
# gate.sh <out-dir> <cmd...>: run a gate binary under the box GPU lock with a tee'd raw log.
set -uo pipefail
out=$1; shift
mkdir -p "$out"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo SLOT_BUSY | tee "$out/gate.log"; exit 75; }
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$out/pre-apps.csv"
[[ ! -s "$out/pre-apps.csv" ]] || { echo GPU_NOT_IDLE | tee "$out/gate.log"; exit 76; }
env | grep '^MEMRA_' | sort > "$out/env.txt"
sha256sum "$1" > "$out/binary.sha256"
echo "START utc=$(date -u +%FT%TZ) cmd=$*" > "$out/gate.log"
export PATH=/usr/local/cuda/bin:$PATH
"$@" 9>&- >> "$out/gate.log" 2>&1
rc=$?
echo "GATE_DONE rc=$rc utc=$(date -u +%FT%TZ)" >> "$out/gate.log"
