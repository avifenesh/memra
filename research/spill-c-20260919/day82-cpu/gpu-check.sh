#!/usr/bin/env bash
# DAY82 section 2's local RTX 5090 check: the door at I18 (run-gen-i18 = c7294b912) and at I20 (run-gen-i20 =
# 8efea3a54), 32 tokens each, order I18, I20, I20, I18, the cell's argv (MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=9986,
# prompt 55 88 13, the door with a 16 GiB host tier). Each must exit 0 with MATCH; I20's tape (the generated ids) and
# its host demand sequence (the `[expert-host-slru] key=` lines without the slot, SHA-256, as day82-read.py reads it)
# must equal I18's. Waits for an idle card (the lock free, no compute app, 40 GiB MemAvailable), then holds
# /tmp/memra-5090.lock for the four runs, inside the 1200% CPU cap. Never signals another process.
# usage: bash day82-cpu/gpu-check.sh <out-dir>
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
T=/home/avifenesh/projects/wt-spill-c
BINS=$T/target/c-bins
LOCK=/tmp/memra-5090.lock
ART=/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
OUT=${1:?out dir}
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*"; }
n=0
while :; do
    if flock -n "$LOCK" true 2>/dev/null \
       && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
       && [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $((40 * 1024 * 1024)) ]; then
        break
    fi
    n=$((n + 1)); [ $n -ge 34560 ] && { log "card never idle in 48 h"; exit 1; }
    [ $((n % 360)) -eq 0 ] && log "waiting for an idle card ($((n / 720)) h)"
    sleep 5
done
sha256sum "$BINS/run-gen-i18" "$BINS/run-gen-i20" > "$OUT/binary.sha256"
exec 9> "$LOCK"
flock -w 600 9 || { log "lock not taken in 600 s"; exit 1; }
log "lock held; runs start"
for label in i18-a i20-a i20-b i18-b; do
    bin=run-gen-${label%-*}
    systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 \
        MEMRA_MOE_SLOTS=9986 "$BINS/$bin" "$ART" 55 88 13 --experts-via-tier --expert-bank-host-bytes=17179869184 \
        > "$OUT/$label.log" 2>&1
    echo "$?" > "$OUT/$label.exit"
done
flock -u 9
log "runs done; lock released"
