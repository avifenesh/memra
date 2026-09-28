#!/usr/bin/env bash
# DAY88 section 4's local RTX 5090 check: the promoted binary run-gen-p88 (0155bc69f) with no flag (the door, the
# registered pool and the prefetch by default) against run-gen-i22 (4b378a064) with --experts-via-tier
# --expert-bank-host-bytes=17179869184 (the door as the cells qualified it), order p88 i22 i22 p88, then p88 with
# MEMRA_EXPERTS_VIA_TIER=0 (the rollback: the legacy with its prefetch). The cells' spill argv (MEMRA_MOE_RESIDENT=0
# MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986, prompt 55 88 13). Waits for an idle card (the lock free, no compute app, 40 GiB
# MemAvailable) up to 48 h, then holds /tmp/memra-5090.lock for the five runs, each under nice 19 inside a 600% CPU
# scope with MemoryMax=24G (the door pins its 15 GB bank). Never signals another process.
# usage: bash day88-cpu/gpu-check.sh <out-dir>
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
sha256sum "$BINS/run-gen-p88" "$BINS/run-gen-i22" > "$OUT/binary.sha256"
exec 9> "$LOCK"
flock -w 600 9 || { log "lock not taken in 600 s"; exit 1; }
log "lock held; runs start"
run() { # $1 label  $2 binary  $3.. env words and flags
    local label=$1 bin=$2; shift 2
    local envs=() flags=()
    for a in "$@"; do case $a in MEMRA_*) envs+=("$a") ;; *) flags+=("$a") ;; esac; done
    local started; started=$(date +%s.%N)
    systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=24G nice -n 19 env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 \
        MEMRA_MOE_SLOTS=9986 "${envs[@]}" "$BINS/$bin" "$ART" 55 88 13 "${flags[@]}" > "$OUT/$label.log" 2>&1
    echo "$?" > "$OUT/$label.exit"
    awk -v s="$started" -v e="$(date +%s.%N)" 'BEGIN {printf "%.3f\n", e - s}' > "$OUT/$label.wall"
}
q=(--experts-via-tier --expert-bank-host-bytes=17179869184)
run p88-a run-gen-p88
run i22-a run-gen-i22 "${q[@]}"
run i22-b run-gen-i22 "${q[@]}"
run p88-b run-gen-p88
run leg run-gen-p88 MEMRA_EXPERTS_VIA_TIER=0
flock -u 9
log "runs done; lock released"
