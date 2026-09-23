#!/usr/bin/env bash
# vm-host-556b - #556 affected-case native check (attempt 2, device budget tuned to one 53.8 MB 9B entry): host-tier demote/promote/restore identity plus graceful SIGTERM drain
# Runs on the VM as root. Attempt 1 (default 1024 MB budget held both 53.8 MB seed entries, no eviction) stays on disk renamed. Evidence dir is never reused; a failed run stays on disk.
set -uo pipefail
export PATH=/root/.cargo/bin:$PATH
GPU=GPU-7edaf141-7bed-7e99-10d0-fc79d455c842
W=/data/work/memra-native-20260923
TESTED=8a3adc6e712a3c23337e28aeb755ab50955ad832
SRC=$W/pr556-${TESTED:0:10}/src
BIN=$W/pr556-${TESTED:0:10}/build/target/release/memra-server
MODEL=/data/models/kernel-oracles/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
EV=$W/host-556-${TESTED:0:10}-cache80
log() { echo "[$(date -u +%FT%TZ)] $*"; }

[[ -e $EV ]] && { log "REFUSED: $EV exists"; exit 1; }
nvidia-smi conf-compute -grs | grep -q ': ready$' || { log "REFUSED: CC GPU not ready"; exit 1; }
python3 /root/cuinit.py | grep -q '^readback_ok True$' || { log "REFUSED: cuInit probe failed"; exit 1; }
mkdir -p "$EV/lease"
{
    echo "tested_commit=$TESTED"
    echo "tools_head=$(cd "$SRC" && git rev-parse HEAD)"
    echo "cc_status=$(nvidia-smi conf-compute -f | tail -1)"
    echo "cc_ready=$(nvidia-smi conf-compute -grs | tail -1)"
    nvidia-smi --query-gpu=uuid,name,driver_version,pstate,clocks.max.sm,power.limit --format=csv,noheader
} > "$EV/conditions.txt"
sha256sum "$BIN" "$MODEL" > "$EV/inputs.sha256"
for arm in default teeth; do
    log "identity gate arm=$arm start"
    teeth=0; [[ $arm == teeth ]] && teeth=1
    (cd "$SRC" && memra-gpu-run --gpus "$GPU" --receipt "$EV/lease/$arm" -- \
        env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock MEMRA_HOSTGATE_CACHE_MB=80 MEMRA_HOSTGATE_TEETH=$teeth \
        bash tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN" "$EV/$arm") > "$EV/$arm.gate.log" 2>&1
    rc=$?
    logs=$(find "$EV/$arm" -name '*.log' | sort)
    n=$(echo "$logs" | grep -c .)
    clean=$(grep -l '^\[server\] GPU worker shutdown complete' $logs 2>/dev/null | wc -l)
    log "identity gate arm=$arm rc=$rc server_logs=$n clean_shutdown=$clean"
done
log "host-556b done"
