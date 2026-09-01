#!/usr/bin/env bash
# Run the standing serve-smoke unchanged with the Step35 SWA ring enabled on every boot.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
BIN=${BIN:-$TARGET/release/memra-server}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/serve-smoke-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}
EXPECTED_BINARY=${EXPECTED_BINARY:-7f04f76715d637c46a379366a833d518aed9d465a5dcfd1ffee53be79d9b9cef}

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

wait_idle() {
    local _
    for _ in $(seq 1 180); do
        [[ -z $(compute_apps) ]] && return 0
        sleep 1
    done
    compute_apps
    return 1
}

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    source=$(git -C "$REPO" rev-parse HEAD)
    binary=$(sha256sum "$BIN" | awk '{print $1}')
    echo "source_commit=$source"
    echo "binary_sha256=$binary"
    echo "serve_smoke_sha256=$(sha256sum "$REPO/tools/serve-smoke.sh" | awk '{print $1}')"
    echo "cache_meter_gate_sha256=$(sha256sum "$REPO/tools/cache-meter-gate.py" | awk '{print $1}')"
    echo "target_link=$(readlink -f "$REPO/target")"
    git -C "$REPO" status --short --branch --untracked-files=no
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT"
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ $binary == "$EXPECTED_BINARY" ]]
    [[ $(readlink -f "$REPO/target") == "$TARGET" ]]
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; exit 1; }
    snapshot "$OUT/nvidia-smi-before.log" preflight

    set +e
    timeout 14400 env CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_SWA_RING=1 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 \
        "$REPO/tools/serve-smoke.sh" "$MODEL" "$DRAFT" \
        2>&1 | tee "$OUT/serve-smoke.log"
    smoke_rc=${PIPESTATUS[0]}
    set -e
    echo "$smoke_rc" >"$OUT/serve-smoke.rc"
    [[ -f /tmp/serve-smoke.log ]] && cp /tmp/serve-smoke.log "$OUT/final-server.log"
    [[ -f /tmp/serve-smoke-affinity.json ]] \
        && cp /tmp/serve-smoke-affinity.json "$OUT/affinity-record.json"
    wait_idle
    snapshot "$OUT/nvidia-smi-after.log" final
    grep -E 'serve-smoke: [0-9]+ failed' "$OUT/serve-smoke.log" | tail -1 \
        | tee "$OUT/verdict.txt"
    grep -q '\[admission\].*capped at 4639 rows' "$OUT/final-server.log"
    echo "serve_smoke_rc=$smoke_rc" | tee -a "$OUT/verdict.txt"
    echo "lock_released=$(date -u +%FT%TZ)" | tee -a "$OUT/verdict.txt"
    exit "$smoke_rc"
) 9>/tmp/memra-gpu.lock
