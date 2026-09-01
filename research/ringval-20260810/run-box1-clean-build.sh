#!/usr/bin/env bash
# Rebuild every ringval executable from a new target directory and prove server relinkage.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
REPO=${REPO:-$HOME/memra-cx-ringval}
TARGET=${TARGET:-$HOME/memra-cx-ringval-target-ringval}
STAMP=${RINGVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${RINGVAL_OUT:-$HOME/ringval/receipts/clean-build-$STAMP}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-019428e217e297cb5981d201a4a520aee69222a6}

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

(
    flock -w 60 9 || { echo LOCK_TIMEOUT; exit 75; }
    echo "lock_acquired=$(date -u +%FT%TZ) host=$(hostname) stamp=$STAMP"
    source=$(git -C "$REPO" rev-parse HEAD)
    echo "source_commit=$source"
    git -C "$REPO" status --short --branch --untracked-files=no
    [[ $source == "$EXPECTED_SOURCE" ]]
    [[ ! -e $TARGET ]] || {
        echo "FAIL: clean target already exists: $TARGET"
        exit 1
    }
    apps=$(compute_apps)
    [[ -z $apps ]] || { echo "$apps"; exit 1; }
    snapshot "$OUT/nvidia-smi-before.log" preflight
    {
        rustc --version
        cargo --version
        nvcc --version
        df -h "$HOME"
    } | tee "$OUT/toolchain.log"
    echo "target_dir=$TARGET"
    echo "build_start=$(date -u +%FT%TZ)"
    CARGO_TARGET_DIR="$TARGET" cargo build --manifest-path "$REPO/Cargo.toml" \
        --release -p memra-server --bin memra-server
    CARGO_TARGET_DIR="$TARGET" cargo build --manifest-path "$REPO/Cargo.toml" \
        --release -p memra-engine --bins
    CARGO_TARGET_DIR="$TARGET" cargo build --manifest-path "$REPO/Cargo.toml" \
        --release -p memra-tokenizer --bin tok-check
    echo "build_done=$(date -u +%FT%TZ)"
    bins=(
        memra-server kernel-check run-gen run-spec replay-acceptance concat-prime-probe
        prime-batch-gate decode-batch-gate tok-check
    )
    for bin in "${bins[@]}"; do
        test -x "$TARGET/release/$bin"
        sha256sum "$TARGET/release/$bin"
        stat -c 'artifact=%n bytes=%s mtime=%y' "$TARGET/release/$bin"
    done | tee "$OUT/binary-identity.txt"
    grep -aFq 'capped at' "$TARGET/release/memra-server"
    grep -aFq 'SWA ring lapped checkpoint' "$TARGET/release/memra-server"
    {
        echo 'server_marker=capped at'
        echo 'server_marker=SWA ring lapped checkpoint'
    } >"$OUT/server-ring-string.txt"
    snapshot "$OUT/nvidia-smi-after.log" final
    echo "clean_build_verdict=PASS"
    echo "lock_released=$(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
