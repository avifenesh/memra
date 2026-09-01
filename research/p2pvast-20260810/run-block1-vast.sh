#!/usr/bin/env bash
# Bounded Vast block 1: stop serving, prove the engine/native peer path, restore serving + soak.
set -euo pipefail

ROOT=${P2PVAST_ROOT:-/workspace/memra}
OUT=${P2PVAST_OUT:-/root/p2pvast-receipts/block1}
BASE=${P2PVAST_BASE:-http://127.0.0.1:8002}
MODEL=${P2PVAST_MODEL:-/workspace/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

stop_runtime() {
    pkill -f 'release/memra-serve[r]' 2>/dev/null || true
    pkill -f '[s]oak.py' 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! pgrep -f 'release/memra-serve[r]|[s]oak.py' >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "FAIL: serving runtime did not stop within 60s"
    return 1
}

restore_runtime() {
    echo "RESTORE_BEGIN $(date -u +%FT%TZ)"
    if ! pgrep -f 'release/memra-serve[r]' >/dev/null 2>&1; then
        setsid nohup /root/start-memra.sh > /var/log/memra-server.log 2>&1 < /dev/null &
    fi
    sleep 90
    curl -fsS "$BASE/v1/models" > "$OUT/restart-models.json"
    if ! pgrep -f '[s]oak.py' >/dev/null 2>&1; then
        setsid nohup python3 /root/soak.py > /dev/null 2>&1 < /dev/null &
    fi
    sleep 2
    pgrep -af '[m]emra-server|[s]oak.py' > "$OUT/restart-processes.log"
    cp /var/log/memra-server.log "$OUT/restart-server.log"
    grep -E '\[pp\].*(cross-device|peer|transport)' /var/log/memra-server.log \
        > "$OUT/restart-peer-lines.log" || true
    echo "RESTORE_OK $(date -u +%FT%TZ)"
}

finish() {
    local rc=$?
    trap - EXIT
    set +e
    restore_runtime
    local restore_rc=$?
    set -e
    if (( rc == 0 && restore_rc != 0 )); then
        rc=$restore_rc
    fi
    exit "$rc"
}
trap finish EXIT

echo "BLOCK1_BEGIN $(date -u +%FT%TZ)"
cd "$ROOT"
git rev-parse HEAD > "$OUT/source-commit.txt"
git status --short --branch > "$OUT/source-status.txt"
curl -fsS "$BASE/v1/models" > "$OUT/pre-models.json"
pgrep -af '[m]emra-server|[s]oak.py' > "$OUT/pre-processes.log" || true
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"

stop_runtime
echo "RUNTIME_STOPPED $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-stopped.csv" 2>&1 || true
nvidia-smi topo -m > "$OUT/topology.txt"
nvidia-smi -q > "$OUT/nvidia-smi-query.txt"

source /root/.cargo/env
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc
cargo build --release --bin memra-server --bin pp-transport-smoke --bin gguf-inspect \
    2>&1 | tee "$OUT/build.log"
sha256sum target/release/memra-server target/release/pp-transport-smoke \
    target/release/gguf-inspect > "$OUT/binary-sha256.txt"
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120 \
    research/p2pvast-20260810/p2p_probe.cu -o "$OUT/p2p-probe" \
    2>&1 | tee "$OUT/p2p-probe-build.log"

target/release/gguf-inspect "$MODEL" > "$OUT/model-inspect.log" 2>&1
if [[ -x /root/simpleP2P-official && -x /root/p2pBandwidthLatencyTest-official ]]; then
    git -C /root/cuda-samples-p2p rev-parse HEAD > "$OUT/cuda-samples-commit.txt"
    sha256sum /root/simpleP2P-official /root/p2pBandwidthLatencyTest-official \
        > "$OUT/cuda-samples-binary-sha256.txt"
fi

set +e
"$OUT/p2p-probe" 2>&1 | tee "$OUT/p2p-microbench-compat.log"
probe_compat_rc=${PIPESTATUS[0]}
env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64 \
    "$OUT/p2p-probe" 2>&1 | tee "$OUT/p2p-microbench-host-driver.log"
probe_host_rc=${PIPESTATUS[0]}

if [[ -x /root/simpleP2P-official ]]; then
    /root/simpleP2P-official 2>&1 | tee "$OUT/official-simplep2p-compat.log"
    simple_compat_rc=${PIPESTATUS[0]}
    env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64 \
        /root/simpleP2P-official 2>&1 | tee "$OUT/official-simplep2p-host-driver.log"
    simple_host_rc=${PIPESTATUS[0]}
else
    simple_compat_rc=127
    simple_host_rc=127
fi

if [[ -x /root/p2pBandwidthLatencyTest-official ]]; then
    /root/p2pBandwidthLatencyTest-official 2>&1 | tee "$OUT/official-bandwidth-compat.log"
    bandwidth_compat_rc=${PIPESTATUS[0]}
    env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64 \
        /root/p2pBandwidthLatencyTest-official 2>&1 \
        | tee "$OUT/official-bandwidth-host-driver.log"
    bandwidth_host_rc=${PIPESTATUS[0]}
else
    bandwidth_compat_rc=127
    bandwidth_host_rc=127
fi

env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    target/release/pp-transport-smoke 2>&1 | tee "$OUT/engine-transport-compat.log"
engine_compat_rc=${PIPESTATUS[0]}
env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
    target/release/pp-transport-smoke 2>&1 | tee "$OUT/engine-transport-host-driver.log"
engine_host_rc=${PIPESTATUS[0]}
set -e

printf '%s\n' \
    "probe_compat=$probe_compat_rc" \
    "probe_host_driver=$probe_host_rc" \
    "simplep2p_compat=$simple_compat_rc" \
    "simplep2p_host_driver=$simple_host_rc" \
    "bandwidth_compat=$bandwidth_compat_rc" \
    "bandwidth_host_driver=$bandwidth_host_rc" \
    "engine_compat=$engine_compat_rc" \
    "engine_host_driver=$engine_host_rc" \
    > "$OUT/probe-exit-codes.txt"

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post-probe.csv"
echo "BLOCK1_MEASUREMENT_DONE $(date -u +%FT%TZ)"
