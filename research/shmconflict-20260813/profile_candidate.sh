#!/usr/bin/env bash
set -euo pipefail

if [[ "${MEMRA_5090_LOCK_HELD:-}" != 1 ]]; then
    echo "refusing candidate profile: hold /tmp/memra-5090.lock and set MEMRA_5090_LOCK_HELD=1" >&2
    exit 2
fi

model=${1:-}
case "$model" in
    q27)
        model_path=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
        port=18847
        ;;
    q35)
        model_path=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
        port=18848
        ;;
    *)
        echo "usage: $0 q27|q35" >&2
        exit 2
        ;;
esac

root=$(cd "$(dirname "$0")/../.." && pwd)
out="$root/research/shmconflict-20260813/raw/profile-candidate-$model"
server=/home/avifenesh/.cache/memra-targets/cx-shmconflict-candidate-v1/release/memra-server
fatbin=/home/avifenesh/.cache/memra-targets/cx-shmconflict-candidate-v1/release/build/memra-engine-052022fb9838c766/out/flash_attn.fatbin
request="$root/research/shmconflict-20260813/profile_request.py"
report_stem="/tmp/memra-shmconflict-20260813/$model-candidate-v1-ncu-r1"
report="$report_stem.ncu-rep"
ncu=/usr/local/cuda-13.1/bin/ncu

if [[ -e "$out" ]]; then
    echo "refusing to overwrite candidate profile output: $out" >&2
    exit 2
fi
mkdir -p "$out"

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | tee "$out/compute-apps-before.log"
if [[ -s "$out/compute-apps-before.log" ]]; then
    echo "refusing candidate profile: GPU compute application present" >&2
    exit 2
fi

{
    date -u '+NCU_LOCK_ACQUIRED=%Y-%m-%dT%H:%M:%SZ'
    stat -c 'INBOX=%y %s %n' /home/avifenesh/.lanectl/inbox/cx-shmconflict.md
    nvidia-smi --query-gpu=index,name,pstate,clocks.current.sm,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader
    sha256sum "$server" "$request" "$fatbin"
    "$ncu" --version
} 2>&1 | tee "$out/ncu-preflight.log"

profile_pid=
cleanup() {
    local pid
    while read -r pid; do
        [[ -n "$pid" ]] && sudo -n kill -TERM "$pid" 2>/dev/null || true
    done < <(pgrep -f -x "$server" || true)
    if [[ -n "$profile_pid" ]] && kill -0 "$profile_pid" 2>/dev/null; then
        kill -TERM "$profile_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

sudo -n env \
    CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MODELS="$model=$model_path" \
    MEMRA_ADDR="127.0.0.1:$port" \
    MEMRA_COMPAT=openai \
    MEMRA_TAG=cx-shmconflict-local-ncu \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_CTX=8192 \
    MEMRA_PREFIX_CACHE_MB=4096 \
    MEMRA_PREFIX_DEDUP=1 \
    MEMRA_REUSE_POOL=0 \
    MEMRA_AFFINITY=0 \
    MEMRA_MAX_SESSIONS=8 \
    "$ncu" \
        --target-processes application-only \
        --kernel-name-base function \
        --kernel-name 'regex:^fa_prefill_qw_db$' \
        --filter-mode per-launch-config \
        --launch-count 1 \
        --section SpeedOfLight \
        --section Occupancy \
        --section SchedulerStats \
        --section WarpStateStats \
        --section InstructionStats \
        --section MemoryWorkloadAnalysis \
        --section SourceCounters \
        --export "$report_stem" \
        --force-overwrite \
        "$server" > >(tee "$out/ncu-launch.log") 2>&1 &
profile_pid=$!

ready=0
for _ in $(seq 1 1500); do
    if curl -fsS "http://127.0.0.1:$port/readyz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$profile_pid" 2>/dev/null; then
        echo "NCU target exited before readiness" >&2
        wait "$profile_pid"
        exit 1
    fi
    sleep 0.2
done
if [[ "$ready" != 1 ]]; then
    echo "NCU target readiness timeout" >&2
    exit 1
fi

python3 "$request" \
    --base "http://127.0.0.1:$port" \
    --model "$model" \
    --cache-salt "shmconflict-$model-candidate-v1-ncu-r1" \
    --timeout 900 2>&1 | tee "$out/ncu-request.log"

while read -r pid; do
    [[ -n "$pid" ]] && sudo -n kill -TERM "$pid"
done < <(pgrep -f -x "$server")
wait "$profile_pid"
profile_pid=
trap - EXIT

[[ -s "$report" ]]
"$ncu" --import "$report" --page raw --csv 2>&1 \
    | tee "$out/ncu-raw.csv" >/dev/null
"$ncu" --import "$report" --page details --csv 2>&1 \
    | tee "$out/ncu-details.csv" >/dev/null
"$ncu" --import "$report" --page source --print-source sass --csv 2>&1 \
    | tee "$out/ncu-source-sass.csv" >/dev/null
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass --function fa_prefill_qw_db "$fatbin" 2>&1 \
    | tee "$out/fa_prefill_qw_db.sass" >/dev/null

{
    date -u '+NCU_END=%Y-%m-%dT%H:%M:%SZ'
    sha256sum "$report"
    nvidia-smi --query-gpu=index,name,pstate,clocks.current.sm,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader
} 2>&1 | tee "$out/ncu-postflight.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | tee "$out/compute-apps-after.log"
test ! -s "$out/compute-apps-after.log"
