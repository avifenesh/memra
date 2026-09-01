#!/usr/bin/env bash
# Full-information K-policy matrix on box1. Caller stages this committed tree; this script
# owns one complete GPU window and retains raw client/server output before analysis.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/kpolicy-20260808
TS=${KPOLICY_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${KPOLICY_OUT:-$LANE/raw/$TS}
mkdir -p "$OUT/responses"

DRIVER=$OUT/driver.log
POINTS=$OUT/points.jsonl
SUMMARY=$OUT/SUMMARY.md
PORT=${KPOLICY_PORT:-8143}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS=${KPOLICY_REPS:-3}
CLASSES=${KPOLICY_CLASSES:-"cold-short cold-long cached-long"}
MODELS=${KPOLICY_MODELS:-"q9 q27"}
MAX_TOKENS=${KPOLICY_MAX_TOKENS:-128}

Q9=${Q9:-/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
Q9_DRAFT=${Q9_DRAFT:-/scratch-models/draft-9b-owntrim-nvfp4head-q4blk.gguf}
Q27=${Q27:-/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
Q27_DRAFT=${Q27_DRAFT:-/scratch-models/draft-daily-owntrim-nvfp4head-q4blk.gguf}
K_Q9=${KPOLICY_K_Q9:-"0 1 2 3 5"}
K_Q27=${KPOLICY_K_Q27:-"0 2 3 5"}
SHORT=$ROOT/research/e2e/prompts/p1-code-short.txt
LONG=$ROOT/research/e2e/prompts/p3-agentic-long-v3.txt
BIN=${CARGO_TARGET_DIR:-$ROOT/target}/release/memra-server

exec > >(tee "$DRIVER") 2>&1

cleanup_pid=
stop_server() {
    local pid=${1:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap 'stop_server "$cleanup_pid"' EXIT INT TERM

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 240); do
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

gpu_sample() {
    {
        printf '%s,' "$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
            --format=csv,noheader | paste -sd ';' -
    } >> "$OUT/gpu.csv"
}

model_paths() {
    case "$1" in
        q9) printf '%s\n%s\n' "$Q9" "$Q9_DRAFT" ;;
        q27) printf '%s\n%s\n' "$Q27" "$Q27_DRAFT" ;;
        *) echo "unknown model $1" >&2; return 2 ;;
    esac
}

model_ks() {
    case "$1" in
        q9) echo "$K_Q9" ;;
        q27) echo "$K_Q27" ;;
        *) return 2 ;;
    esac
}

arm_order() {
    local model=$1 rep=$2
    local ks
    ks=$(model_ks "$model")
    case "$model:$rep" in
        q9:1) echo "0 1 2 3 5" ;;
        q9:2) echo "5 3 2 1 0" ;;
        q9:3) echo "2 5 1 0 3" ;;
        q27:1) echo "0 2 3 5" ;;
        q27:2) echo "5 3 2 0" ;;
        q27:3) echo "2 5 0 3" ;;
        *) echo "$ks" ;;
    esac
}

class_order() {
    case "$1" in
        1) echo "$CLASSES" ;;
        2) echo "$CLASSES" | awk '{ for (i=NF; i>=1; --i) printf "%s%s", $i, i == 1 ? ORS : OFS }' ;;
        *) echo "$CLASSES" | awk '{
               if (NF < 2) { print; next }
               for (i=2; i<=NF; ++i) printf "%s ", $i
               print $1
           }' ;;
    esac
}

echo "=== kpolicy matrix $TS ==="
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
git status --short --untracked-files=no
for model in $MODELS; do
    while read -r artifact; do
        test -f "$artifact" || { echo "FAIL: missing artifact $artifact"; exit 1; }
    done < <(model_paths "$model")
done
test -f "$SHORT" && test -f "$LONG"

echo "=== artifact manifest ==="
{
    for model in $MODELS; do model_paths "$model"; done
} | sort -u | xargs sha256sum > "$OUT/artifact-sha256.txt"
cat "$OUT/artifact-sha256.txt"

echo "=== release build ==="
cargo build --release -p memra-server > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
sha256sum "$BIN" > "$OUT/binary-sha256.txt"
cat "$OUT/binary-sha256.txt"

exec 9>/tmp/memra-gpu.lock
flock -w "${KPOLICY_LOCK_WAIT:-14400}" 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
echo "GPU lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true
gpu_sample

for model in $MODELS; do
    mapfile -t artifacts < <(model_paths "$model")
    model_path=${artifacts[0]}
    draft_path=${artifacts[1]}
    for rep in $(seq 1 "$REPS"); do
        for k in $(arm_order "$model" "$rep"); do
            if ! echo " $(model_ks "$model") " | grep -q " $k "; then
                continue
            fi
            label="${model}-k${k}-r${rep}"
            server_log="$OUT/${label}-server.log"
            port_free || {
                echo "FAIL: port $PORT is already listening before $label"
                ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
                exit 1
            }
            echo "=== arm $label ==="
            env \
                -u MEMRA_PP_STAGES \
                -u MEMRA_PP_DEVICES \
                -u MEMRA_PP_SHARD \
                -u MEMRA_SERVE_SPEC \
                -u MEMRA_SPEC_GATE \
                -u MEMRA_SPEC_GATE_LOW \
                -u MEMRA_SPEC_GATE_HIGH \
                -u MEMRA_API_KEY \
                -u MEMRA_API_KEYS \
                CUDA_VISIBLE_DEVICES=0 \
                MEMRA_MODELS="${model}=${model_path}+${draft_path}" \
                MEMRA_ADDR="$ADDR" \
                MEMRA_COMPAT=openai \
                MEMRA_CTX=8192 \
                MEMRA_PREFIX_CACHE_MB=512 \
                MEMRA_MAX_SESSIONS=2 \
                MEMRA_REUSE_POOL=2 \
                MEMRA_PRIME_CHUNK=2048 \
                MEMRA_SPEC_STATS=1 \
                MEMRA_SPEC_K="$k" \
                "$BIN" > "$server_log" 2>&1 &
            cleanup_pid=$!
            if ! wait_up "$cleanup_pid"; then
                echo "FAIL: $label server did not become ready"
                tail -100 "$server_log" || true
                exit 1
            fi
            if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
                    | grep -q "pid=$cleanup_pid,"; then
                echo "FAIL: $label responder is not child pid $cleanup_pid"
                ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
                exit 1
            fi

            warm_log="$OUT/${label}-warmup.log"
            python3 "$LANE/measure_client.py" \
                --base "$BASE" --model "$model" --class cold-short \
                --prompt "$SHORT" --k "$k" --rep "$((90 + rep))" \
                --max-tokens 16 --out "$OUT/warmup.jsonl" \
                --raw-dir "$OUT/responses" > "$warm_log" 2>&1

            for prompt_class in $(class_order "$rep"); do
                case "$prompt_class" in
                    cold-short) prompt=$SHORT ;;
                    cold-long|cached-long) prompt=$LONG ;;
                    *) echo "FAIL: unknown prompt class $prompt_class"; exit 1 ;;
                esac
                cell_log="$OUT/${model}-${prompt_class}-k${k}-r${rep}.log"
                python3 "$LANE/measure_client.py" \
                    --base "$BASE" --model "$model" --class "$prompt_class" \
                    --prompt "$prompt" --k "$k" --rep "$rep" \
                    --max-tokens "$MAX_TOKENS" --out "$POINTS" \
                    --raw-dir "$OUT/responses" > "$cell_log" 2>&1
                cat "$cell_log"
                salt="kpolicy-${model}-${prompt_class}-k${k}-r${rep}"
                if ! grep -F "tenant=\"$salt\"" "$server_log" \
                        | grep -q " K=$k "; then
                    echo "FAIL: $label $prompt_class has no matching per-request K receipt"
                    grep -F "[spec-k]" "$server_log" | tail -20 || true
                    exit 1
                fi
            done

            curl -sf "$BASE/metrics" > "$OUT/${label}-metrics.json" 2>&1 || true
            stop_server "$cleanup_pid"
            cleanup_pid=
            sleep 3
            gpu_sample
        done
    done
done

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
gpu_sample
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

analyze_args=()
for model in $MODELS; do
    analyze_args+=(--expect "$model:$(model_ks "$model" | tr ' ' ',')")
done
python3 "$LANE/analyze.py" "$POINTS" \
    --expected-reps "$REPS" \
    --classes "$(echo "$CLASSES" | tr ' ' ',')" \
    "${analyze_args[@]}" \
    --out "$SUMMARY"
echo "KPOLICY_MATRIX_DONE out=$OUT"
