#!/usr/bin/env bash
# Exact single-card Q27 serve shape qualified by research/sellgate-20260812.
set -Eeuo pipefail

repo=${SERVETEST_REPO:-/opt/memra-src/memra}
model=${SERVETEST_MODEL:-/scratch/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
keys=${SERVETEST_KEYS:-/root/memra-secrets/keys.toml}
metrics_token_file=${SERVETEST_METRICS_TOKEN_FILE:-/root/memra-secrets/metrics-token}
server=${SERVETEST_SERVER:-$repo/target/release/memra-server}

[[ -x "$server" ]] || { printf 'missing server: %s\n' "$server" >&2; exit 1; }
[[ -f "$model" ]] || { printf 'missing model: %s\n' "$model" >&2; exit 1; }
[[ -f "$keys" ]] || { printf 'missing keyring: %s\n' "$keys" >&2; exit 1; }
[[ -f "$metrics_token_file" ]] || {
    printf 'missing metrics token: %s\n' "$metrics_token_file" >&2
    exit 1
}
metrics_token=$(<"$metrics_token_file")
[[ -n "$metrics_token" ]] || { printf 'empty metrics token\n' >&2; exit 1; }

export PATH=/root/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

exec env \
    -u MEMRA_API_KEY \
    -u MEMRA_ALLOW_OPEN_BIND \
    -u MEMRA_MODEL_METADATA \
    -u MEMRA_REQUEST_LEDGER \
    -u MEMRA_PP_STAGES \
    -u MEMRA_PP_DEVICES \
    -u MEMRA_DUAL_PP \
    -u MEMRA_PP_OVERLAP \
    -u MEMRA_PP_HOST_BOUNCE \
    -u MEMRA_PRIME_PIPE \
    -u MEMRA_SERVE_BATCH \
    -u MEMRA_SPEC_K \
    -u MEMRA_SPEC_GATE \
    -u MEMRA_DECODE_BATCH_CAP \
    -u MEMRA_FAST \
    -u MEMRA_MOE_RESIDENT \
    -u MEMRA_MOE_RESIDENT_GB \
    CUDA_VISIBLE_DEVICES=0 \
    MEMRA_MODELS="qwen/qwen3.6-27b=$model" \
    MEMRA_ADDR=127.0.0.1:8002 \
    MEMRA_COMPAT=openai \
    MEMRA_TAG=cx-servetest-q27 \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_CTX=8192 \
    MEMRA_PREFIX_CACHE_MB=4096 \
    MEMRA_PREFIX_DEDUP=1 \
    MEMRA_REUSE_POOL=0 \
    MEMRA_AFFINITY=0 \
    MEMRA_MAX_SESSIONS=96 \
    MEMRA_API_KEYS="$keys" \
    MEMRA_METRICS_TOKEN="$metrics_token" \
    "$server"
