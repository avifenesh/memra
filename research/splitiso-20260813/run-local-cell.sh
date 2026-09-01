#!/usr/bin/env bash
# Local-5090 binding for the shared exactness cell. The base runner owns all fail-closed checks.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
GPU_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d ' ')

exec env \
    SPLITISO_ROOT="$REPO/research/splitiso-20260813/local-run" \
    SPLITISO_REPO="$REPO" \
    SPLITISO_SERVER="$REPO/target/release/memra-server" \
    SPLITISO_MODEL="${SPLITISO_MODEL:-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0-lcprestore-exact.gguf}" \
    SPLITISO_TMPDIR=/home/avifenesh/tmp-lanes \
    SPLITISO_GPU_PHYSICAL=0 \
    SPLITISO_GPU_UUID="$GPU_UUID" \
    SPLITISO_GPU_LOCK=/tmp/memra-5090.lock \
    SPLITISO_GLOBAL_LOCK= \
    SPLITISO_PORT="${SPLITISO_PORT:-18832}" \
    "$REPO/research/splitiso-20260813/run-box1-cell.sh"
