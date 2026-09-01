#!/usr/bin/env bash
# toolchain-ab launcher — byte-faithful port of the step37 prod serve launcher
# (darklanes deploy/step37/step37_serve_launch.sh) minus the metering seam
# (MEMRA_ADMIN_ADDR, MEMRA_ADMIN_TOKEN_FILE, MEMRA_API_KEYS, MEMRA_REQUEST_LEDGER,
# MEMRA_TENANT_BUDGETS omitted = the supported no-accounting shape), loopback bind.
# Same flag list, same order, same values. models.toml = byte copy of prod registry
# (vendor sampling defaults temp=0.5 top_p=0.9 govern the no-params request shape).
# BIN is the arm under test; BOOT_NONCE enters the server env for PID-verified
# arm identity via /proc/<pid>/environ.
set -euo pipefail
BIN=${1:?binary path}
NONCE=${2:?boot nonce}
MODEL=/data/models/step37-flash-nvfp4
PORT=18620
LOCK=/tmp/memra-ab37.lock
D=/home/ubuntu/toolchain-ab
grep -aq "checkpoint SWA restore refused" "$BIN" || { echo "ABORT: binary lacks ring-restore marker"; exit 1; }
grep -aq "MEMRA_STEP_GEMM_PRIME_SUFFIX" "$BIN" || { echo "ABORT: binary lacks suffix-door marker"; exit 1; }
grep -aq "MEMRA_STEP_VISION_DIR" "$BIN" || { echo "ABORT: binary lacks step-vision marker"; exit 1; }
exec flock -n "$LOCK" env \
    BOOT_NONCE="$NONCE" \
    MEMRA_OPROJ_TAIL=1 \
    MEMRA_DEV1_ROUTER=1 \
    MEMRA_LEN_MIRROR_LAZY=1 \
    MEMRA_ASYNC_CHAIN=8 \
    MEMRA_SHEXP_OVERLAP=1 \
    MEMRA_ROUTES_PRESTAGE=1 \
    MEMRA_MOE_DIRECT=1 \
    MEMRA_OPROJ_DIRECT=1 \
    MEMRA_STEP_TP=0-44@0,1 \
    MEMRA_STEP_TP_NATIVE_P2P=1 \
    MEMRA_STEP_NVFP4_DEV_ROUTES=1 \
    MEMRA_STEP_TP_DECODE_V2=1 \
    MEMRA_STEP_TP_QKV_FUSED=1 \
    MEMRA_BF16_MMV=1 \
    MEMRA_STEP_TP_DEV_ROUTER=1 \
    MEMRA_STEP_TP_DCW=1 \
    MEMRA_RMS_BLOCK=1024 \
    MEMRA_SIG_EXPF_DEV=1 \
    MEMRA_HEAD_SPLIT=1 \
    MEMRA_FA_DCW_MEMSET=0 \
    MEMRA_NO_LOCAL_SHADOW=1 \
    MEMRA_FUSE_ROPE_APPEND=1 \
    MEMRA_SEL_MIRROR=1 \
    MEMRA_FA_COMBINE_S=1 \
    MEMRA_MAX_CTX=262144 \
    MEMRA_LOAD_MTP=1 \
    MEMRA_MTP_HEADS=3 \
    MEMRA_SPEC_K=3 \
    MEMRA_SPEC_PMIN=0.5 \
    MEMRA_SPEC_PMIN0=1 \
    MEMRA_SERVE_SPEC=1 \
    MEMRA_CTX=262144 \
    MEMRA_AFFINITY=1 \
    "MEMRA_STEP_VISION_DIR=$MODEL" \
    MEMRA_SERVE_DEVPENALTY=0 \
    "MEMRA_MODELS=stepfun/step-3.7-flash=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" \
    "MEMRA_MODEL_METADATA=$D/models.toml" \
    MEMRA_MAX_SESSIONS=24 \
    MEMRA_DRAIN_S=30 \
    "$BIN"
