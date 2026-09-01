#!/usr/bin/env bash
# 140-era launcher: env copied EXACTLY from the step37 house-prod dark launch that the
# 140.6 seal ran under (serve_launch.sh of the 2026-08-29 lane; fingerprint
# memra-c9a617ca994b), loopback bind, this box's model path. Includes the two doors the
# 2026-08-29 incident later removed (MEMRA_NVFP4_BANK_V2, MEMRA_SEL_DOWN8) - faithful to
# the era that measured 140.6. No vision (armed 08-30, after the seal). models.toml is
# the 08-29 launch-day render (sampling defaults identical to current: temp 0.5 top_p 0.9).
set -euo pipefail
BIN=${1:?binary path}
NONCE=${2:?boot nonce}
MODEL=/data/models/step37-flash-nvfp4
PORT=18620
LOCK=/tmp/memra-ab37.lock
D=/home/ubuntu/toolchain-ab
grep -aq "checkpoint SWA restore refused" "$BIN" || { echo "ABORT: binary lacks ring-restore marker"; exit 1; }
ENVV="MEMRA_OPROJ_TAIL=1 MEMRA_DEV1_ROUTER=1 MEMRA_LEN_MIRROR_LAZY=1 MEMRA_ASYNC_CHAIN=8 MEMRA_SHEXP_OVERLAP=1 MEMRA_ROUTES_PRESTAGE=1 MEMRA_MOE_DIRECT=1 MEMRA_OPROJ_DIRECT=1 MEMRA_STEP_TP=0-44@0,1 MEMRA_STEP_TP_NATIVE_P2P=1 MEMRA_STEP_NVFP4_DEV_ROUTES=1 MEMRA_STEP_TP_DECODE_V2=1 MEMRA_STEP_TP_QKV_FUSED=1 MEMRA_BF16_MMV=1 MEMRA_STEP_TP_DEV_ROUTER=1 MEMRA_STEP_TP_DCW=1 MEMRA_RMS_BLOCK=1024 MEMRA_NVFP4_BANK_V2=1 MEMRA_SIG_EXPF_DEV=1 MEMRA_HEAD_SPLIT=1 MEMRA_FA_DCW_MEMSET=0 MEMRA_NO_LOCAL_SHADOW=1 MEMRA_FUSE_ROPE_APPEND=1 MEMRA_SEL_DOWN8=1 MEMRA_SEL_MIRROR=1 MEMRA_FA_COMBINE_S=1 MEMRA_MAX_CTX=262144"
POLICY="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
exec flock -n "$LOCK" env $ENVV $POLICY \
    BOOT_NONCE="$NONCE" \
    MEMRA_SERVE_SPEC=1 MEMRA_CTX=262144 MEMRA_AFFINITY=1 \
    "MEMRA_MODELS=stepfun/step-3.7-flash=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" \
    "MEMRA_MODEL_METADATA=$D/models-140era.toml" \
    "$BIN"
