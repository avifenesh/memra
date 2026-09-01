#!/usr/bin/env bash
# perf-chain launcher. The two base env lists are copied VERBATIM from the
# toolchain-ab-20260831 harness (launch-140era.sh = ERA_BASE + DOORS, launch.sh =
# ERA_BASE + the current deploy extras); this file only adds the env knobs cells 1-3
# need, each as a named MODE so every arm is auditable from its receipt.
#
#   era                    140-era serving env INCLUDING the two doors the 2026-08-29
#                          incident removed (MEMRA_NVFP4_BANK_V2, MEMRA_SEL_DOWN8).
#                          The v2-bank door CORRUPTS OUTPUT TEXT: this mode is for
#                          wall-clock pricing only and is never a serving configuration.
#   era-nodoors            Same env, both doors UNSET. This is the FIXED env of the commit
#                          bisect (cell 3): the only era-shaped env that boots on every
#                          commit in range, because memra >= 75bf4ce76 refuses to boot
#                          step37 with the v2-bank door set.
#   current                Current deploy-shape env (doors gone, vision armed).
#   current-novision       current with MEMRA_STEP_VISION_DIR unset (cell 2).
#   fixed-nofiltered       era-nodoors + MEMRA_SPEC_GRAPH_FILTERED=0.
#   fixed-nochaingraph     era-nodoors + MEMRA_MTP_CHAIN_GRAPH=0.
#   fixed-nodcw            era-nodoors + MEMRA_STEP35_DRAFT_DCW=0.
#   current-nofiltered     current + MEMRA_SPEC_GRAPH_FILTERED=0 (the prod rollback seam).
#
# models.toml is a byte copy of the deployment registry. Its vendor sampling defaults
# (temperature 0.5 / top_p 0.9) are what govern the no-sampling-params request shape the
# sealed digits protocol sends -- and top_p 0.9 makes that shape a FILTERED regime, which
# is why the *filtered* draft-graph door is on the suspect list. The registry's
# deny_unknown_fields struct field set is identical at both ends of the bisect range
# (diffed empty), so ONE registry parses on every commit under test.
set -euo pipefail
BIN=${1:?binary path}
NONCE=${2:?boot nonce}
MODE=${3:?mode}
MODEL=/data/models/step37-flash-nvfp4
PORT=18640
LOCK=/tmp/memra-pc37.lock
D=/home/ubuntu/perf-chain

# Binary identity gate carried over from both the era and the prod launcher.
grep -aq "checkpoint SWA restore refused" "$BIN" || { echo "ABORT: binary lacks ring-restore marker"; exit 1; }

ERA_BASE="MEMRA_OPROJ_TAIL=1 MEMRA_DEV1_ROUTER=1 MEMRA_LEN_MIRROR_LAZY=1 MEMRA_ASYNC_CHAIN=8 MEMRA_SHEXP_OVERLAP=1 MEMRA_ROUTES_PRESTAGE=1 MEMRA_MOE_DIRECT=1 MEMRA_OPROJ_DIRECT=1 MEMRA_STEP_TP=0-44@0,1 MEMRA_STEP_TP_NATIVE_P2P=1 MEMRA_STEP_NVFP4_DEV_ROUTES=1 MEMRA_STEP_TP_DECODE_V2=1 MEMRA_STEP_TP_QKV_FUSED=1 MEMRA_BF16_MMV=1 MEMRA_STEP_TP_DEV_ROUTER=1 MEMRA_STEP_TP_DCW=1 MEMRA_RMS_BLOCK=1024 MEMRA_SIG_EXPF_DEV=1 MEMRA_HEAD_SPLIT=1 MEMRA_FA_DCW_MEMSET=0 MEMRA_NO_LOCAL_SHADOW=1 MEMRA_FUSE_ROPE_APPEND=1 MEMRA_SEL_MIRROR=1 MEMRA_FA_COMBINE_S=1 MEMRA_MAX_CTX=262144"
DOORS="MEMRA_NVFP4_BANK_V2=1 MEMRA_SEL_DOWN8=1"
POLICY="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
CURRENT_EXTRA="MEMRA_SERVE_DEVPENALTY=0 MEMRA_MAX_SESSIONS=24 MEMRA_DRAIN_S=30"

ENVV="$ERA_BASE"; EXTRA=""
case "$MODE" in
  era)                ENVV="$ERA_BASE $DOORS" ;;
  era-nodoors)        : ;;
  fixed-nofiltered)   EXTRA="MEMRA_SPEC_GRAPH_FILTERED=0" ;;
  fixed-nochaingraph) EXTRA="MEMRA_MTP_CHAIN_GRAPH=0" ;;
  fixed-nodcw)        EXTRA="MEMRA_STEP35_DRAFT_DCW=0" ;;
  current)
    grep -aq "MEMRA_STEP_VISION_DIR" "$BIN" || { echo "ABORT: binary lacks step-vision marker"; exit 1; }
    EXTRA="MEMRA_STEP_VISION_DIR=$MODEL $CURRENT_EXTRA" ;;
  current-novision)   EXTRA="$CURRENT_EXTRA" ;;
  current-nofiltered)
    grep -aq "MEMRA_STEP_VISION_DIR" "$BIN" || { echo "ABORT: binary lacks step-vision marker"; exit 1; }
    EXTRA="MEMRA_STEP_VISION_DIR=$MODEL $CURRENT_EXTRA MEMRA_SPEC_GRAPH_FILTERED=0" ;;
  *) echo "ABORT: unknown mode $MODE"; exit 1 ;;
esac

exec flock -n "$LOCK" env $ENVV $POLICY $EXTRA \
    BOOT_NONCE="$NONCE" \
    PC_MODE="$MODE" \
    MEMRA_SERVE_SPEC=1 MEMRA_CTX=262144 MEMRA_AFFINITY=1 \
    "MEMRA_MODELS=stepfun/step-3.7-flash=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" \
    "MEMRA_MODEL_METADATA=$D/models.toml" \
    "$BIN"
