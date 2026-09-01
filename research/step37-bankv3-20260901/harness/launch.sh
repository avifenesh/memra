#!/usr/bin/env bash
# bank-v3 launcher. ERA_BASE / POLICY / CURRENT_EXTRA are copied VERBATIM from
# research/perf-chain-20260831/harness/launch.sh, which took them from the
# toolchain-ab-20260831 harness, which took them from the 140-era serving `agentic8.sh`
# ENVV line. Not retyped: the whole point of pricing against that env is that it is the
# same env, and a re-derived list is a different measurement wearing the same name.
#
# What this file adds is the FOUR CUMULATIVE ARMS of milestone 4. The 2026-08-29 removal
# deleted three programs behind one env var, so the perf chain could only price the bundle
# (-21.5% wall / -23.7% decode). These arms turn that into three numbers:
#
#   v3-off        no bank door.                        The baseline: current main's program.
#   v3-sm         + MEMRA_NVFP4_BANK_SM=1              PROGRAM 1: slot-major banks + _sel_v2.
#   v3-sm-gu      + MEMRA_NVFP4_SEL_GU=1               PROGRAM 2: fused gate+up sweep.
#   v3-sm-gu-d8   + MEMRA_NVFP4_SEL_DOWN8=1            PROGRAM 3: fused down+combine.
#
# Cumulative, not one-at-a-time, because programs 2 and 3 read slot-major rows and are
# INERT without program 1 — a one-at-a-time sweep would price two no-ops. Each arm's
# contribution is the delta to the arm below it, and `boot.sh`'s expectation table proves
# from /proc which doors are actually set, so "armed" is never merely intended.
#
# THE OLD NAMES ARE NEVER SET BY ANY ARM. MEMRA_NVFP4_BANK_V2 / MEMRA_SEL_DOWN8 are refused
# at boot by design (worker.rs `removed_bank_v2_doors_refusal`); setting them here would
# only produce a boot failure, and boot.sh asserts their absence in every mode.
set -euo pipefail
BIN=${1:?binary path}
NONCE=${2:?boot nonce}
MODE=${3:?mode}
MODEL=/data/models/step37-flash-nvfp4
PORT=18640
LOCK=/tmp/memra-bv3.lock
D=/home/ubuntu/bankv3/lane

# Binary identity gate carried over from the era and prod launchers.
grep -aq "checkpoint SWA restore refused" "$BIN" || { echo "ABORT: binary lacks ring-restore marker"; exit 1; }
# ARM-IDENTITY MARKER for this lane, in BOTH directions. A binary that predates milestone 3
# has none of the three door names in its strings, so it cannot be an arm of a door cell no
# matter what the environment says; and the `gate-main` arm below must be exactly such a
# binary, so for it the same test is INVERTED rather than skipped. Skipping a marker test for
# one arm is how an arm set quietly stops being what it claims — this keeps the check
# meaningful on both sides. ("arm identity binds on binary md5 + marker test".)
case "$MODE" in
  gate-main)
    for m in MEMRA_NVFP4_BANK_SM MEMRA_NVFP4_SEL_GU MEMRA_NVFP4_SEL_DOWN8; do
      grep -aq "$m" "$BIN" && { echo "ABORT: gate-main needs a PRE-milestone-3 binary but this one carries the $m door"; exit 1; }
    done ;;
  *)
    for m in MEMRA_NVFP4_BANK_SM MEMRA_NVFP4_SEL_GU MEMRA_NVFP4_SEL_DOWN8; do
      grep -aq "$m" "$BIN" || { echo "ABORT: binary lacks the $m door — not a milestone-3 binary"; exit 1; }
    done ;;
esac

ERA_BASE="MEMRA_OPROJ_TAIL=1 MEMRA_DEV1_ROUTER=1 MEMRA_LEN_MIRROR_LAZY=1 MEMRA_ASYNC_CHAIN=8 MEMRA_SHEXP_OVERLAP=1 MEMRA_ROUTES_PRESTAGE=1 MEMRA_MOE_DIRECT=1 MEMRA_OPROJ_DIRECT=1 MEMRA_STEP_TP=0-44@0,1 MEMRA_STEP_TP_NATIVE_P2P=1 MEMRA_STEP_NVFP4_DEV_ROUTES=1 MEMRA_STEP_TP_DECODE_V2=1 MEMRA_STEP_TP_QKV_FUSED=1 MEMRA_BF16_MMV=1 MEMRA_STEP_TP_DEV_ROUTER=1 MEMRA_STEP_TP_DCW=1 MEMRA_RMS_BLOCK=1024 MEMRA_SIG_EXPF_DEV=1 MEMRA_HEAD_SPLIT=1 MEMRA_FA_DCW_MEMSET=0 MEMRA_NO_LOCAL_SHADOW=1 MEMRA_FUSE_ROPE_APPEND=1 MEMRA_SEL_MIRROR=1 MEMRA_FA_COMBINE_S=1 MEMRA_MAX_CTX=262144"
POLICY="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
CURRENT_EXTRA="MEMRA_SERVE_DEVPENALTY=0 MEMRA_MAX_SESSIONS=24 MEMRA_DRAIN_S=30"

# The three restored doors, cumulative.
SM="MEMRA_NVFP4_BANK_SM=1"
GU="MEMRA_NVFP4_SEL_GU=1"
D8="MEMRA_NVFP4_SEL_DOWN8=1"

# SPEC: pricing arms serve the real shape (spec on, K=3, the serving policy above). The
# BYTE-GATE arms turn spec OFF, because a speculative path makes the greedy tape depend on
# draft/verify scheduling and the gate would be measuring the scheduler, not the bank. That
# is a gate-craft choice, not a shortcut: the byte gate's job is the bank's arithmetic.
SPEC=1
EXTRA=""
case "$MODE" in
  v3-off)          EXTRA="$CURRENT_EXTRA" ;;
  v3-sm)           EXTRA="$CURRENT_EXTRA $SM" ;;
  v3-sm-gu)        EXTRA="$CURRENT_EXTRA $SM $GU" ;;
  v3-sm-gu-d8)     EXTRA="$CURRENT_EXTRA $SM $GU $D8" ;;
  # byte/content-gate modes: spec off, otherwise identical to the pricing arm of the same name
  gate-off)        EXTRA="$CURRENT_EXTRA"; SPEC=0 ;;
  gate-sm)         EXTRA="$CURRENT_EXTRA $SM"; SPEC=0 ;;
  gate-sm-gu)      EXTRA="$CURRENT_EXTRA $SM $GU"; SPEC=0 ;;
  gate-sm-gu-d8)   EXTRA="$CURRENT_EXTRA $SM $GU $D8"; SPEC=0 ;;
  # GATE (b0): the PRE-RESTORE binary (d3ac87f80), same env as gate-off. This is what closes
  # the one gap the perf-CI neutrality argument leaves open — the four-arm byte gate proves the
  # arms agree with EACH OTHER, and the FLAGS rows additionally claim "OFF = byte-for-byte the
  # current serving path". That is a claim about main, so it is measured against main.
  gate-main)       EXTRA="$CURRENT_EXTRA"; SPEC=0 ;;
  *) echo "ABORT: unknown mode $MODE"; exit 1 ;;
esac

exec flock -n "$LOCK" env $ERA_BASE $POLICY $EXTRA \
    BOOT_NONCE="$NONCE" \
    BV3_MODE="$MODE" \
    MEMRA_SERVE_SPEC=$SPEC MEMRA_CTX=262144 MEMRA_AFFINITY=1 \
    "MEMRA_MODELS=stepfun/step-3.7-flash=$MODEL" \
    "MEMRA_ADDR=127.0.0.1:$PORT" \
    "MEMRA_MODEL_METADATA=$D/harness/models.toml" \
    "$BIN"
