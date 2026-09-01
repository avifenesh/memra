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

# DEFAULT-FLIP MARKER, in BOTH directions, and it is the whole reason the flip battery is
# trustworthy. The flip arms carry NO door env at all -- the DEFAULT is the thing under test --
# so the environment cannot distinguish a flipped binary from an unflipped one, and an unflipped
# binary in a `flip-on` arm would measure the OFF program while every env assertion passed. That
# is the arm-identity-not-liveness failure exactly. `door_default_on()`'s WARN string exists only
# in a flipped binary, so it is the positive marker for the flip arms and the NEGATIVE marker for
# every pre-flip arm. Both directions were executed and observed to abort before any row was
# banked; a marker test that has only ever passed is not a control.
FLIPMARK="DEFAULT-ON answer is kept"
case "$MODE" in
  flip-*|gflip-*)
    grep -aq "$FLIPMARK" "$BIN" || { echo "ABORT: mode $MODE measures the DEFAULT, but this binary has no default-ON door (pre-flip build) — its 'no door env' arm would be the OFF program wearing the ON label"; exit 1; } ;;
  gate-main) ;;   # pre-restore binary: the door-name test above already pins it
  *)
    grep -aq "$FLIPMARK" "$BIN" && { echo "ABORT: mode $MODE arms doors EXPLICITLY and asserts the others ABSENT, but this binary defaults them ON — an 'absent' door here would be armed anyway"; exit 1; } ;;
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
  # ---- DOWN8 DEFAULT-FLIP QUALIFICATION (2026-09-01) --------------------------------------
  # INERTNESS PROBE, and it is why this battery is not the one the brief asked for. `down8` is
  # gated at its call site on `device_routed && sel_down8_on() && shard.slot_major && nsb<=32`,
  # and `shard.slot_major` is `ep2 || bank_slot_major_on()`. So arming DOWN8 while BANK_SM stays
  # off is a SILENT NO-OP on the TP serving path: no refusal, no warning, `down8=false door=true`
  # in the engagement line, and the +5.48% simply does not happen. This mode exists to make that
  # a RECEIPT rather than a code reading, on the pre-flip binary where the env is the axis.
  v3-d8only)       EXTRA="$CURRENT_EXTRA $D8" ;;
  # The actual deployable shape, and it was NEVER PRICED by milestone 4: that cell ran cumulative
  # arms only, so down8's +5.48% is the delta from `sm+gu` -> `sm+gu+d8`. `sm+d8` with GU OFF --
  # which is what "flip down8, leave the others OFF" actually deploys -- has no row anywhere.
  v3-sm-d8)        EXTRA="$CURRENT_EXTRA $SM $D8" ;;
  # THE FLIP ARMS, on a flipped binary. `flip-on` sets NO door env: the DEFAULT is the program
  # under test, which is the only shape that proves a default rather than a recipe. `flip-off` is
  # the ROLLBACK SEAM exercised as an arm -- a seam that has never been measured is not a seam.
  flip-on)         EXTRA="$CURRENT_EXTRA" ;;
  flip-off)        EXTRA="$CURRENT_EXTRA MEMRA_NVFP4_BANK_SM=0 MEMRA_NVFP4_SEL_DOWN8=0" ;;
  # THE SURGICAL SEAM (revuto finding, PR #76): down8 rolled back ALONE while the default keeps
  # the slot-major layout armed - the arm an operator reaches for under incident pressure. The
  # joint flip-off above proves both =0 parses but never this composition.
  flip-d8off)      EXTRA="$CURRENT_EXTRA MEMRA_NVFP4_SEL_DOWN8=0" ;;
  # byte/content-gate twins of the two flip arms (spec off, same reason as the gate-* modes).
  gflip-on)        EXTRA="$CURRENT_EXTRA"; SPEC=0 ;;
  gflip-off)       EXTRA="$CURRENT_EXTRA MEMRA_NVFP4_BANK_SM=0 MEMRA_NVFP4_SEL_DOWN8=0"; SPEC=0 ;;
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
