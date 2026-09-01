#!/usr/bin/env bash
# MEMRA_MTP_SKIP refusal teeth: execute every FATAL path and quote its stderr
# (loud-failures law: a failure path that was never executed is not a tooth).
# usage: refusal-teeth.sh <memra-server-bin> <q38.gguf> <ownhead-fixture.gguf> <ranks.txt> <evidence_dir>
set -uo pipefail
BIN=$1; MODEL=$2; FIXTURE=$3; RANKS=$4; EV=$5
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18131}
mkdir -p "$EV"
: > /tmp/mtp-skip-gate/empty-ranks.txt
FAIL=0

arm() { # $1 name  $2 expect-substr  $3... env pairs
    local name=$1 expect=$2; shift 2
    local log="$EV/refusal-$name.log"
    # A refused boot exits on its own; timeout is a backstop, not the assertion.
    timeout 300 flock -w 600 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
        MEMRA_COMPAT=openai "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=4096 \
        "$@" "$BIN" >"$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "FAIL[$name]: server exited 0; the refusal did not fire"; FAIL=1; return
    fi
    if grep -qF "$expect" "$log"; then
        echo "PASS[$name]: exit=$rc, stderr carries the cause:"
        grep -F "$expect" "$log" | head -2 | sed 's/^/    /'
    else
        echo "FAIL[$name]: exit=$rc but expected cause missing (want: $expect)"; FAIL=1
        tail -5 "$log" | sed 's/^/    /'
    fi
}

arm garbage-value "MEMRA_MTP_SKIP=\"2\": expected 1" \
    "MEMRA_MODELS=gate=$MODEL" MEMRA_MTP_SKIP=2
arm contradictory-mtp-draft "together with MEMRA_MTP_DRAFT is contradictory" \
    "MEMRA_MODELS=gate=$MODEL" MEMRA_MTP_SKIP=1 \
    "MEMRA_MTP_DRAFT=/data/ai-ml/models/q38-gguf/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
arm empty-d2t "yields an EMPTY d2t list" \
    "MEMRA_MODELS=gate=$MODEL" MEMRA_MTP_SKIP=1 \
    MEMRA_FRSPEC_TRIM=/tmp/mtp-skip-gate/empty-ranks.txt
arm own-head-artifact "ships its own MTP-block lm_head" \
    "MEMRA_MODELS=gate=$FIXTURE" MEMRA_MTP_SKIP=1 "MEMRA_FRSPEC_TRIM=$RANKS"

if [ "$FAIL" = 0 ]; then echo "refusal-teeth: ALL GREEN"; else echo "refusal-teeth: FAILED"; exit 1; fi
