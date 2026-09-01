#!/usr/bin/env bash
# Gate 1b: run-gen greedy argmax MATCH, pre-change binary vs branch binary, flag unset.
# (run-gen's GGUF path loads without MTP, so this pins the loader refactor's byte identity
# on the plain decode surface; the serving-shape identity lives in the dspark smoke runs.)
# usage: rungen-match.sh <run-gen-base> <run-gen-branch> <q38.gguf> <evidence_dir>
set -uo pipefail
BASE=$1; BRANCH=$2; MODEL=$3; EV=$4
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
mkdir -p "$EV"
FAIL=0
P1='Write a Python function that parses an ISO-8601 timestamp and returns epoch seconds.'
P2='Describe, in three sentences, why tidal forces lock a moon to its planet.'
P3='List the shell commands to find the five largest files under /var and archive them.'

run_one() { # $1 bin  $2 tag  $3 prompt-index  $4 prompt
    local out="$EV/rungen-$2-p$3.log"
    flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_CHAT=1 MEMRA_NGEN=64 \
        "$1" "$MODEL" --prompt "$4" >"$out" 2>&1 || { echo "FAIL: rungen $2 p$3 died"; FAIL=1; }
    grep "^OUTPUT TEXT:" "$out" | sha256sum | cut -c1-16
}

i=1
for P in "$P1" "$P2" "$P3"; do
    A=$(run_one "$BASE" base "$i" "$P")
    B=$(run_one "$BRANCH" branch "$i" "$P")
    if [ -n "$A" ] && [ "$A" = "$B" ]; then
        echo "p$i MATCH ($A)"
    else
        echo "FAIL: p$i MISMATCH base=$A branch=$B"; FAIL=1
    fi
    i=$((i+1))
done
if [ "$FAIL" = 0 ]; then echo "rungen-match: ALL GREEN"; else echo "rungen-match: FAILED"; exit 1; fi
