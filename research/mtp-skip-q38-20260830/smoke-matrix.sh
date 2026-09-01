#!/usr/bin/env bash
# MEMRA_MTP_SKIP smoke matrix: three full tools/dspark-serve-smoke.sh runs on the q38
# production shape (dspark + FR-Spec trim), then cross-run byte identity + receipt parity.
#   A: branch binary, flag unset (default-OFF arm)
#   B: branch binary, MEMRA_MTP_SKIP=1 (skip arm)
#   C: pre-change binary @ origin/main (gate-1 baseline)
# usage: smoke-matrix.sh <repo_root> <base_bin> <branch_bin> <q38.gguf> <dflash2_dir> <ranks.txt> <evidence_dir>
set -uo pipefail
ROOT=$1; BASE=$2; BRANCH=$3; MODEL=$4; DRAFT=$5; RANKS=$6; EV=$7
export MEMRA_GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
export MEMRA_FRSPEC_TRIM="$RANKS"
mkdir -p "$EV"
FAIL=0

run_smoke() { # $1 tag  $2 bin  $3 skip-env(0|1)
    local tag=$1 bin=$2 skip=$3
    echo "===== smoke run $tag (skip=$skip) ====="
    if [ "$skip" = 1 ]; then export MEMRA_MTP_SKIP=1; else unset MEMRA_MTP_SKIP; fi
    bash "$ROOT/tools/dspark-serve-smoke.sh" "$MODEL" "$DRAFT" "$bin" "$EV/smoke-$tag" \
        > "$EV/smoke-$tag.out" 2>&1
    local rc=$?
    tail -3 "$EV/smoke-$tag.out"
    grep -hE "^\[mtp-skip\]|^\[frspec-trim\]|^\[dspark\]|^\[mtp-chain\]" \
        "$EV/smoke-$tag/serve-on.log" > "$EV/receipts-$tag-on.txt" || true
    grep -hE "^\[mtp-skip\]|^\[frspec-trim\]|^\[dspark\]|^\[mtp-chain\]" \
        "$EV/smoke-$tag/serve-off.log" > "$EV/receipts-$tag-off.txt" || true
    [ "$rc" = 0 ] || { echo "FAIL: smoke run $tag exited $rc"; FAIL=1; }
    unset MEMRA_MTP_SKIP
}

run_smoke A "$BRANCH" 0
run_smoke B "$BRANCH" 1
run_smoke C "$BASE" 0

h() { jq -r '(.choices[0].message.reasoning // "") + "" + (.choices[0].message.content // "")' "$1" | sha256sum | cut -c1-16; }

echo "===== cross-run byte identity (greedy rows) ====="
for i in 1 2 3; do
    for kind in on off; do
        A=$(h "$EV/smoke-A/$kind-r$i.json"); B=$(h "$EV/smoke-B/$kind-r$i.json"); C=$(h "$EV/smoke-C/$kind-r$i.json")
        if [ "$A" = "$B" ] && [ "$A" = "$C" ]; then
            echo "$kind-r$i IDENTICAL across A/B/C ($A)"
        else
            echo "FAIL: $kind-r$i diverged A=$A B=$B C=$C"; FAIL=1
        fi
    done
done

echo "===== boot receipt parity ====="
if diff -u "$EV/receipts-A-on.txt" "$EV/receipts-C-on.txt" > "$EV/receipts-A-vs-C.diff"; then
    echo "A-vs-C (flag unset, branch vs pre-change): boot receipts byte-identical"
else
    echo "FAIL: flag-unset boot receipts differ from the pre-change binary:"; FAIL=1
    cat "$EV/receipts-A-vs-C.diff"
fi
TRIM_A=$(grep -oE "TRIMMED to [0-9]+ rows" "$EV/receipts-A-on.txt" | head -1)
TRIM_B=$(grep -oE "TRIMMED to [0-9]+ rows" "$EV/receipts-B-on.txt" | head -1)
if [ -n "$TRIM_A" ] && [ "$TRIM_A" = "$TRIM_B" ]; then
    echo "dspark trim parity: '$TRIM_A' in both arms"
else
    echo "FAIL: dspark trim rows differ: A='$TRIM_A' B='$TRIM_B'"; FAIL=1
fi
if grep -q "^\[mtp-skip\] MEMRA_MTP_SKIP=1: skipping" "$EV/receipts-B-on.txt"; then
    echo "skip receipt (arm B): $(grep '^\[mtp-skip\] MEMRA_MTP_SKIP=1' "$EV/receipts-B-on.txt" | head -1)"
else
    echo "FAIL: arm B boot log carries no [mtp-skip] skip receipt"; FAIL=1
fi
if grep -q "^\[mtp-skip\]" "$EV/receipts-A-on.txt"; then
    echo "FAIL: arm A (flag unset) printed an [mtp-skip] line"; FAIL=1
fi

if [ "$FAIL" = 0 ]; then echo "smoke-matrix: ALL GREEN"; else echo "smoke-matrix: FAILED"; exit 1; fi
