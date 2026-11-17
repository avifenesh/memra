#!/usr/bin/env bash
# memra#1299 (PR #1303) ring cell (2x RTX PRO 6000 Blackwell, PP-2, the IQ4_XS GGUF route of
# docs/models/step37-flash.md). Run on the box under the lease the coordinator names; this script
# takes no lock itself.
#
# Arms (sm_120a builds of concat-prime-probe, identical except hybrid_forward.rs; both carry
# walk-widths-patch.py, which makes the chunked prime walk print every call's width). Which pair a
# run used is in its guard.log (the binaries' sha256) and in the receipt's README.
#   - ring/bins: base = main bbc6f83aea, head = 7f44a7ef5e (PR #1303 on it).
#   - ring/bins2, the landing pair after main moved: base = main 63c351a2f4, head = 1ea13cce9e
#     (PR #1303 rebased onto it); these also carry widths-patch.py (the GEMM prime's calls).
#     The run of 2026-11-14 with this pair is void: the probe's tick arm built its cache on one
#     device and the second stage refused it before the continuation, in both arms.
#   - ring/bins3 and later: both arms carry the probe fix (the tick arm's cache comes from
#     pp::new_cache); the pair's commits are named in the receipt's README.
#   base: with a ring plan, PP-2's dynamic prime schedule gives calls wider than
#         PRIME_CHUNK_MAX_TOKENS (4,809 rows at t = 32,768); the ring holds the window plus one
#         call of at most 4,096 rows, so the append fails.
#   head: every pipeline range wider than the cap is split into parts within it.
#
# Cell: `concat-prime-probe <gguf> tickinv --prompt-a @ring32k.txt --budgets 0 --splits 64
# --steps 0` with MEMRA_STEP_GEMM_PRIME_SUFFIX=0. No decode steps: the cell is about the prime's
# calls, and the probe's plain decode loop is not a proven path on a pipeline (memra#1295). ring32k.txt is a prefix of
# research/e2e/prompts/p6-64k.txt that tokenizes to 32,832 tokens with BOS (mkring.py).
#   ref : one prime call of all T tokens at position 0 (the batched GEMM entry);
#   sp64: a 64-token call at position 0, then one call for the 32,768-token continuation at
#         position 64. With the suffix door off, the continuation takes the walk on the PP-2
#         dynamic schedule: the path of the issue.
#
# Order: the base arm alone first, under its own short limit (RING_BASE_TIMEOUT). It proves the
# vehicle: the probe must reach the continuation on the pair and fail there with the ring error.
# If the base arm ends any other way (it dies earlier, it exits 0, it runs into its limit), the
# cell stops there with exit 3 and the head arm is not run: the cards go back at once.
#
# PASS, written down before the run (ring-compare.py applies it):
#   - base: the probe exits non-zero, its log has `SWA ring append does not fit`, and the
#     `[prime-walk]` line of the continuation lists a call wider than 4,096 rows;
#   - head: the probe exits 0, the continuation's `[prime-walk]` line lists calls that are each
#     within 4,096 rows and at least 16, summing to the continuation's length, and the sp64 row
#     is printed (the continuation primed to its last row);
#   - the continuation is at least 27,916 tokens (the shortest length at which main's schedule
#     has a call over the ring's room).
# Reported, not judged: sp64 against ref on the head (EXACT or DIFFER, last-row maxdiff, argmax). The two arms of that comparison are two prime programs (the batched
# GEMM prime and the walk), which the suffix door's own record says are not byte-identical.
#
# usage: ring-check.sh GGUF_SHARD1 BIN_DIR OUT_DIR [DEVICES]
#   BIN_DIR   holds concat-prime-probe-{base,head}
#   DEVICES   CUDA_VISIBLE_DEVICES for the pair (default 2,3; ring-lease.sh passes the UUIDs)
#   RING_TIMEOUT (env) the head arm's bound, default 15m
#   RING_BASE_TIMEOUT (env) the base arm's bound, default 6m (never more than RING_TIMEOUT's
#   value when that is given in seconds, as the waiter gives it)
# exit: 0 PASS, 1 FAIL, 2 a file is missing, 3 the base arm did not prove the vehicle (head not run)
set -uo pipefail
GG=${1:?GGUF shard 1}; BIN=${2:?bin dir}; OUT=${3:?out dir}; DEV=${4:-2,3}
HERE=$(cd "$(dirname "$0")" && pwd)
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=$DEV NVIDIA_TF32_OVERRIDE=0 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off
export MEMRA_STEP_GEMM_PRIME_SUFFIX=0
mkdir -p "$OUT"
G=$OUT/guard.log
P=$HERE/ring32k.txt
{ date -Is; hostname; echo "CUDA_VISIBLE_DEVICES=$DEV MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off MEMRA_STEP_GEMM_PRIME_SUFFIX=0"
  sha256sum "$BIN"/concat-prime-probe-base "$BIN"/concat-prime-probe-head "$P"
  ls -la "$(dirname "$GG")"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.limit,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } > "$G" 2>&1
[ -f "$GG" ] || { echo "GGUF missing: $GG" | tee -a "$G"; exit 2; }
[ -f "$P" ] || { echo "prompt missing: $P" | tee -a "$G"; exit 2; }
run_arm() {  # run_arm ARM LIMIT
  local arm=$1 t0 r
  t0=$(date +%s)
  timeout "$2" "$BIN/concat-prime-probe-$arm" "$GG" tickinv --prompt-a "@$P" \
    --budgets 0 --splits 64 --steps 0 > "$OUT/$arm.log" 2>&1
  r=$?
  echo "$(date -Is) $arm probe rc=$r wall=$(( $(date +%s) - t0 ))s limit=$2 $(grep -m1 -o 'tickinv: T=[0-9]*' "$OUT/$arm.log")" | tee -a "$G"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$G" 2>&1
}
HEAD_LIMIT=${RING_TIMEOUT:-15m}
BASE_LIMIT=${RING_BASE_TIMEOUT:-6m}
case "$HEAD_LIMIT" in
  *[0-9]s) [ "${HEAD_LIMIT%s}" -lt 360 ] && [ -z "${RING_BASE_TIMEOUT:-}" ] && BASE_LIMIT=$HEAD_LIMIT ;;
esac
run_arm base "$BASE_LIMIT"
python3 "$HERE/ring-compare.py" "$OUT" --base-only | tee "$OUT/base-verdict.txt"
if [ "${PIPESTATUS[0]}" != 0 ]; then
  { cat "$OUT/base-verdict.txt"
    echo "the base arm did not reach the ring error at the continuation: the head arm was not run"
    echo "VERDICT VEHICLE-FAIL"; } > "$OUT/verdict.txt"
  tail -n 2 "$OUT/verdict.txt" | tee -a "$G"
  exit 3
fi
run_arm head "$HEAD_LIMIT"
python3 "$HERE/ring-compare.py" "$OUT" | tee "$OUT/verdict.txt"
exit "${PIPESTATUS[0]}"
