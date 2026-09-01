#!/usr/bin/env bash
# SPEC-ENGAGEMENT DIAGNOSTIC (fast, shallow, UNTIMED). The cell-3 boot armed the spec route
# ("[glm5-spec] serve route ARMED: draft source = dflash2 @ b33c0347") and reported
# "[spec-gate] policy placement=single-or-non-pp2 LOW=2 HIGH=4 spec-admission=on", yet the
# verify walk NEVER RAN: 0 [glm5-acc] lines, 0 door T/X/K/W announces, 0 "verify walk BATCHED
# per layer", and W1K decode 24.03 tok/s against the PLAIN boot's 24.01 - a spec arm that
# measures exactly plain is a spec arm that is not engaging.
#
# Two candidate causes, differenced here before any 88-minute 1M prime is spent:
#   A  MEMRA_REUSE_POOL=0, which THIS WINDOW introduced (for a greedy+vendor 1M pair that has
#      since been dropped). It caps the DFlash2/dspark WHOLE-SESSION pool, and the boot logs
#      "[dspark] harvest=dflash ... dflash2=true", so a 0 cap may leave the DFlash2 arm with
#      nowhere to build its session. No prior spec battery set it.
#   B  auto-K (MEMRA_SPEC_K unset) choosing K=0 at concurrency 1 on this placement.
#
# Arms (each a fresh boot, one 16k real-corpus request, engagement read from the server log):
#   diag-poolvar-autok   REUSE_POOL left at DEFAULT, K auto      <- isolates cause A
#   diag-poolvar-k3      REUSE_POOL left at DEFAULT, K pinned 3  <- isolates cause B
# If A is the cause, arm 1 engages. If B is the cause, only arm 2 engages, and "the ship
# config at c=1 is plain by auto-K policy" becomes a finding in its own right.
set -uo pipefail
OUT=/root/out-1m
D=$OUT/receipts/c3diag
mkdir -p "$D"
SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 MEMRA_SPEC_STATS=1)
{
date -u +%FT%TZ
for arm in poolvar-autok poolvar-k3; do
  case $arm in
    poolvar-autok) EXTRA=(MEMRA_REUSE_POOL=2) ;;
    poolvar-k3)    EXTRA=(MEMRA_REUSE_POOL=2 MEMRA_SPEC_K=3) ;;
  esac
  echo; echo "######## ARM $arm extras=${SPEC[*]} ${EXTRA[*]} ########"
  bash "$OUT/serve.sh" start "diag-$arm" "${SPEC[@]}" "${EXTRA[@]}" || { echo "ARM $arm BOOTFAIL"; continue; }
  bash "$OUT/rung.sh" "diag-$arm" "D16K" 64400 128 greedy "$D/$arm" || echo "ARM $arm rung failed"
  echo "--- ENGAGEMENT VERDICT for $arm ---"
  L=$OUT/logs/boot-diag-$arm.log
  for pat in "\[glm5-acc\]" "verify walk BATCHED per layer" "\[glm5-vrows\]" "PMIN=0.700" \
             "\[bf16-tcols-wide\] engaged" "\[bf16-tcols-x1\] engaged" "\[topk-shards\] engaged" \
             "\[glm5-verify-ws\] engaged" "\[spec-k\]" "\[spec-gate\]"; do
    printf "  %-38s %s\n" "$pat" "$(grep -c -- "$pat" "$L")"
  done
  grep -E "\[spec-k\]|\[spec-gate\]" "$L" | head -3
  grep -E "\[glm5-acc\]" "$L" | tail -3
  bash "$OUT/serve.sh" stop
done
date -u +%FT%TZ
echo "SPECDIAG_DONE"
} 2>&1 | tee "$OUT/logs/c3diag.log"
