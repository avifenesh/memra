#!/usr/bin/env bash
# Merge-forward gate battery (lane/glm5-tp2-fwd, 2026-08-31): the TP-2 seam merged onto
# the verify-batch + decode-diet bringup head. Every GPU step runs under the rig lock,
# TF32 off, exactness only (rig law). Capture-then-gate: each step logs to this directory
# and the battery stops at the first red.
set -u
cd "$(dirname "$0")/../../../.." || exit 70
OUT="research/glm53-flash-bringup-20260827/tp2-20260831/fwd-merge-gates"
fails=0

gate_test() { # gate_test <name>
  local name="$1"
  echo "########## cargo test --test $name -- --ignored ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
    timeout 3600 cargo test -p memra-engine --test "$name" -- --ignored --test-threads=1 \
    2>&1 | grep -v '^\[loader-law\]' | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "exit=$rc" | tee -a "$OUT/$name.log"
  if [ "$rc" -ne 0 ]; then fails=$((fails + 1)); echo "RED: $name"; exit "$rc"; fi
}

# Decode-diet door gates (each door's own bit-identity battery on the merged cores).
gate_test hc_fused_pre_gpu
gate_test hc_decode_ws_gpu
gate_test kda_fused_proj_bf16_gpu
gate_test kda_fused_proj_gpu
gate_test mla_decode_split_gpu

# Verify-batch gates (the batched walk over the TP-split cores).
gate_test glm5_verify_batch_gpu
gate_test glm5_tparallel_verify_gpu

# Standing glm5 suites.
gate_test glm5_spec_session_gpu
gate_test glm5_dflash_session_gpu
gate_test glm5_mtp_head_gpu
gate_test hyper_connections_gpu
gate_test glm5_kpool_indexer_gpu

echo "########## ALL cargo-test GATES GREEN ##########"
exit 0
