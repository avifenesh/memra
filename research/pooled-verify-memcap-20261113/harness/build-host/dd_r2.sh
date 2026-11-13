#!/usr/bin/env bash
# CPU build host, memra#1266 review round 2: the new and changed tests, then clippy, on the synced tree.
set -u
R=<lane>
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-test
cd "$R/src" || exit 1
t() { echo "# $(date -Is) $*"; taskset -c 20-31 nice -n 15 ionice -c2 -n7 "$@"; echo "rc=$?"; }
{
  t cargo test --locked -j 12 -p memra-engine --lib pooled_verify_cap_1266
  t cargo test --locked -j 12 -p memra-server --lib pre_cohort_reclaim_1266
  t cargo test --locked -j 12 -p memra-server --lib vmm_pending_releases_keep_the_idle_wait_polling_and_the_reclaims_reap
  t cargo test --locked -j 12 -p memra-server --lib spec_newcomers_prime_before_cohorts_is_wired
  t bash tools/check-flags.sh
  t python3 tools/check-numeric-scope.py
  echo R2-DONE
} > "$R/r2.txt" 2>&1
