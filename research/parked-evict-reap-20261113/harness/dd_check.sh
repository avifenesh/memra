#!/usr/bin/env bash
# CPU build host, memra#1278 (fix/1278-parked-evict-reap on main df1d23a764). TREE = main with the
# branch's changed files synced from the rig. Green: the four new tests and the touched existing
# ones; the memra-server library suite; clippy (server, all targets, -D warnings); fmt. Then the
# red arms (red_arms.py), each restored after its run. Every cargo pinned to CPUs 14-31.
set -u
D=<lane>
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=${LOGS:-$D/logs/checks}
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
W=$T/crates/memra-server/src/worker.rs
J=${JOBS:-10}
cd "$T" || exit 1
sha256sum "$W" | sed "s|$HOME|~|" > "$S"
ct() { taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j "$J" "$@"; }
one() {  # one LABEL FILTER [--exact]
  ct -- "${@:3}" "$2" > "$L/$1.log" 2>&1
  echo "$1 $2 rc=$? $(grep -E '^test result' "$L/$1.log" | tail -n 1)" | tee -a "$S"
  grep -A4 "panicked at" "$L/$1.log" | head -n 8 | cut -c1-240 >> "$S"
  grep -m3 -E '^error' "$L/$1.log" | cut -c1-200 >> "$S"
}
if [ "${SKIP_GREEN:-0}" != 1 ]; then
  one green-1278 parked_evict_reap_1278_tests
  one green-census-1266 worker::tests::vmm_pending_releases_keep_the_idle_wait_polling_and_the_reclaims_reap --exact
  one green-retry worker::tests::cache_alloc_oom_reclaim_retries_exactly_once --exact
  one green-lend worker::tests::a_lend_refused_rewind_survives_an_out_of_memory_fresh_cache --exact
  one green-lender worker::tests::the_headroom_ladder_skips_a_parked_lender --exact
  one green-precohort pre_cohort_reclaim_1266
  # The climb's existing test (its name holds "reclaim_spare").
  one green-spare reclaim_spare
  ct > "$L/suite.log" 2>&1
  echo "server suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" | tee -a "$S"
  grep -E "^test .*FAILED$" "$L/suite.log" | head -n 20 >> "$S"
  taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-server --all-targets -j "$J" -- -D warnings > "$L/clippy.log" 2>&1
  echo "clippy rc=$?" | tee -a "$S"
  grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 5 >> "$S"
  taskset -c 14-31 nice -n 15 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" | tee -a "$S"
fi
if [ "${SKIP_RED:-0}" != 1 ]; then
  R=$L/red
  mkdir -p "$R"
  python3 "$D/red_arms.py" "$T" "$R" keep > "$R/keep.log" 2>&1 || { echo "KEEP-FAIL" | tee -a "$S"; exit 1; }
  red() {  # red ARM FILTER
    python3 "$D/red_arms.py" "$T" "$R" "$1" > "$R/$1.log" 2>&1 || { echo "red $1 PATCH-FAIL" | tee -a "$S"; python3 "$D/red_arms.py" "$T" "$R" restore >> "$R/$1.log" 2>&1; return; }
    ct -- "$2" >> "$R/$1.log" 2>&1
    rc=$?
    echo "red $1 rc=$rc $(grep -E '^test result' "$R/$1.log" | tail -n 1) | $(grep -m1 -E 'panicked at|^error\[' "$R/$1.log" | cut -c1-160)" | tee -a "$S"
    grep -E '^test .*(FAILED|ok)$' "$R/$1.log" | head -n 8 >> "$S"
    python3 "$D/red_arms.py" "$T" "$R" restore >> "$R/$1.log" 2>&1
  }
  red ladder-reap-off parked_evict_reap_1278_tests
  red climb-reap-off parked_evict_reap_1278_tests
  red retry-reap-off parked_evict_reap_1278_tests
  red main-worker parked_evict_reap_1278_tests
  red ladder-site-off parked_evict_reap_1278_tests
  red drain-reap-off parked_evict_reap_1278_tests
  red settle-reap-off parked_evict_reap_1278_tests
  red spec-miss-reap-off parked_evict_reap_1278_tests
  cmp -s "$W" "$R/fix.worker.rs" && echo "restored ok" | tee -a "$S" || echo "RESTORE MISMATCH" | tee -a "$S"
fi
echo "DONE $(date -Is)" | tee -a "$S"
