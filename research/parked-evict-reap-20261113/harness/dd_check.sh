#!/usr/bin/env bash
# CPU build host, memra#1278 (fix/1278-parked-evict-reap rebased on main bbc6f83aea, with the
# fold of the #1310 review). D is the lane's scratch dir: D/tree is the branch's working tree,
# D/main.worker.rs is main's worker.rs, D/red_arms.py this dir's patcher. Green: the lane's tests
# and the touched existing ones; the memra-server library suite; clippy (server, all targets,
# -D warnings); rustfmt on the touched Rust file. Then the red arms (red_arms.py), each restored
# after its run: the named test must fail with the quoted text. The green pass builds the release
# profile; the red arms rebuild the dev profile incrementally (one changed function each). Every
# cargo pinned to CPUs 14-31. The doc links, the public boundary and the support-records contract
# read git, so they run in the checkout (cpu/checkout-checks.log).
set -u
D=${D:?the lane scratch dir}
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
M=worker::parked_evict_reap_1278_tests
cd "$T" || exit 1
{ echo "head $(cat "$D/head.txt" 2>/dev/null)"; sha256sum "$W" | sed "s|$HOME|~|"; } > "$S"
ct() { taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j "$J" "$@"; }
ctd() { CARGO_INCREMENTAL=1 taskset -c 14-31 nice -n 15 cargo test -p memra-server --lib -j "$J" "$@"; }
one() {  # one LABEL FILTER [--exact]
  ct -- "${@:3}" "$2" > "$L/$1.log" 2>&1
  echo "$1 $2 rc=$? $(grep -E '^test result' "$L/$1.log" | tail -n 1)" | tee -a "$S"
  grep -A4 "panicked at" "$L/$1.log" | head -n 8 | cut -c1-240 >> "$S"
  grep -m3 -E '^error' "$L/$1.log" | cut -c1-200 >> "$S"
}
if [ "${SKIP_GREEN:-0}" != 1 ]; then
  one green-1278 parked_evict_reap_1278_tests
  grep -E "^test $M::.* ok$" "$L/green-1278.log" >> "$S"
  one green-census-1266 worker::tests::vmm_pending_releases_keep_the_idle_wait_polling_and_the_reclaims_reap --exact
  one green-retry worker::tests::cache_alloc_oom_reclaim_retries_exactly_once --exact
  one green-lend worker::tests::a_lend_refused_rewind_survives_an_out_of_memory_fresh_cache --exact
  one green-lender worker::tests::the_headroom_ladder_skips_a_parked_lender --exact
  one green-precohort pre_cohort_reclaim_1266
  # The climb's existing test (its name holds "reclaim_spare") and the kv-flex shed's wiring.
  one green-spare reclaim_spare
  one green-kvflex kv_flex
  ct > "$L/suite.log" 2>&1
  echo "server suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" | tee -a "$S"
  grep -E "^test .*FAILED$" "$L/suite.log" | head -n 20 >> "$S"
  taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-server --all-targets -j "$J" -- -D warnings > "$L/clippy.log" 2>&1
  echo "clippy rc=$?" | tee -a "$S"
  grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 5 >> "$S"
  rustfmt --edition 2024 --check crates/memra-server/src/worker.rs > "$L/fmt.log" 2>&1; echo "rustfmt (worker.rs) rc=$?" | tee -a "$S"
fi
if [ "${SKIP_RED:-0}" != 1 ]; then
  R=$L/red
  mkdir -p "$R"
  cp "$D/main.worker.rs" "$L/main.worker.rs"
  python3 "$D/red_arms.py" "$T" "$R" keep > "$R/keep.log" 2>&1 || { echo "KEEP-FAIL" | tee -a "$S"; exit 1; }
  # The fix itself in the red arms' profile: the nine tests pass before any arm runs.
  ctd -- parked_evict_reap_1278_tests > "$R/green-dev.log" 2>&1
  echo "red baseline (dev profile, the fix) rc=$? $(grep -E '^test result' "$R/green-dev.log" | tail -n 1)" | tee -a "$S"
  red() {  # red ARM TEST TEXT: TEST must fail and its panic must hold TEXT
    python3 "$D/red_arms.py" "$T" "$R" "$1" > "$R/$1.log" 2>&1 || { echo "red $1 PATCH-FAIL" | tee -a "$S"; python3 "$D/red_arms.py" "$T" "$R" restore >> "$R/$1.log" 2>&1; return; }
    ctd -- parked_evict_reap_1278_tests >> "$R/$1.log" 2>&1
    rc=$?
    named=$(grep -c "^test $M::$2 \.\.\. FAILED$" "$R/$1.log")
    text=$(grep -A8 -F -- "---- $M::$2 stdout ----" "$R/$1.log" | grep -c -F -- "$3")
    echo "red $1 rc=$rc $(grep -E '^test result' "$R/$1.log" | tail -n 1 | cut -c1-60) | $2 failed=$named | \"$3\" in its panic=$text" | tee -a "$S"
    grep -A8 -F -- "---- $M::$2 stdout ----" "$R/$1.log" | grep -m2 -E '^ +(left|right):|^assertion|lost, moved or new' | cut -c1-200 >> "$S"
    python3 "$D/red_arms.py" "$T" "$R" restore >> "$R/$1.log" 2>&1
  }
  C=every_eviction_a_reading_or_a_retry_counts_on_reaps_first
  ARMS=${ARMS:-all}
  want() { [ "$ARMS" = all ] || case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  want ladder-reap-off && red ladder-reap-off the_parked_ladder_stops_after_the_one_eviction_its_reap_funds "on_demand=true: one eviction funds the arrival"
  want climb-reap-off && red climb-reap-off the_restorable_climb_stops_after_the_one_eviction_its_reap_funds "the parked-session ladder ran out"
  want retry-reap-off && red retry-reap-off a_reclaim_retry_gets_the_evicted_bytes_after_its_reap "CUDA_ERROR_OUT_OF_MEMORY"
  want retry-rule-off && red retry-rule-off a_reclaim_retry_runs_on_the_bytes_its_reap_alone_released "CUDA_ERROR_OUT_OF_MEMORY"
  want retry-trim-off && red retry-trim-off a_reclaim_retry_maps_the_flushed_prefix_blocks_after_its_trim "on_demand=true"
  want flush-reap-off && red flush-reap-off a_failed_materialization_flush_is_reaped_before_the_cold_allocation "the cold allocation evicted no parked session"
  want rung-reap-off && red rung-reap-off a_failed_right_size_rung_is_reaped_before_the_next_one "on_demand=true: the second rung lands"
  want reap-line-always && red reap-line-always a_reap_that_released_nothing_prints_no_line_outside_the_teardowns "reap (reclaim-on-defer) released=0"
  want main-worker && red main-worker "$C" "lost, moved or new: reclaim_parked_ladder("
  want ladder-site-off && red ladder-site-off "$C" "lost, moved or new: pool.is_some() }, || { vmm_reap_for(\"reclaim-on-defer\");"
  want drain-reap-off && red drain-reap-off "$C" "lost, moved or new: oom_teardown_fence(&engine, &loaded); vmm_reap_for(\"admission-drain\");"
  want settle-reap-off && red settle-reap-off "$C" "lost, moved or new: synchronize_model_devices(engine, loaded, why); let reaped = vmm_reap_for(why);"
  want spec-miss-reap-off && red spec-miss-reap-off "$C" "lost, moved or new: let released = vmm_retry_reap(engine, loaded, &first_err, \"spec pool miss\");"
  want flush-site-off && red flush-site-off "$C" "lost, moved or new: let n = flush_then_reap( || px.evict_all(),"
  want park-compact-fail-off && red park-compact-fail-off "$C" "lost, moved or new: reused = None; vmm_reap_trim_for(engine, loaded, \"park-compact grow failed\");"
  want affinity-fail-drop-off && red affinity-fail-drop-off "$C" "lost, moved or new: drop(e); drop(ckpt); vmm_reap_trim_for(engine, loaded,"
  want right-size-site-off && red right-size-site-off "$C" "lost, moved or new: Err(_) => None, }, || { vmm_reap_trim_for(engine, loaded, \"spec right-size\");"
  want shed-reap-off && red shed-reap-off "$C" "lost, moved or new: self.floor as f64 / 1e6, ); vmm_reap_for(\"kv-flex shed\"); freed }"
  want probe-prefix-reread && red probe-prefix-reread "$C" "lost, moved or new: px.evict_all()"
  want probe-parked-reread && red probe-parked-reread "$C" "lost, moved or new: evict_oldest_parked_freeing("
  cmp -s "$W" "$R/fix.worker.rs" && echo "restored ok" | tee -a "$S" || echo "RESTORE MISMATCH" | tee -a "$S"
fi
echo "DONE $(date -Is)" | tee -a "$S"
