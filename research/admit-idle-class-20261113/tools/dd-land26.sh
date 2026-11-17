#!/usr/bin/env bash
# Dev desk, CPU only: PR #1308's checks again on the branch rebased onto main 5a6ae2438e. tree =
# the head's tracked files, synced from the workstation and checked there against a sha256
# manifest. Checks: the memra-server library suite (both memra#1310 censuses are in it), the whole
# memra-engine library suite, clippy (engine and server, all targets, -D warnings), rustfmt on the
# touched files, tools/test_validation_plan.py and tools/test_choice_verifier.py.
# Every cargo pinned to CPUs 14-31.
# usage: dd-land26.sh LABEL   -> logs/fold/LABEL/summary.txt, logs/fold/LABEL.done
set -u
D=$HOME/.cache/memra-1304
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
LABEL=$1
L=$D/logs/fold/$LABEL
mkdir -p "$L"
rm -f "$D/logs/fold/$LABEL.done"
S=$L/summary.txt
cd "$D/tree" || exit 1
run() { taskset -c 14-31 nice -n 15 "$@"; }
census() { grep -E "^test .*($1) \.\.\. " "$L/suite.log" | head -n 2; }
{ echo "start $(date -Is)"; sha256sum crates/memra-server/src/worker.rs crates/memra-server/src/worker/host_glm.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs crates/memra-engine/src/hybrid_forward.rs crates/memra-engine/src/bin/dsv41_gate.rs | sed "s|$HOME|~|"; } > "$S"
run cargo test --release --locked -p memra-server --lib -j 14 > "$L/suite.log" 2>&1
echo "server suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" >> "$S"
grep -E "^test .*FAILED$" "$L/suite.log" | head -n 20 >> "$S"
census 'every_eviction_a_reading_or_a_retry_counts_on_reaps_first|vmm_pending_releases_keep_the_idle_wait_polling_and_the_reclaims_reap' >> "$S"
echo "server crates compiled: $(grep -cE '^ +Compiling memra-' "$L/suite.log")" >> "$S"
run cargo test --release --locked -p memra-engine --lib -j 14 > "$L/engine-lib.log" 2>&1
echo "engine lib suite rc=$? $(grep -E '^test result' "$L/engine-lib.log" | tail -n 1)" >> "$S"
grep -E "^test .*FAILED$" "$L/engine-lib.log" | head -n 20 >> "$S"
echo "engine dflash tests in it: $(grep -cE '^test .*dflash.* \.\.\. ok$' "$L/engine-lib.log") ok" >> "$S"
run cargo clippy --release --locked -p memra-engine -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$? $(grep -cE '^(warning|error)' "$L/clippy.log") warning/error lines" >> "$S"
grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 12 >> "$S"
rustfmt --edition 2024 --check crates/memra-server/src/worker.rs crates/memra-server/src/worker/host_glm.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs > "$L/fmt.log" 2>&1
echo "rustfmt (touched files) rc=$?" >> "$S"
run python3 tools/test_validation_plan.py > "$L/validation-plan.log" 2>&1
echo "tools/test_validation_plan.py rc=$? $(grep -E '^Ran [0-9]+ tests' "$L/validation-plan.log" | tail -n 1) $(tail -n 1 "$L/validation-plan.log")" >> "$S"
run python3 tools/test_choice_verifier.py > "$L/choice-verifier.log" 2>&1
echo "tools/test_choice_verifier.py rc=$? $(grep -E '^Ran [0-9]+ tests' "$L/choice-verifier.log" | tail -n 1) $(tail -n 1 "$L/choice-verifier.log")" >> "$S"
echo "done $(date -Is)" >> "$S"
touch "$D/logs/fold/$LABEL.done"
