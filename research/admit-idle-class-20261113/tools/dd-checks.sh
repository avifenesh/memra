#!/usr/bin/env bash
# Dev desk: memra#1304 (fix/1304-admit-class on main 01960ee1b1). tree = main with the branch's
# changed files synced from the rig. Checks: the three new/changed tests and the drafter wiring pin,
# the memra-server library suite, clippy (server, all targets), fmt; red arms with the tests kept:
#   R1 the idle booking restored to the whole growth -> an_idle_unified_box_books_no_prefix_growth
#   R2 the idle refusal restored to an unconditional 400 -> an_idle_refusal_is_retryable_...
#   R3 the reject branch answering EngineError::context_length again -> the_idle_refusal_is_...
# Every cargo pinned to CPUs 14-31.
set -u
D=$HOME/.cache/memra-1304
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
[ -d "$CARGO_TARGET_DIR" ] || cp -a "$HOME/.cache/memra-1255/target-1258-check" "$CARGO_TARGET_DIR"
L=$D/logs/checks
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
W=$T/crates/memra-server/src/worker.rs
cd "$T" || exit 1
sha256sum "$W" | sed "s|$HOME|~|" > "$S"
NEW="an_idle_unified_box_books_no_prefix_growth an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity the_idle_refusal_is_classified_against_the_post_load_capacity unified_memory_admission_books_the_prefix_caches_allowed_growth the_drafter_refusal_is_wired_from_the_gate_to_every_dflash_route"
one() {  # one LABEL TEST
  taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 -- --exact "worker::tests::$2" > "$L/$1-$2.log" 2>&1
  echo "$1 $2 rc=$? $(grep -E '^test result' "$L/$1-$2.log" | tail -n 1)" | tee -a "$S"
  grep -A4 "panicked at" "$L/$1-$2.log" | head -n 6 | cut -c1-240 >> "$S"
}
for t in $NEW; do one green "$t"; done
taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 > "$L/suite.log" 2>&1
echo "server suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" | tee -a "$S"
grep -E "^test .*FAILED$" "$L/suite.log" | head -n 10 >> "$S"
taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$?" | tee -a "$S"
grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 5 >> "$S"
taskset -c 14-31 nice -n 15 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" | tee -a "$S"
cp "$W" "$L/worker.keep"
red() {  # red LABEL TEST OLD NEW
  python3 - "$W" "$3" "$4" <<'EOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
assert s.count(old) == 1, (old, s.count(old))
open(p, "w").write(s.replace(old, new))
EOF
  one "$1" "$2"
  cp "$L/worker.keep" "$W"
  cmp "$W" "$L/worker.keep" && echo "$1 restored" | tee -a "$S"
}
red R1 an_idle_unified_box_books_no_prefix_growth "    if unified && !idle {" "    let _ = idle;
    if unified {"
red R2 an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity "    if fits_capacity {
        return EngineError::overloaded(" "    if fits_capacity && false {
        return EngineError::overloaded("
red R3 the_idle_refusal_is_classified_against_the_post_load_capacity "                            fail_request(req, idle_refusal(admission_cap, fits_capacity, largest));" "                            let _ = (fits_capacity, largest);
                            fail_request(req, EngineError::context_length(format!(\"request context {admission_cap} does not fit\")));"
echo "1304-CHECKS-DONE $(date -Is)" | tee -a "$S"
