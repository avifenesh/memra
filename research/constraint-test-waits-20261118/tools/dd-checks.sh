#!/usr/bin/env bash
# CPU only (memra#1335): the two constraint compiler tests on main's test code and on the fixed test code, plain and
# under each induced delay of mutate.py; then the fixed tests 20 times, the whole memra-server library suite, clippy
# and fmt on the fixed tree. Every cargo pinned to CPUs 14-31 at nice 15, sm_120a build with CUDA 13.1.
# The tree is the fix's commit; base-main.rs and base-fix.rs are constrained.rs at main and at the fix.
# usage: dd-checks.sh LABEL   -> logs/LABEL/summary.txt (marker CHECKS-DONE)
set -u
D=$HOME/.cache/memra-1335
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
L=$D/logs/${1:-checks}
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
F=$T/crates/memra-server/src/constrained.rs
A=compiler_abandons_runaway_job_and_drains_next_job
B=compiler_refuses_at_cap_then_rearms_after_runaways_finish
cd "$T" || exit 1
run() { taskset -c 14-31 nice -n 15 "$@"; }
echo "start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$S"
sha256sum "$D/base-main.rs" "$D/base-fix.rs" | sed "s|$HOME|~|" >> "$S"
cell() {  # NAME BASE ARM
  cp "$D/base-$2.rs" "$F"
  if [ "$3" != none ]; then python3 "$D/mutate.py" "$3" "$F" > "$L/$1.mutate.log" 2>&1 || { echo "$1 ARM-NOT-APPLIED" >> "$S"; return; }; fi
  touch "$F"
  run cargo test --release --locked -j 14 -p memra-server --lib -- "$A" "$B" > "$L/$1.log" 2>&1
  echo "$1 rc=$? load $(cut -d' ' -f1 /proc/loadavg) $(grep -E '^test result' "$L/$1.log" | tail -n 1 | cut -d';' -f1-2); $(grep -E "^test .*($A|$B) \.\.\. " "$L/$1.log" | sed -E 's/^test constrained::tests::compiler_([a-z]+)_.* \.\.\. /\1=/' | tr '\n' ' '); $(grep -aE "did not|panicked" "$L/$1.log" | head -n 2 | tr '\n' ' ' | cut -c1-200)" >> "$S"
}
cell main-plain main none
cell main-late-start main late-start
cell main-late-dequeue main late-dequeue
cell fix-late-start fix late-start
cell fix-late-dequeue fix late-dequeue
cell fix-plain fix none
p=0
for i in $(seq 1 20); do
  run cargo test --release --locked -j 14 -p memra-server --lib -- "$A" "$B" > "$L/fix-repeat-$i.log" 2>&1 && p=$((p + 1))
done
echo "fix-plain repeated: $p of 20 passed; load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$S"
run cargo test --release --locked -j 14 -p memra-server --lib > "$L/suite.log" 2>&1
echo "server lib suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" >> "$S"
grep -E '^test .* \.\.\. FAILED$' "$L/suite.log" | head -n 10 >> "$S"
run cargo clippy --release --locked -j 14 -p memra-server --all-targets -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy (memra-server, all targets) rc=$? $(grep -cE '^error' "$L/clippy.log") error lines" >> "$S"
taskset -c 14-31 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" >> "$S"
cmp -s "$D/base-fix.rs" "$F" && echo "tree restored to the fix" >> "$S"
echo "CHECKS-DONE $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$S"
