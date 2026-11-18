#!/usr/bin/env bash
# CPU only (memra#1340, the review fold): red then green for the two folds, then the regression of the first run's
# arms on the folded test code, the fixed tests 20 times, the whole memra-server library suite, clippy and fmt.
# Every cargo pinned to CPUs 14-31 at nice 15, sm_120a build with CUDA 13.1. The tree is the fold's commit.
#   base-prefold.rs / base-fold.rs   constrained.rs before the fold (be6dda7ade) and at the fold
#   base-wmain.rs / base-wfold.rs    worker.rs at main and at the fold
# usage: dd-checks-fold.sh LABEL   -> logs/LABEL/summary.txt (marker CHECKS-DONE)
set -u
D=$HOME/.cache/memra-1335
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
L=$D/logs/${1:-fold}
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
FC=$T/crates/memra-server/src/constrained.rs
FW=$T/crates/memra-server/src/worker.rs
A=compiler_abandons_runaway_job_and_drains_next_job
B=compiler_refuses_at_cap_then_rearms_after_runaways_finish
C=slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress
cd "$T" || exit 1
run() { taskset -c 14-31 nice -n 15 "$@"; }
echo "start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$S"
sha256sum "$D/base-prefold.rs" "$D/base-fold.rs" "$D/base-wmain.rs" "$D/base-wfold.rs" | sed "s|$HOME|~|" >> "$S"
cell() {  # NAME CONSTRAINED_BASE WORKER_BASE ARM TARGET_FILE(c|w|-) TESTS...
  local name=$1 cb=$2 wb=$3 arm=$4 tf=$5; shift 5
  cp "$D/base-$cb.rs" "$FC"; cp "$D/base-$wb.rs" "$FW"
  if [ "$arm" != none ]; then
    local f=$FC; [ "$tf" = w ] && f=$FW
    python3 "$D/mutate.py" "$arm" "$f" > "$L/$name.mutate.log" 2>&1 || { echo "$name ARM-NOT-APPLIED" >> "$S"; return; }
  fi
  touch "$FC" "$FW"
  local t0=$SECONDS
  run cargo test --release --locked -j 14 -p memra-server --lib -- "$@" > "$L/$name.log" 2>&1
  echo "$name rc=$? load $(cut -d' ' -f1 /proc/loadavg) wall $((SECONDS - t0))s $(grep -E '^test result' "$L/$name.log" | tail -n 1 | cut -d';' -f1-2); $(grep -E "^test .*($A|$B|$C) \.\.\. " "$L/$name.log" | sed -E 's/^test [a-z_:]*::(compiler|slow)_([a-z]+)_.* \.\.\. /\2=/' | tr '\n' ' '); $(grep -aE "did not|was reported|panicked" "$L/$name.log" | head -n 2 | tr '\n' ' ' | cut -c1-230)" >> "$S"
}
# fold 1: the supervisor waits the wrong bound (production mutation)
cell prefold-wrong-deadline prefold wfold wrong-deadline c "$A" "$B"
cell fold-wrong-deadline fold wfold wrong-deadline c "$A" "$B"
# fold 2: the third test under a late first dequeue
cell worker-main-late-dequeue fold wmain worker-late-dequeue w "$C"
cell worker-fold-late-dequeue fold wfold worker-late-dequeue w "$C"
# the first run's arms again, on the folded test code
cell fold-late-start fold wfold late-start c "$A" "$B"
cell fold-late-dequeue fold wfold late-dequeue c "$A" "$B"
cell fold-plain fold wfold none - "$A" "$B" "$C"
p=0
for i in $(seq 1 20); do
  run cargo test --release --locked -j 14 -p memra-server --lib -- "$A" "$B" "$C" > "$L/fold-repeat-$i.log" 2>&1 && p=$((p + 1))
done
echo "fold-plain repeated (three tests): $p of 20 passed; load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$S"
run cargo test --release --locked -j 14 -p memra-server --lib > "$L/suite.log" 2>&1
echo "server lib suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" >> "$S"
grep -E '^test .* \.\.\. FAILED$' "$L/suite.log" | head -n 10 >> "$S"
run cargo clippy --release --locked -j 14 -p memra-server --all-targets -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy (memra-server, all targets) rc=$? $(grep -cE '^error' "$L/clippy.log") error lines" >> "$S"
taskset -c 14-31 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" >> "$S"
cmp -s "$D/base-fold.rs" "$FC" && cmp -s "$D/base-wfold.rs" "$FW" && echo "tree restored to the fold" >> "$S"
echo "CHECKS-DONE $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$S"
