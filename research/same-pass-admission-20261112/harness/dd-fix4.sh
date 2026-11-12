#!/usr/bin/env bash
# Dev desk (memra#1255): CPU checks of the PR head rebased onto main 11b368f7d7 (the landing of
# memra#1259 changes crates/memra-engine/src/spec.rs and unified_memory.rs only). tree-fix4 =
# hardlinked copy of tree-fix3 with those two files (and the head's worker.rs) written by rsync.
# usage: dd-fix4.sh   (files already copied to $D/head-files/)
set -u
D=$HOME/.cache/memra-1255
echo "$$" >> "$D/pids.txt"
cd "$D" || exit 1
grep -q FIX3-DONE "$D/fix3.out" || { echo "fix3 run did not finish"; exit 1; }
[ -d tree-fix4 ] || cp -al tree-fix3 tree-fix4 || exit 1
rsync -c "$D/head-files/spec.rs" tree-fix4/crates/memra-engine/src/spec.rs
rsync -c "$D/head-files/unified_memory.rs" tree-fix4/crates/memra-engine/src/unified_memory.rs
rsync -c "$D/head-files/worker.rs" tree-fix4/crates/memra-server/src/worker.rs
sha256sum tree-fix4/crates/memra-engine/src/spec.rs tree-fix4/crates/memra-engine/src/unified_memory.rs tree-fix4/crates/memra-server/src/worker.rs
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-fix2-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/head
mkdir -p "$L"
cd "$D/tree-fix4" || exit 1
fail=0
step() {
  local name=$1; shift
  echo "== $name: $(date -Is)"
  nice -n 15 ionice -c2 -n7 "$@" > "$L/check-$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc"
  grep -E "^test result|^error|FAILED|panicked" "$L/check-$name.log" | sort | uniq -c | sort -rn | head -n 8
  [ $rc = 0 ] || fail=1
}
step fmt cargo fmt --all -- --check
step clippy cargo clippy --release --all-targets -j 14 -- -D warnings
step test-server cargo test --release -p memra-server -j 14
step flags bash tools/check-flags.sh
step numeric python3 tools/check-numeric-scope.py
echo "HEAD-CHECKS-DONE fail=$fail $(date -Is)"
