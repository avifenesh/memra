#!/usr/bin/env bash
# Dev desk: the review fold of memra#1327 in this lane's wt-trial checkout (sm_120a, CUDA 13.1, CPUs 14-31):
#   at REF: cargo fmt --check, workspace clippy -D warnings on all targets, memra-engine dsv41 lib tests;
#   at each mutant ref: the split MoE lib tests, which must fail (red arms).
# usage: dd_fold.sh TAG REF MUTANT_REF...   -> out/fold-TAG/{fmt,clippy,engine,red-*}.log, out/fold-TAG.done
set -u
W=$HOME/.cache/dsv41-cfix2
D=$W/wt-trial
TAG=$1; REF=$2; shift 2
O=$W/out/fold-$TAG
mkdir -p "$O"
rm -f "$W/out/fold-$TAG.done"
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$D/target
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
P="$W/out/fold-$TAG.part"; : > "$P"
git -C "$D" checkout -q --detach "$REF" || { echo "checkout failed; FOLD-DONE" > "$W/out/fold-$TAG.done"; exit 1; }
cd "$D" || exit 2
git log --oneline -1 >> "$P"
taskset -c 14-31 cargo fmt --all -- --check > "$O/fmt.log" 2>&1; echo "fmt rc=$?" >> "$P"
nice -n 10 taskset -c 14-31 cargo test --release -j 16 -p memra-engine --lib dsv41 > "$O/engine.log" 2>&1; echo "engine-dsv41 rc=$? $(grep -a '^test result' "$O/engine.log" | cut -c1-60)" >> "$P"
nice -n 10 taskset -c 14-31 cargo clippy --release -j 16 --all-targets --workspace -- -D warnings > "$O/clippy.log" 2>&1; echo "clippy rc=$?" >> "$P"
for m in "$@"; do
  git checkout -q --detach "$m" || { echo "$m checkout failed" >> "$P"; continue; }
  nice -n 10 taskset -c 14-31 cargo test --release -j 16 -p memra-engine --lib the_split_moe > "$O/red-$m.log" 2>&1
  echo "red $m ($(git log --oneline -1 | cut -c1-10)) rc=$? $(grep -a '^test result' "$O/red-$m.log" | cut -c1-60)" >> "$P"
  grep -a "^test dsv41\|panicked at\|^  left\|^ right\|assertion" "$O/red-$m.log" | cut -c1-200 | head -8 >> "$P"
done
git checkout -q --detach "$REF"
{ cat "$P"; echo "FOLD-DONE $(date +%T)"; } > "$W/out/fold-$TAG.done"
