#!/usr/bin/env bash
# Dev desk: the fold's whole CPU chain on the synced tree, in order: the checks (dd-checks-fold.sh
# LABEL), and when the suite, clippy and rustfmt are clean, the sm_120a release builds for the
# RTX 5090 cells (the n-choice accounting observer and memra-server), the red arms (dd-red.sh)
# and the whole memra-engine library suite.
# usage: dd-chain.sh CHECK_LABEL RED_LABEL
set -u
D=$HOME/.cache/memra-1304
C=$1; R=$2
L=$D/logs/fold
mkdir -p "$L" "$D/bins"
rm -f "$L/observer.done" "$L/chain.done"
bash "$D/dd-checks-fold.sh" "$C"
S=$L/$C/summary.txt
if grep -q '^server suite rc=0 ' "$S" && grep -q '^clippy rc=0 ' "$S" && grep -q '^rustfmt (touched files) rc=0' "$S"; then
  export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
  export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
  export CARGO_TARGET_DIR=$D/target-check
  for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
  cd "$D/tree" || exit 1
  sha256sum crates/memra-server/src/worker.rs | sed "s|$HOME|~|" > "$L/observer.log"
  taskset -c 14-31 nice -n 15 cargo build --release --locked -p memra-server --example choice_accounting_gate --bin memra-server -j 14 >> "$L/observer.log" 2>&1
  echo "build rc=$?" >> "$L/observer.log"
  cp -a "$CARGO_TARGET_DIR/release/examples/choice_accounting_gate" "$CARGO_TARGET_DIR/release/memra-server" "$D/bins/" && sha256sum "$D/bins/choice_accounting_gate" "$D/bins/memra-server" | sed "s|$HOME|~|" >> "$L/observer.log"
  touch "$L/observer.done"
  bash "$D/dd-red.sh" "$R"
  taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 > "$L/$C/engine-lib.log" 2>&1
  echo "engine lib suite rc=$? $(grep -E '^test result' "$L/$C/engine-lib.log" | tail -n 1)" >> "$S"
else
  echo "checks not clean: builds and red arms not run" >> "$S"
fi
touch "$L/chain.done"
