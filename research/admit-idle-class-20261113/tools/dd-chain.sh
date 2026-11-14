#!/usr/bin/env bash
# Dev desk: the fold's whole CPU chain on the synced tree, in order: the checks (dd-checks-fold.sh
# LABEL), and when the suite, clippy and rustfmt are clean, the sm_120a release build of the
# n-choice accounting observer (for the RTX 5090 kv cell) and the red arms (dd-red.sh).
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
  taskset -c 14-31 nice -n 15 cargo build --release --locked -p memra-server --example choice_accounting_gate -j 14 >> "$L/observer.log" 2>&1
  echo "build rc=$?" >> "$L/observer.log"
  cp -a "$CARGO_TARGET_DIR/release/examples/choice_accounting_gate" "$D/bins/" && sha256sum "$D/bins/choice_accounting_gate" | sed "s|$HOME|~|" >> "$L/observer.log"
  touch "$L/observer.done"
  bash "$D/dd-red.sh" "$R"
else
  echo "checks not clean: observer and red arms not run" >> "$S"
fi
touch "$L/chain.done"
