#!/usr/bin/env bash
# Runs ON the GB10 box: build the two served binaries for the gate-fix verification, one at a
# time into the shared memra-batchspec-target:
#   default  = batch-spec (memset removal + spec-gate fix), tree ~/projects/memra-batchspec
#   combo    = research/combo-tc-gdnbf16 merged with it,    tree ~/projects/memra-combo
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-gate
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target
DEF=$(cat ~/projects/memra-batchspec/BATCHSPEC_SHA)
COMBO=$(cat ~/projects/memra-combo/COMBO_SHA)
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$T nice -n 5 cargo build --release -j 16 -p memra-server --bin memra-server > "$OUT/build-default.log" 2>&1
echo "BUILD=$?" >> "$OUT/build-default.log"
mkdir -p "$HOME/memra-run/bin-batchspec-$DEF"
cp "$T/release/memra-server" "$HOME/memra-run/bin-batchspec-$DEF/"
cd ~/projects/memra-combo
CARGO_TARGET_DIR=$T nice -n 5 cargo build --release -j 16 -p memra-server --bin memra-server > "$OUT/build-combo.log" 2>&1
echo "BUILD=$?" >> "$OUT/build-combo.log"
mkdir -p "$HOME/memra-run/bin-combo-$COMBO"
cp "$T/release/memra-server" "$HOME/memra-run/bin-combo-$COMBO/"
sha256sum "$HOME/memra-run/bin-batchspec-$DEF/memra-server" "$HOME/memra-run/bin-combo-$COMBO/memra-server" > "$OUT/bins.sha256"
echo "BUILDS_DONE default=$DEF combo=$COMBO"
