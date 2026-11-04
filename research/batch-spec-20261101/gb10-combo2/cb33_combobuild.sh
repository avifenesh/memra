#!/usr/bin/env bash
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-combo2
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target
COMBO=$(cat ~/projects/memra-combo/COMBO_SHA)
cd ~/projects/memra-combo
CARGO_TARGET_DIR=$T nice -n 5 cargo build --release -j 16 -p memra-server --bin memra-server > "$OUT/build-combo.log" 2>&1
echo "BUILD=$?" >> "$OUT/build-combo.log"
mkdir -p "$HOME/memra-run/bin-combo-$COMBO"
cp "$T/release/memra-server" "$HOME/memra-run/bin-combo-$COMBO/"
echo "BUILDS_DONE combo=$COMBO"
