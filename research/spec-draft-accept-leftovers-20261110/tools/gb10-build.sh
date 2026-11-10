#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: build memra-server from ~/.cache/memra-1160/TREE
# (whose content the caller just synced) and store it as bin/memra-server-ARM.
# usage: gb10-build.sh ARM TREE
set -u
ARM=$1
TREE=$2
D=$HOME/.cache/memra-1160
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$D/target
cd "$D/$TREE" || exit 1
sha256sum crates/memra-engine/src/spec.rs
# A synced tree keeps older mtimes than the last build here: touch, or cargo calls it unchanged.
find crates -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.h' -o -name '*.toml' \) -exec touch {} +
date -Is
nice -n 10 cargo build --release --locked -j 12 -p memra-server --bin memra-server > "$D/build-$ARM.log" 2>&1
rc=$?
grep -E "Compiling memra|Finished|^error" "$D/build-$ARM.log" | tail -n 12
echo "cargo rc=$rc"
date -Is
[ $rc = 0 ] || exit 1
mkdir -p "$D/bin" && cp -a "$D/target/release/memra-server" "$D/bin/memra-server-$ARM" && sha256sum "$D/bin/memra-server-$ARM"
