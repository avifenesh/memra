#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: build memra-server from ~/.cache/memra-1186/tree
# (whose content the caller just synced) and store it as bin/memra-server.
set -u
D=$HOME/.cache/memra-1186
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$D/target
cd "$D/tree" || exit 1
sha256sum crates/memra-engine/src/spec.rs crates/memra-engine/cu/spec_sample.cu
# A synced tree keeps older mtimes than the last build here: touch, or cargo calls it unchanged.
find crates -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.h' -o -name '*.toml' \) -exec touch {} +
date -Is
nice -n 10 cargo build --release --locked -j 12 -p memra-server --bin memra-server > "$D/build.log" 2>&1
rc=$?
grep -E "Compiling memra|Finished|^error" "$D/build.log" | tail -n 12
echo "cargo rc=$rc"
date -Is
[ $rc = 0 ] || exit 1
mkdir -p "$D/bin" && cp -a "$D/target/release/memra-server" "$D/bin/memra-server" && sha256sum "$D/bin/memra-server"
