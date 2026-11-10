#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: build memra-server from ~/.cache/memra-carry/tree
# (whose content the caller just synced) and store it as bin/memra-server-ARM. The first call
# copies the base commit's binary and its cargo target directory from the earlier build tree of
# the same commit (read only there), so this build recompiles the memra crates and no dependency.
set -u
ARM=$1
D=$HOME/.cache/memra-carry
SRC=$HOME/.cache/memra-1186
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$D/target
mkdir -p "$D/bin"
if [ ! -e "$D/bin/memra-server-base" ] && [ -e "$SRC/bin/memra-server" ]; then
  cp -a "$SRC/bin/memra-server" "$D/bin/memra-server-base"
fi
if [ ! -d "$D/target" ] && [ -d "$SRC/target" ]; then
  cp -a "$SRC/target" "$D/target"
fi
sha256sum "$D/bin/memra-server-base"
cd "$D/tree" || exit 1
sha256sum crates/memra-engine/src/spec.rs crates/memra-server/src/worker.rs
# A synced tree keeps older mtimes than the last build here: touch, or cargo calls it unchanged.
find crates -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.h' -o -name '*.toml' \) -exec touch {} +
date -Is
nice -n 10 cargo build --release --locked -j 12 -p memra-server --bin memra-server > "$D/build-$ARM.log" 2>&1
rc=$?
grep -E "Compiling memra|Finished|^error" "$D/build-$ARM.log" | tail -n 12
echo "cargo rc=$rc"
date -Is
[ $rc = 0 ] || exit 1
cp -a "$D/target/release/memra-server" "$D/bin/memra-server-$ARM" && sha256sum "$D/bin/memra-server-$ARM"
df -h "$D" | tail -n 1
