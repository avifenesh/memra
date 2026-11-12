#!/usr/bin/env bash
# Runs ON a GB10 host (memra#1255), CPU only, inside a sparklock lease: the base tree's
# memra-server for sm_121a, then the fix arm (the base tree with the fix's two changed server
# sources), so only the server crate recompiles. Own scratch dir only.
# usage: spark-build.sh
set -u
R=$HOME/.cache/memra-1255
L=$R/logs/build.log; mkdir -p "$R/logs" "$R/bins"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
echo "$$" >> "$R/pids.txt"
cd "$R/tree" || exit 1
: > "$L"
for arm in base fix; do
  cp "$R/arms/$arm/worker.rs" crates/memra-server/src/worker.rs
  cp "$R/arms/$arm/lib.rs" crates/memra-server/src/lib.rs
  touch crates/memra-server/src/worker.rs crates/memra-server/src/lib.rs
  echo "== $arm $(date -Is) $(sha256sum crates/memra-server/src/worker.rs | cut -c1-16)" >> "$L"
  nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 || { echo "BUILD-FAIL $arm" >> "$L"; exit 1; }
  cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/memra-server-$arm-121a"
  sha256sum "$R/bins/memra-server-$arm-121a" >> "$L"
  strings "$R/bins/memra-server-$arm-121a" | grep -o -m1 'memra-0\.[0-9]*\.[0-9]*-[0-9a-f]\{12\}' >> "$L"
done
echo BUILD-DONE >> "$L"
