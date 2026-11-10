#!/usr/bin/env bash
# Runs ON a GB10 host, unleased, only while no timed cell runs there (CPU only): origin/main's
# memra-server, banked as bins/main-<sha10>. usage: main_build.sh SHA
set -u
D=$HOME/.cache/memra-gb10-r4
SHA=$1
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
L=$D/logs/main-build.log
: > "$L"
git -C "$D/tree" fetch -q origin main || { echo MAIN-BUILD-FAIL fetch; exit 1; }
git -C "$D/tree" checkout -q --detach "$SHA" || { echo MAIN-BUILD-FAIL checkout; exit 1; }
(cd "$D/tree" && CARGO_TARGET_DIR=$D/target nice -n 10 cargo build --release --locked -j 16 \
  -p memra-server --bin memra-server) >> "$L" 2>&1 || { echo MAIN-BUILD-FAIL build; tail -n 20 "$L"; exit 1; }
out=$D/bins/main-${SHA:0:10}; mkdir -p "$out"
cp -a "$D/target/release/memra-server" "$out/"
sha256sum "$out/memra-server"
echo MAIN-BUILD-DONE
