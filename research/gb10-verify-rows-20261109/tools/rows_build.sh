#!/usr/bin/env bash
# Runs ON a GB10 host, unleased, only while no timed cell runs there (CPU only): lever 2's
# binaries (kernel-check, run-spec, memra-server, session-gate) at a commit delivered as a git
# bundle, banked as bins/rows-<sha10>. usage: rows_build.sh BUNDLE SHA
set -u
D=$HOME/.cache/memra-gb10-r4
BUNDLE=$1; SHA=$2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
L=$D/logs/rows-build.log
: > "$L"
git -C "$D/tree" fetch -q origin main || { echo ROWS-BUILD-FAIL fetch-main; exit 1; }
git -C "$D/tree" fetch -q "$BUNDLE" perf/fa-verify-rows || { echo ROWS-BUILD-FAIL fetch; exit 1; }
git -C "$D/tree" checkout -q --detach "$SHA" || { echo ROWS-BUILD-FAIL checkout; exit 1; }
(cd "$D/tree" && CARGO_TARGET_DIR=$D/target nice -n 10 cargo build --release --locked -j 16 \
  -p memra-engine --bin kernel-check --bin run-spec --bin session-gate -p memra-server --bin memra-server) >> "$L" 2>&1 \
  || { echo ROWS-BUILD-FAIL build; tail -n 30 "$L"; exit 1; }
out=$D/bins/rows-${SHA:0:10}; mkdir -p "$out"
cp -a "$D/target/release/kernel-check" "$D/target/release/run-spec" "$D/target/release/session-gate" "$D/target/release/memra-server" "$out/"
sha256sum "$out/memra-server"
echo ROWS-BUILD-DONE
