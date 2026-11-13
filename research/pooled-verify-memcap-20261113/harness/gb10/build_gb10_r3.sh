#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease), memra#1266 review round 2: arm F5 = the
# round-2 head (src5, unpacked from src5-lite.tar.gz) on main a17add18f3; main stays M3 (its
# crates are a17add18f3's). Stops at CAP_MIN; the next call resumes (cargo keeps its target).
# usage: build_gb10_r3.sh CAP_MIN   -> bins/F5/memra-server
set -u
R=<lane>; CAP=${1:-25}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
L=$R/logs/build.log; mkdir -p "$R/logs"
[ -x "$R/bins/F5/memra-server" ] && { echo "$(date -Is) BUILD-DONE F5 (already)" >> "$L"; exit 0; }
if [ ! -d "$R/src5/crates" ]; then mkdir -p "$R/src5" && tar -xzf "$R/src5-lite.tar.gz" -C "$R/src5" || exit 1; fi
cd "$R/src5" || exit 1
E=crates/memra-engine/src; V=crates/memra-server/src
echo "== F5 $(date -Is)" >> "$L"; sha256sum $E/spec.rs $E/lib.rs $V/worker.rs $V/lib.rs | sed "s|$HOME|~|" >> "$L"
timeout -s INT "$(( CAP * 60 - 30 ))" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
  || { echo "$(date -Is) BUILD-STOP F5 (rc=$?)" >> "$L"; exit 1; }
mkdir -p "$R/bins/F5"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/F5/"
sha256sum "$R/bins/F5/memra-server" | sed "s|$HOME|~|" >> "$L"
echo "$(date -Is) BUILD-DONE F5" >> "$L"
