#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease), memra#1266 review round 1: the sm_121a
# memra-server of the round-1 fix tree (src2, unpacked once from src2-lite.tar.gz) into bins/F2.
# Main's binary (bins/M) is unchanged. Stops at CAP_MIN so it never outlives its lease; the next
# call resumes. usage: build_gb10_f2.sh CAP_MIN   -> bins/F2/memra-server, logs/build.log
set -u
R=<lane>; CAP=${1:-25}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
L=$R/logs/build.log; mkdir -p "$R/logs"
[ -x "$R/bins/F2/memra-server" ] && { echo "$(date -Is) BUILD-DONE F2 (already)" >> "$L"; exit 0; }
if [ ! -d "$R/src2/crates" ]; then mkdir -p "$R/src2" && tar -xzf "$R/src2-lite.tar.gz" -C "$R/src2" || exit 1; fi
cd "$R/src2" || exit 1
echo "== F2 $(date -Is)" >> "$L"
sha256sum crates/memra-engine/src/spec.rs crates/memra-server/src/worker.rs crates/memra-server/src/lib.rs | sed "s|$HOME|~|" >> "$L"
timeout -s INT $(( CAP * 60 - 60 )) nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
  || { echo "$(date -Is) BUILD-STOP F2 (rc=$?)" >> "$L"; exit 1; }
mkdir -p "$R/bins/F2"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/F2/"
sha256sum "$R/bins/F2/memra-server" | sed "s|$HOME|~|" >> "$L"
echo "$(date -Is) BUILD-DONE F2" >> "$L"
