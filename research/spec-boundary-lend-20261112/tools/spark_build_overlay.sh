#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease): one memra-server built from the head
# tree (src-head, target-head; spark_build_head.sh unpacked it) with arms/worker.ARM.rs over its
# worker.rs, which is put back from the tree's tarball afterwards. Stops at CAP_MIN.
# usage: spark_build_overlay.sh CAP_MIN ARM HEADTAG   -> bins/ARM-HEADTAG/memra-server
set -u
R=$HOME/.cache/memra-1221b; CAP=${1:-25}; ARM=$2; TAG=$3
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target-head
OUT=$R/bins/$ARM-$TAG; L=$R/logs/build-$ARM-$TAG.log; mkdir -p "$OUT"
cd "$R/src-head" || exit 1
W=crates/memra-server/src/worker.rs
cp "$R/arms/worker.$ARM.rs" "$W"; touch "$W"
{ echo "== $ARM-$TAG $(date -Is)"; sha256sum "$R/arms/worker.$ARM.rs" | sed "s|$HOME|~|"; } >> "$L"
timeout -s INT $(( CAP * 60 )) nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1
rc=$?
tar -xzf "$R/$TAG.tar.gz" -C "$R/src-head" "$W" && touch "$W"
[ $rc = 0 ] || { echo "$(date -Is) BUILD-STOP (rc=$rc)" >> "$L"; exit 1; }
cp -a "$CARGO_TARGET_DIR/release/memra-server" "$OUT/"
sha256sum "$OUT/memra-server" | sed "s|$HOME|~|" >> "$L"
echo "$(date -Is) BUILD-DONE $ARM-$TAG" >> "$L"
