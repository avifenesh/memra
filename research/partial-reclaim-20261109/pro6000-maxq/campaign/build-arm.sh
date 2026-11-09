#!/bin/bash
# Partial-reclaim lane (wrx): build one memra-server release binary.
#   build-arm.sh REF SHA
# Shallow, blobless, sparse clone under <lane>/repo (crates, tools, docs and the one
# research dir the sources include), one target dir for both arms so B rebuilds only what its diff touches.
# Pinned to cores 24-31,56-63 at nice 19 (the box's other lane owns the rest from 07:00).
set -u
. <kit>/env.sh
R=<lane>
export TMPDIR=$R/tmp
mkdir -p "$TMPDIR" "$R/bins" "$R/logs"
REF=$1; SHA=$2; short=${SHA:0:10}
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" >> "$R/status.txt"; }
if [ ! -d "$R/repo/.git" ]; then
  taskset -c 24-31,56-63 git clone -q --depth 1 --filter=blob:none --sparse --no-checkout -b "$REF" \
    https://github.com/avifenesh/memra.git "$R/repo" || { log "clone FAILED"; exit 2; }
  git -C "$R/repo" sparse-checkout set crates tools docs research/reasoning-schema-20260823 || { log "sparse FAILED"; exit 2; }
fi
taskset -c 24-31,56-63 git -C "$R/repo" fetch -q --depth 1 origin "$REF" || { log "fetch $REF FAILED"; exit 3; }
git -C "$R/repo" checkout -q --detach "$SHA" || { log "checkout $SHA FAILED"; exit 4; }
log "build $short start"
(cd "$R/repo" && CARGO_TARGET_DIR=$R/target MEMRA_CUDA_ARCH=120a taskset -c 24-31,56-63 nice -n 19 \
   cargo build --release --locked -j 16 -p memra-server --bin memra-server) > "$R/logs/build-$short.log" 2>&1 \
  || { log "build $short FAILED: $(grep -m3 '^error' "$R/logs/build-$short.log" | tr '\n' ' ')"; exit 5; }
cp "$R/target/release/memra-server" "$R/bins/memra-server-$short"
log "built $short $(sha256sum "$R/bins/memra-server-$short" | cut -c1-16)"
