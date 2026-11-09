#!/bin/bash
# Steady-traffic cell (#1124), wrx: build A = main fa0ee88acf and B = the same main with #1124 merged in the working
# tree (no commit; its tree hash is printed and must equal the rig's merge ffb2b4b909^{tree}). One worktree and one
# target dir under the kit, nice 19, pinned to cores 24-31,56-63.
set -u
. <kit>/env.sh
R=$K/reclaim
mkdir -p "$R/bins" "$R/logs"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" >> "$R/status.txt"; }
cd "$REPO" || exit 2
[ -d "$R/wt" ] || git worktree add -q --detach "$R/wt" fa0ee88acf || { log "worktree FAILED"; exit 2; }
build() { # label
  (cd "$R/wt" && CARGO_TARGET_DIR=$R/target MEMRA_CUDA_ARCH=120a taskset -c 24-31,56-63 nice -n 19 \
     cargo build --release --locked -j 16 -p memra-server --bin memra-server) > "$R/logs/build-$1.log" 2>&1 \
    || { log "build $1 FAILED: $(grep -m3 '^error' "$R/logs/build-$1.log" | tr '\n' ' ')"; exit 3; }
  cp "$R/target/release/memra-server" "$R/bins/memra-server-$1"
  log "built $1 $(sha256sum "$R/bins/memra-server-$1" | cut -c1-16)"
}
log "build A start"; build fa0ee88acf
git -C "$R/wt" merge --no-ff --no-commit origin/fix/partial-reclaim-discrete > "$R/logs/merge.log" 2>&1 || { log "merge FAILED"; exit 4; }
log "merge tree $(git -C "$R/wt" write-tree)"
log "build B start"; build merge-fa0ee88acf-3448243319
log "builds done"
