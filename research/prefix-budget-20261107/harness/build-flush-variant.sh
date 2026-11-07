#!/usr/bin/env bash
# Measurement-only variant of the candidate: the admission reclaim flushes (as on main) while the
# budget, the live-free insert clamp and the victim order stay the candidate's. Built from the
# lane worktree with one line patched, then the line is restored; nothing here is committed.
set -u
W=<tree>
F=$W/crates/memra-server/src/worker.rs
cd "$W" || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty; refusing"; exit 2; }
python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
a = "                        let reclaim_short = if device_is_unified(&engine) {"
b = "                        let reclaim_short = if true || device_is_unified(&engine) {"
assert s.count(a) == 1
open(p, "w").write(s.replace(a, b))
EOF
export CARGO_TARGET_DIR=<lane>/target MEMRA_CUDA_ARCH=120a
flock ~/.cache/memra-rig-build.lock nice -n 15 cargo build --release -j 2 -p memra-server --bin memra-server 2>&1 | grep -E '^error|Finished' | tail -3
rc=${PIPESTATUS[0]}
git checkout -- crates/memra-server/src/worker.rs
[ "$rc" = 0 ] || { echo "BUILD FAILED rc=$rc"; exit "$rc"; }
cp "$CARGO_TARGET_DIR/release/memra-server" <bins>/memra-server-flushvariant-$(git rev-parse --short=10 HEAD)
sha256sum <bins>/memra-server-flushvariant-*
echo BUILD_DONE
