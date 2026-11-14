#!/usr/bin/env bash
# Runs ON a GB10 host inside a sparklock lease (memra#1316), CPU only: memra-server for sm_121a
# from the tree ~/.cache/memra-1316/tree (the workstation rsyncs the branch's crates
# there), into ~/.cache/memra-1316/bins/ARM/.
#   ARM B     the red arm: arm_b.py sets the retry budget to zero in the tree's ctx_boot.rs for
#             this build (the context is created once, as before the change) and the file is
#             restored afterwards;
#   any other the tree as it is.
# The build stops at LIMIT (a duration for timeout(1)) so it never outlives the lease; a stopped
# build keeps its finished crates for the next call. The arm's memra-runtime sources are hashed
# next to the binaries.
# usage: build_unit.sh ARM LIMIT      (ends by creating jobs/build-ARM.done with the exit code)
set -u
ARM=$1; LIMIT=$2
R=$HOME/.cache/memra-1316
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
mkdir -p "$R/logs" "$R/bins/$ARM" "$R/jobs"
cd "$R/tree" || exit 1
F=crates/memra-runtime/src/ctx_boot.rs
if [ "$ARM" = B ]; then
  cp "$F" "$R/ctx_boot.rs.head"
  python3 "$R/research/gb10-boot-oom-20261114/tools/arm_b.py" "$F" || exit 1
fi
src=$(sha256sum crates/memra-runtime/src/*.rs)
# cargo decides by mtime: a file copied from the workstation or restored after arm B can be older
# than the last build, which would then be reused for the wrong arm.
touch "$F"
timeout "$LIMIT" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server \
  > "$R/logs/build-$ARM.log" 2>&1
rc=$?
[ "$ARM" = B ] && cp "$R/ctx_boot.rs.head" "$F"
echo "$ARM build rc=$rc $(grep -E 'Finished|^error' "$R/logs/build-$ARM.log" | tail -n 1)"
grep -q 'Compiling memra-runtime' "$R/logs/build-$ARM.log" || { echo "$ARM: memra-runtime was not rebuilt; refusing the binary"; rc=1; }
if [ $rc = 0 ]; then
  cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$ARM/"
  echo "$src" > "$R/bins/$ARM/runtime-src.sha"
  sha256sum "$R/bins/$ARM/memra-server" | sed "s|$HOME|~|"
fi
echo "$rc" > "$R/jobs/build-$ARM.done"
