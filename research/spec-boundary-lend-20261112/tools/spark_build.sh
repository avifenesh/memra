#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease): memra#1221 part 2's sm_121a memra-server
# arms from one tree (src, unpacked once from src-lite.tar.gz: the branch without research/ except
# the files the crates include) and one target directory. An arm is the tree with arms/worker.ARM.rs
# (tools/arms_1221b.py). Stops at CAP_MIN (default 25) so it never outlives its lease; the next
# call resumes. usage: spark_build.sh CAP_MIN ARM...   -> bins/ARM-HEAD/memra-server
set -u
R=$HOME/.cache/memra-1221b; CAP=${1:-25}; shift
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
L=$R/logs/build.log; mkdir -p "$R/logs"
end=$(( $(date +%s) + CAP * 60 ))
H=$(cat "$R/head.txt")
if [ ! -d "$R/src/crates" ]; then mkdir -p "$R/src" && tar -xzf "$R/src-lite.tar.gz" -C "$R/src" || exit 1; fi
cd "$R/src" || exit 1
for a in "$@"; do
  OUT=$a-$H
  [ -x "$R/bins/$OUT/memra-server" ] && continue
  left=$(( end - $(date +%s) )); [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before $OUT" >> "$L"; exit 1; }
  cp "$R/arms/worker.$a.rs" crates/memra-server/src/worker.rs; touch crates/memra-server/src/worker.rs
  echo "== $OUT $(date -Is)" >> "$L"; sha256sum "$R/arms/worker.$a.rs" | sed "s|$HOME|~|" >> "$L"
  timeout -s INT "$left" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP $OUT (rc=$?)" >> "$L"; exit 1; }
  mkdir -p "$R/bins/$OUT"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$OUT/"
  sha256sum "$R/bins/$OUT/memra-server" | sed "s|$HOME|~|" >> "$L"
done
echo "$(date -Is) BUILD-DONE $*" >> "$L"
