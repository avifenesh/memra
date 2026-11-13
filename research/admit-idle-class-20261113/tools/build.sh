#!/usr/bin/env bash
# memra#1304, runs ON GB10 cb33 inside a sparklock lease, CPU only: memra-server (sm_121a) from
# tree-ARM for each ARM given (main = origin/main 01960ee1b1, head = the fix branch), one shared
# target dir seeded from a recent memra-server build on this unit.
# usage: build.sh ARM...   -> bins/ARM/memra-server
set -u
R=$HOME/.cache/memra-1304
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
mkdir -p "$R/logs" "$R/bins"
[ -d "$CARGO_TARGET_DIR" ] || cp -a "$HOME/.cache/memra-1258/target-cap" "$CARGO_TARGET_DIR"
for ARM in "$@"; do
  L=$R/logs/build-$ARM.log
  cd "$R/tree-$ARM" || exit 1
  sha256sum crates/memra-server/src/worker.rs | sed "s|$HOME|~|" > "$L"
  nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "BUILD-FAIL $ARM $(date -Is)" | tee -a "$L"; exit 1; }
  mkdir -p "$R/bins/$ARM"
  cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$ARM/"
  sha256sum "$R/bins/$ARM/memra-server" | sed "s|$HOME|~|" >> "$L"
  echo "BUILD-DONE $ARM $(date -Is)" | tee -a "$L"
done
