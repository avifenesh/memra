#!/usr/bin/env bash
# Runs ON a GB10 host inside a sparklock lease (memra#1307), CPU only: memra-server for sm_121a of
# each arm named. The tree ~/.cache/memra-1307/tree holds the head's crates; an arm is that tree
# with the engine files from ~/.cache/memra-1307/arms/ARM/ (spec.rs, dflash.rs, unified_memory.rs,
# dsv4_gpu.rs, and lib.rs in the arms that carry it: the base or the head, each with the [diag1307]
# lines of diag1307.py), so the arms
# differ only in the change under test. Binary -> ~/.cache/memra-1307/bins/ARM/memra-server, with
# the arm files' sha256 beside it; an arm already built from the same files is skipped.
# usage: build1307.sh ARM...
set -u
R=$HOME/.cache/memra-1307
T=$R/tree
S=$T/crates/memra-engine/src
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
mkdir -p "$R/logs"
echo "$$" >> "$R/pids.txt"
cd "$T" || exit 1
rc=0
for arm in "$@"; do
  A=$R/arms/$arm/crates/memra-engine/src; B=$R/bins/$arm
  F=$(cd "$A" && ls spec.rs dflash.rs unified_memory.rs dsv4_gpu.rs lib.rs 2>/dev/null | tr '\n' ' ')
  want=$(cd "$A" && sha256sum $F)
  if [ -x "$B/memra-server" ] && [ "$(cat "$B/arm.sha" 2>/dev/null)" = "$want" ]; then
    echo "$arm already built"; continue
  fi
  for f in $F; do cmp -s "$A/$f" "$S/$f" || cp "$A/$f" "$S/$f"; done
  nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server > "$R/logs/build-$arm.log" 2>&1
  r=$?
  echo "$arm build rc=$r $(grep -E 'Finished|^error' "$R/logs/build-$arm.log" | tail -n 1)"
  if [ $r = 0 ]; then
    mkdir -p "$B"
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$B/memra-server"
    echo "$want" > "$B/arm.sha"
    sha256sum "$B/memra-server" | sed "s|$HOME|~|"
  else
    rc=$r; break
  fi
done
echo "BUILD-1307-DONE rc=$rc $(date -Is)"
exit $rc
