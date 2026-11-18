#!/usr/bin/env bash
# Runs ON unit-a inside this lane's sparklock lease (memra lane 1255), CPU only: memra-server
# for sm_121a of each arm named. tars/ARM.tar is `git archive` of the arm's commit without
# research/, tars/ARM-res.tar the research files the build and the cells read. One source tree
# (src/) is switched between arms with a checksum rsync that gives changed files a fresh mtime,
# so cargo rebuilds only what differs. Binary -> bins/ARM/memra-server (+ sha256).
# usage: unit-a-build.sh ARM...
set -u
R=$HOME/.cache/memra-1255/gb10
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
mkdir -p "$R/logs" "$R/bins" "$R/src" "$R/stage"
echo "$$" >> "$R/pids.txt"
rc=0
for arm in "$@"; do
  L=$R/logs/build-$arm.log
  if [ -x "$R/bins/$arm/memra-server" ] && grep -q "BUILD $arm rc=0" "$R/logs/builds.txt" 2>/dev/null; then
    echo "$arm already built" | tee -a "$R/logs/builds.txt"; continue
  fi
  S=$R/stage/$arm
  rm -rf "$S"; mkdir -p "$S"
  tar -x -C "$S" -f "$R/tars/$arm.tar" && tar -x -C "$S" -f "$R/tars/$arm-res.tar" || { rc=2; break; }
  rsync -rlp --checksum --delete --exclude target "$S/" "$R/src/" || { rc=2; break; }
  ( cd "$R/src" && nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server ) > "$L" 2>&1
  r=$?
  if [ $r = 0 ]; then
    mkdir -p "$R/bins/$arm"
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$arm/memra-server"
    cp -a "$R/src/research" "$R/bins/$arm/" 2>/dev/null
    sha256sum "$R/bins/$arm/memra-server" | sed "s|$HOME|~|" | tee -a "$R/logs/builds.txt"
  fi
  echo "BUILD $arm rc=$r $(date -Is) $(grep -E 'Finished|^error' "$L" | tail -n 1)" | tee -a "$R/logs/builds.txt"
  [ $r = 0 ] || { rc=$r; break; }
done
echo "BUILD-DONE rc=$rc $(date -Is)" | tee -a "$R/logs/builds.txt"
exit $rc
