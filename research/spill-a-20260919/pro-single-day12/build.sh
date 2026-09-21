#!/usr/bin/env bash
# Day 12 native build receipt: memra-server release twice from /root/wt-a. base = origin/main
# be07f2d36 (the tier before memra#384), fix = the lane tip (refs/bundle/a12, branch lane-a-day12).
# The tree is left on lane-a-day12 so the cells run the lane's gate script. No GPU cell.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/a-day12
out=$R/build
mkdir -p "$out" "$R/bins"
cd /root/wt-a
date -u +%FT%TZ > "$out/started.txt"
rustc --version > "$out/toolchain.txt"; nvcc --version | tail -1 >> "$out/toolchain.txt"
for arm in base fix; do
  if [ "$arm" = base ]; then
    git checkout -q --detach be07f2d36b66c89071433fb6d7d5a1fed3954f8b
  else
    git checkout -q lane-a-day12
  fi
  git rev-parse HEAD > "$out/source-$arm.txt"
  git status --porcelain > "$out/dirty-$arm.txt"
  nice -n 19 cargo build --release -p memra-server --offline -j 16 > "$out/build-$arm.log" 2>&1
  echo $? > "$out/exit-$arm"
  cp target/release/memra-server "$R/bins/memra-server-$arm"
  sha256sum "$R/bins/memra-server-$arm" > "$out/binary-$arm.sha256"
done
date -u +%FT%TZ > "$out/finished.txt"
echo "base exit $(cat "$out/exit-base") fix exit $(cat "$out/exit-fix")"
