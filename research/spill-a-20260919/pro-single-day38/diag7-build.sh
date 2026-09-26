#!/usr/bin/env bash
# DAY38 section 13i, outside any hold: the bisection arms, each the sitting's tip plus one diag7-<arm>.patch, built into
# bins/<arm> with the tree taken back to the tip after each. usage: diag3-build.sh
set -uo pipefail
R=/root/spill-receipts/a-day38
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
TIP=$(cat "$R/tree-tip.sha")
L=$R/diag7-build.log
for arm in x25; do
  P=research/spill-a-20260919/pro-single-day38/diag7-$arm.patch
  mkdir -p "$R/bins/$arm"
  git apply "$P" >> "$L" 2>&1 || { echo "$arm rc=2 (patch)" >> "$L"; exit 2; }
  nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
  git diff --stat > "$R/bins/$arm/patch-applied.stat"
  git checkout -q -- crates
  [ "$(git rev-parse HEAD)" = "$TIP" ] && git diff --quiet || { echo "$arm rc=2 (tree not restored)" >> "$L"; exit 2; }
  [ $brc -eq 0 ] || { echo "$arm rc=1 (build)" >> "$L"; exit 1; }
  cp target/release/memra-server "$R/bins/$arm/memra-server"
  sha256sum "$R/bins/$arm/memra-server" > "$R/bins/$arm/memra-server.sha256"
  echo "$arm rc=0" >> "$L"
done
echo "rc=0" >> "$L"
