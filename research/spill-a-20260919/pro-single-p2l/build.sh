#!/usr/bin/env bash
# Design P2's re-read on L' (DAY67.md section 1): the target-card build, outside any hold, on the box's clone of this lane (/root/wt-a,
# never another lane's tree). From ONE clone: p2 (the tip as built), base (the tip's crates taken to step 1's commit,
# the publication split without a reserve), p (base's crates plus p-arm.patch: P of day 51 on the split lines, the
# chain cell's diagnostic arm), gpp (the crates taken to 358749c9f, G'', the hump clause's positive control), then the
# tip's test binaries; the tree checked back at the tip after each. usage: build.sh <tip_sha> <base_sha> [gpp_sha]
set -uo pipefail
R=/root/spill-receipts/a-p2l
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/p2" "$R/bins/base" "$R/bins/gpp"
L=$R/build-steps.log
GPP=${3:-358749c9f}
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a >> "$L" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 lane/spill-a-p2l-base-20260926 >> "$L" 2>&1
git checkout -q -B lane-a-p2l-tip "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-base.sha"; echo "$GPP" > "$R/tree-gpp.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== p2" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (p2)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/p2/memra-server"
for arm in base gpp; do
  sha=$2; [ $arm = gpp ] && sha=$GPP
  echo "== $arm ($sha)" >> "$L"
  # The arm's crates exactly (files the tip added are removed too): the tip-to-arm crate diff, applied to the index
  # and the tree.
  git diff --binary "$TIP" "$sha" -- crates | git apply --index >> "$L" 2>&1 || { echo "rc=2 ($arm crates)" >> "$R/build.log"; exit 2; }
  nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
  cp target/release/memra-server "$R/bins/$arm/memra-server"
  git reset -q; git checkout -q "$TIP" -- crates; git clean -q -fd crates; clean || { echo "rc=2 (tree after $arm)" >> "$R/build.log"; exit 2; }
  [ $brc -eq 0 ] || { echo "rc=1 ($arm)" >> "$R/build.log"; exit 1; }
done
echo "== the tip's test binaries" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (p2 rebuild)" >> "$R/build.log"; exit 1; }
cmp -s target/release/memra-server "$R/bins/p2/memra-server" || echo "note: the p2 rebuild differs in bytes" >> "$L"
nice -n 5 cargo test -p memra-server --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-engine --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
nice -n 5 cargo test -p memra-tier --test contracts --no-run >> "$L" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
for b in p2 base gpp; do echo "$b reserve-ready wording: $(grep -ac 'payload reserve ready: ' "$R/bins/$b/memra-server") arming wording: $(grep -ac ' fresh pages of ' "$R/bins/$b/memra-server") publication-split wording: $(grep -ac 'demote publication split: ticket seq=' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
