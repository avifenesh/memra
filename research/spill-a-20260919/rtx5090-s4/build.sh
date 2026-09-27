#!/usr/bin/env bash
# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from pro-single-s2/build.sh at a0f9968e3: exact replacements only.
# The design-S2 target-card build (DAY42.md section 1), outside any hold, on the box's clone of this lane (/root/wt-a,
# never another lane's tree). From ONE clone: s2 (the tip as built), g4 (the tip's crates taken to the lane before S2's
# code, G4 and T, equal to main 5d653e851), gpp (the crates taken to 358749c9f, G'', the hump clause's positive control),
# then the tip's test binaries; the tree checked back at the tip after each. usage: build.sh <tip_sha> <g4_sha> [gpp_sha]
set -uo pipefail
R=${A_OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/s2" "$R/bins/g4" "$R/bins/gpp"
L=$R/build-steps.log
GPP=${3:-358749c9f}
cd "${A_TREE:?}" || exit 1
git checkout -q --detach "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-g4.sha"; echo "$GPP" > "$R/tree-gpp.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== s2" >> "$L"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (s2)" >> "$R/build.log"; exit 1; }
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/s2/memra-server"
for arm in g4 gpp; do
  sha=$2; [ $arm = gpp ] && sha=$GPP
  echo "== $arm ($sha)" >> "$L"
  # The arm's crates exactly (files the tip added are removed too): the tip-to-arm crate diff, applied.
  git diff --binary "$TIP" "$sha" -- crates | git apply >> "$L" 2>&1 || { echo "rc=2 ($arm crates)" >> "$R/build.log"; exit 2; }
  systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
  cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/$arm/memra-server"
  git checkout -q "$TIP" -- crates; git reset -q; git clean -q -fd crates; clean || { echo "rc=2 (tree after $arm)" >> "$R/build.log"; exit 2; }
  [ $brc -eq 0 ] || { echo "rc=1 ($arm)" >> "$R/build.log"; exit 1; }
done
echo "== the tip's test binaries" >> "$L"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (s2 rebuild)" >> "$R/build.log"; exit 1; }
cmp -s "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/s2/memra-server" || echo "note: the s2 rebuild differs in bytes" >> "$L"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo test -p memra-server --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (server tests)" >> "$R/build.log"; exit 1; }
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo test -p memra-engine --lib --no-run >> "$L" 2>&1 || { echo "rc=1 (engine tests)" >> "$R/build.log"; exit 1; }
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo test -p memra-tier --test contracts --no-run >> "$L" 2>&1 || { echo "rc=1 (tier tests)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
for b in s2 g4 gpp; do echo "$b span-receipt-sealed wording: $(grep -ac 'span receipt sealed on the copy stream' "$R/bins/$b/memra-server") copy-stream-receipt-line: $(grep -ac 'receipts on the copy stream' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
