#!/usr/bin/env bash
# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from pro-single-th2/build.sh at 67af1b71e: exact replacements only.
# Design T-H prime's target-card build (DAY65.md sections 6 and 7; T-H's build of section 1), outside any hold, on the box's clone of this lane (/root/wt-a).
# From ONE clone: th (the tip's release server) and base (the tip's crates taken to T-H's parent: the one-thread helper). The tree is checked
# back at the tip after base. usage: build.sh <tip_sha> <base_sha>
set -uo pipefail
R=${A_OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/th" "$R/bins/base"
L=$R/build-steps.log
cd "${A_TREE:?}" || exit 1
git checkout -q --detach "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-base.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
echo "== th" >> "$L"
systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=12G nice -n 19 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (th)" >> "$R/build.log"; exit 1; }
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/th/memra-server"
echo "== base ($2)" >> "$L"
git diff --binary "$TIP" "$2" -- crates | git apply --index >> "$L" 2>&1 || { echo "rc=2 (base crates)" >> "$R/build.log"; exit 2; }
systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=12G nice -n 19 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/base/memra-server"
git reset -q; git checkout -q "$TIP" -- crates; git clean -q -fd crates; clean || { echo "rc=2 (tree after base)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (base)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server | tee "$R/binaries.sha256"
for b in th base; do echo "$b threads wording: $(grep -ac 'staged; {} threads' "$R/bins/$b/memra-server") reserve wording: $(grep -ac '; reserve {} of {} staged' "$R/bins/$b/memra-server")"; done > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
