#!/usr/bin/env bash
# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from pro-single-l2/build.sh at 21984b527: exact replacements only.
# Design L' target-card build (DAY63.md section 4), outside any hold, on the box's clone of this lane (/root/wt-a).
# From ONE clone: l (the tip's release server and its engine and server lib test binaries), redgate (the tip plus
# gate-red-arm.patch: a refused span drops its staging buffer; the release server), base (the tip's crates taken to
# the parent: fresh leases and the staging set on the first demote), red (the tip plus red-arm.patch: a pooled
# backing keeps its capacity as its length; its engine and server lib test binaries only). Test executables are found through cargo's JSON `executable`
# field and copied beside the receipts; the tree is checked back at the tip after each arm.
# usage: build.sh <tip_sha> <base_sha>
set -uo pipefail
R=${A_OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/l" "$R/bins/base" "$R/bins/red" "$R/bins/redgate"
L=$R/build-steps.log
cd "${A_TREE:?}" || exit 1
git checkout -q --detach "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-base.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
tests() { # $1 dest dir: the engine and server lib test executables
  for pkg in memra-engine memra-server; do
    systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo test -p $pkg --lib --no-run --message-format=json 2>>"$L" | python3 -c "
import sys, json
for l in sys.stdin:
    try: j = json.loads(l)
    except Exception: continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['profile']['test'] and j['target']['kind'] == ['lib']:
        print(j['executable'])" > "$1/$pkg.path" || return 1
    [ -s "$1/$pkg.path" ] || return 1
    cp "$(head -1 "$1/$pkg.path")" "$1/$pkg-tests" || return 1
  done
}
echo "== l" >> "$L"
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (l)" >> "$R/build.log"; exit 1; }
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/l/memra-server"
tests "$R/bins/l" || { echo "rc=1 (l tests)" >> "$R/build.log"; exit 1; }
echo "== red" >> "$L"
cp research/spill-a-20260919/pro-single-l2/red-arm.patch "$R/red-arm.patch"
git apply --index "$R/red-arm.patch" >> "$L" 2>&1 || { echo "rc=2 (red patch)" >> "$R/build.log"; exit 2; }
tests "$R/bins/red"; trc=$?
git reset -q; git checkout -q "$TIP" -- crates; clean || { echo "rc=2 (tree after red)" >> "$R/build.log"; exit 2; }
[ $trc -eq 0 ] || { echo "rc=1 (red tests)" >> "$R/build.log"; exit 1; }
echo "== redgate (the gate change's red arm)" >> "$L"
cp research/spill-a-20260919/pro-single-l2/gate-red-arm.patch "$R/gate-red-arm.patch"
git apply --index "$R/gate-red-arm.patch" >> "$L" 2>&1 || { echo "rc=2 (gate red patch)" >> "$R/build.log"; exit 2; }
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1; grc=$?
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/redgate/memra-server"
git reset -q; git checkout -q "$TIP" -- crates; clean || { echo "rc=2 (tree after redgate)" >> "$R/build.log"; exit 2; }
[ $grc -eq 0 ] || { echo "rc=1 (redgate)" >> "$R/build.log"; exit 1; }
echo "== base ($2)" >> "$L"
git diff --binary "$TIP" "$2" -- crates | git apply --index >> "$L" 2>&1 || { echo "rc=2 (base crates)" >> "$R/build.log"; exit 2; }
systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp "${CARGO_TARGET_DIR:?}/release/memra-server" "$R/bins/base/memra-server"
git reset -q; git checkout -q "$TIP" -- crates; git clean -q -fd crates; clean || { echo "rc=2 (tree after base)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (base)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server "$R"/bins/*/memra-*-tests | tee "$R/binaries.sha256"
{ echo "red marker: engine $(grep -ac 'day63 red arm' "$R/bins/red/memra-engine-tests")"
  echo "l tests marker (must be 0): $(grep -ac 'day63 red arm' "$R/bins/l/memra-engine-tests")"
  echo "redgate marker: $(grep -ac 'day63 gate red arm' "$R/bins/redgate/memra-server") (l must be 0: $(grep -ac 'day63 gate red arm' "$R/bins/l/memra-server"))"
  for b in l base; do echo "$b pooled wording: $(grep -ac 'pinned, pooled ' "$R/bins/$b/memra-server") boot staging wording: $(grep -ac 'span staging set allocated at boot' "$R/bins/$b/memra-server")"; done; } > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
