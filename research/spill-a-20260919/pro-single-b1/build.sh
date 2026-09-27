#!/usr/bin/env bash
# Design B1's target-card build (DAY59.md section 7), outside any hold, on the box's clone of this lane (/root/wt-a).
# From ONE clone: b1 (the tip's release server and its engine and server lib test binaries), base (the tip's crates
# taken to B1's parent: the per-plane program), red (the tip plus red-arm.patch: the kernel skips each item's last
# byte; its engine and server lib test binaries only). Test executables are found through cargo's JSON `executable`
# field and copied beside the receipts; the tree is checked back at the tip after each arm.
# usage: build.sh <tip_sha> <base_sha>
set -uo pipefail
R=/root/spill-receipts/a-b1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/b1" "$R/bins/base" "$R/bins/red"
L=$R/build-steps.log
[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git /root/wt-a >> "$L" 2>&1
cd /root/wt-a || exit 1
git fetch -q origin lane/spill-a-20260919 >> "$L" 2>&1
git checkout -q -B lane-a-b1 "$1" >> "$L" 2>&1 || { echo "rc=2 (checkout tip)" >> "$R/build.log"; exit 2; }
TIP=$(git rev-parse HEAD); echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-base.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
tests() { # $1 dest dir: the engine and server lib test executables
  for pkg in memra-engine memra-server; do
    nice -n 5 cargo test -p $pkg --lib --no-run --message-format=json 2>>"$L" | python3 -c "
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
echo "== b1" >> "$L"
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (b1)" >> "$R/build.log"; exit 1; }
cp target/release/memra-server "$R/bins/b1/memra-server"
tests "$R/bins/b1" || { echo "rc=1 (b1 tests)" >> "$R/build.log"; exit 1; }
echo "== red" >> "$L"
cp research/spill-a-20260919/pro-single-b1/red-arm.patch "$R/red-arm.patch"
git apply --index "$R/red-arm.patch" >> "$L" 2>&1 || { echo "rc=2 (red patch)" >> "$R/build.log"; exit 2; }
tests "$R/bins/red"; trc=$?
git reset -q; git checkout -q "$TIP" -- crates; clean || { echo "rc=2 (tree after red)" >> "$R/build.log"; exit 2; }
[ $trc -eq 0 ] || { echo "rc=1 (red tests)" >> "$R/build.log"; exit 1; }
echo "== base ($2)" >> "$L"
git diff --binary "$TIP" "$2" -- crates | git apply --index >> "$L" 2>&1 || { echo "rc=2 (base crates)" >> "$R/build.log"; exit 2; }
nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
cp target/release/memra-server "$R/bins/base/memra-server"
git reset -q; git checkout -q "$TIP" -- crates; git clean -q -fd crates; clean || { echo "rc=2 (tree after base)" >> "$R/build.log"; exit 2; }
[ $brc -eq 0 ] || { echo "rc=1 (base)" >> "$R/build.log"; exit 1; }
sha256sum "$R"/bins/*/memra-server "$R"/bins/*/memra-*-tests | tee "$R/binaries.sha256"
{ for b in b1 base; do echo "$b batch wording: $(grep -ac 'copy_batch_items_u8' "$R/bins/$b/memra-server")"; done
  echo "red marker: engine $(grep -ac 'b1 red arm' "$R/bins/red/memra-engine-tests") server $(grep -ac 'b1 red arm' "$R/bins/red/memra-server-tests")"
  echo "b1 tests marker (must be 0): $(grep -ac 'b1 red arm' "$R/bins/b1/memra-server-tests")"; } > "$R/markers.txt"
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; } > "$R/host-shape.txt" 2>&1
clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
echo "rc=0" >> "$R/build.log"
