#!/usr/bin/env bash
# Design W's 5090 builds (DAY61.md section 1), outside any hold, under the rig's CPU cap. w: the tip's release server
# and engine lib test executable (this worktree); red: the tip plus pro-single-w/red-arm.patch (engine lib test
# executable only), and base: W's parent's release server, each in its own scratch worktree beside this one, removed at
# the end. usage: build-local.sh <out_root> <tip_sha> <base_sha>
set -uo pipefail
OUT=$1; TIP=$2; BASE=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT/bins/w" "$OUT/bins/red" "$OUT/bins/base"
L=$OUT/build-steps.log
cap() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G "$@"; }
exe() { # $1 tree: the engine lib test executable path
  (cd "$1" && cap cargo test -p memra-engine --lib --no-run --message-format=json 2>>"$L") | python3 -c "
import sys, json
for l in sys.stdin:
    try: j = json.loads(l)
    except Exception: continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['profile']['test'] and j['target']['kind'] == ['lib']:
        print(j['executable'])" | head -1
}
cd "$HERE" || exit 1
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] || { echo "rc=2 (this worktree is not at the tip)" >> "$OUT/build.log"; exit 2; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "rc=2 (this worktree is dirty)" >> "$OUT/build.log"; exit 2; }
git rev-parse HEAD > "$OUT/tree-tip.sha"; echo "$BASE" > "$OUT/tree-base.sha"
cap cargo build --release -p memra-server >> "$L" 2>&1 || { echo "rc=1 (w)" >> "$OUT/build.log"; exit 1; }
cp target/release/memra-server "$OUT/bins/w/memra-server"
e=$(exe "$HERE"); [ -n "$e" ] && cp "$e" "$OUT/bins/w/memra-engine-tests" || { echo "rc=1 (w tests)" >> "$OUT/build.log"; exit 1; }
for arm in red base; do
  W=$HERE-$arm-scratch
  git worktree add -q --detach "$W" "$([ $arm = red ] && echo "$TIP" || echo "$BASE")" >> "$L" 2>&1 || { echo "rc=2 ($arm worktree)" >> "$OUT/build.log"; exit 2; }
  ok=1
  if [ $arm = red ]; then
    (cd "$W" && git apply research/spill-a-20260919/pro-single-w/red-arm.patch) >> "$L" 2>&1 || ok=0
    [ $ok = 1 ] && { e=$(CARGO_TARGET_DIR=$HERE/target exe "$W"); [ -n "$e" ] && cp "$e" "$OUT/bins/red/memra-engine-tests" || ok=0; }
  else
    (cd "$W" && CARGO_TARGET_DIR=$HERE/target/base-scratch cap cargo build --release -p memra-server) >> "$L" 2>&1 \
      && cp "$HERE/target/base-scratch/release/memra-server" "$OUT/bins/base/memra-server" || ok=0
  fi
  git worktree remove --force "$W" >> "$L" 2>&1
  [ $ok = 1 ] || { echo "rc=1 ($arm)" >> "$OUT/build.log"; exit 1; }
done
rm -rf "$HERE/target/base-scratch"
sha256sum "$OUT"/bins/*/memra-* | tee "$OUT/binaries.sha256"
{ echo "red marker: engine $(grep -ac 'day61 red arm' "$OUT/bins/red/memra-engine-tests")"
  echo "w tests marker (must be 0): $(grep -ac 'day61 red arm' "$OUT/bins/w/memra-engine-tests")"; } > "$OUT/markers.txt"
echo "rc=0" >> "$OUT/build.log"
