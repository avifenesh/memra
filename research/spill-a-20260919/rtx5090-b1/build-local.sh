#!/usr/bin/env bash
# B1's 5090 half (DAY59 section 11; integ67's per-hardware ask), builds outside any hold, under the rig's CPU cap, in ONE
# scratch worktree beside this one with its own target dir (both removed at the end). The trees are the target
# sitting's own: b1 = 7edc329d9 (the release server and its engine and server lib test executables), red = b1 plus
# pro-single-b1/red-arm.patch (test executables), base = 9ab479d9c (the release server). Executables go to <bins>,
# outside /tmp and outside the repo. usage: build-local.sh <out_root> <bins_dir> <b1_sha> <base_sha>
set -uo pipefail
OUT=$1; BINS=$2; B1=$3; BASE=$4
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
W=$HERE-b1-5090-scratch
T=$HERE-b1-5090-target
mkdir -p "$OUT" "$BINS/b1" "$BINS/red" "$BINS/base"
L=$OUT/build-steps.log
cap() { systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G "$@"; }
tests() { # $1 dest: the engine and server lib test executables of the scratch tree
  for pkg in memra-engine memra-server; do
    e=$(cd "$W" && CARGO_TARGET_DIR=$T cap cargo test -p $pkg --lib --no-run --message-format=json 2>>"$L" | python3 -c "
import sys, json
for l in sys.stdin:
    try: j = json.loads(l)
    except Exception: continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['profile']['test'] and j['target']['kind'] == ['lib']:
        print(j['executable'])" | head -1)
    [ -n "$e" ] && cp "$e" "$1/$pkg-tests" || return 1
  done
}
echo "$B1" > "$OUT/tree-b1.sha"; echo "$BASE" > "$OUT/tree-base.sha"
git -C "$HERE" worktree add -q --detach "$W" "$B1" >> "$L" 2>&1 || { echo "rc=2 (worktree)" >> "$OUT/build.log"; exit 2; }
rc=0
(cd "$W" && CARGO_TARGET_DIR=$T cap cargo build --release -p memra-server) >> "$L" 2>&1 && cp "$T/release/memra-server" "$BINS/b1/" || rc=1
[ $rc = 0 ] && { tests "$BINS/b1" || rc=1; }
[ $rc = 0 ] && { (cd "$W" && git apply research/spill-a-20260919/pro-single-b1/red-arm.patch) >> "$L" 2>&1 && tests "$BINS/red" || rc=1; }
[ $rc = 0 ] && { (cd "$W" && git checkout -q -- . && git checkout -q --detach "$BASE") >> "$L" 2>&1 && (cd "$W" && CARGO_TARGET_DIR=$T cap cargo build --release -p memra-server) >> "$L" 2>&1 && cp "$T/release/memra-server" "$BINS/base/" || rc=1; }
git -C "$HERE" worktree remove --force "$W" >> "$L" 2>&1
rm -rf "$T"
[ $rc = 0 ] || { echo "rc=1" >> "$OUT/build.log"; exit 1; }
sha256sum "$BINS"/*/memra-* | tee "$OUT/binaries.sha256"
{ for b in b1 base; do echo "$b batch wording: $(grep -ac 'copy_batch_items_u8' "$BINS/$b/memra-server")"; done
  echo "red marker: engine $(grep -ac 'b1 red arm' "$BINS/red/memra-engine-tests") server $(grep -ac 'b1 red arm' "$BINS/red/memra-server-tests")"
  echo "b1 tests marker (must be 0): $(grep -ac 'b1 red arm' "$BINS/b1/memra-server-tests")"; } > "$OUT/markers.txt"
echo "rc=0" >> "$OUT/build.log"
