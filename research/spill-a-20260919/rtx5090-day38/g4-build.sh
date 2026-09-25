#!/usr/bin/env bash
# WP-A day 38 design G4 on the local rig (DAY38 section 17), outside any hold: three release servers and the tip's test
# binaries. g4 = the lane tip (G4); gpp = the tip's crates taken back to the pre-G''' commit (G'', the hump's positive
# control), built, then the tree checked back at the tip; base = 80039a8de (the lane tip before design G's code) in a
# scratch worktree with its own target dir. Every cargo step under the CPU quota. usage: g4-build.sh <out> <pre_g3_sha>
set -uo pipefail
OUT=$1; PRE=$2
cd "$(dirname "$0")/../../.." || exit 1
TIP=$(git rev-parse HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
q() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice "$@"; }
exe() { python3 -c 'import sys,json
name=sys.argv[1]
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]==name: print(m["executable"])' "$1" | tail -1; }
mkdir -p "$OUT/g4" "$OUT/gpp" "$OUT/base"
q cargo build --release -p memra-server 2>&1 | tail -1
cp target/release/memra-server "$OUT/g4/memra-server" || exit 1
st=$(q cargo test -p memra-server --lib --no-run --message-format=json 2>/dev/null | exe memra_server)
et=$(q cargo test -p memra-engine --lib --no-run --message-format=json 2>/dev/null | exe memra_engine)
[ -n "$st" ] && [ -n "$et" ] || { echo "test binaries not found"; exit 1; }
cp "$st" "$OUT/g4/server-tests" && cp "$et" "$OUT/g4/engine-tests" || exit 1
git checkout "$PRE" -- crates || exit 1
q cargo build --release -p memra-server 2>&1 | tail -1
cp target/release/memra-server "$OUT/gpp/memra-server"
git checkout "$TIP" -- crates || exit 1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree did not return to the tip"; exit 1; }
rm -rf /tmp/wt-a-d38g4-base
git worktree add -q --detach /tmp/wt-a-d38g4-base 80039a8de || exit 1
(cd /tmp/wt-a-d38g4-base && q env CARGO_TARGET_DIR=/tmp/wt-a-d38g4-base-target cargo build --release -p memra-server 2>&1 | tail -1)
cp /tmp/wt-a-d38g4-base-target/release/memra-server "$OUT/base/memra-server" || exit 1
git worktree remove --force /tmp/wt-a-d38g4-base
echo "$TIP" > "$OUT/tree.sha"; echo "$PRE" > "$OUT/tree-gpp.sha"
sha256sum "$OUT"/g4/* "$OUT"/gpp/* "$OUT"/base/* | tee "$OUT/binaries.sha256"
for b in g4 gpp base; do echo "$b copy-stream receipt lines: $(grep -ac 'receipts on the copy stream' "$OUT/$b/memra-server") receipt-stream wording: $(grep -ac "receipts on the receipt stream" "$OUT/$b/memra-server")"; done | tee "$OUT/markers.txt"
echo BUILD-DONE
