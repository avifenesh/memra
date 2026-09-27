#!/usr/bin/env bash
# WP-A day 42 design S2 on the local rig (DAY42.md section 1), outside any hold: three release servers and the tip's
# test binaries. s2 = the lane tip (design S2); g4 = the lane before S2's code (its crates equal to main 5d653e851: G4
# and T), built in a scratch worktree; gpp = 358749c9f (G'', the hump clause's positive control, the 5090's day-38
# control), built in the same scratch worktree after g4. The scratch target lives on disk (not the RAM-backed /tmp) and
# is removed at the end with the scratch worktree. Every cargo step under the CPU quota.
# usage: build.sh <out> <g4_sha> [gpp_sha]
set -uo pipefail
OUT=$1; G4=$2; GPP=${3:-358749c9f}
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
SCR=$HOME/.cache/wt-a-d42
mkdir -p "$OUT/s2" "$OUT/g4" "$OUT/gpp" "$SCR"
q cargo build --release -p memra-server 2>&1 | tail -1
cp target/release/memra-server "$OUT/s2/memra-server" || exit 1
st=$(q cargo test -p memra-server --lib --no-run --message-format=json 2>/dev/null | exe memra_server)
et=$(q cargo test -p memra-engine --lib --no-run --message-format=json 2>/dev/null | exe memra_engine)
[ -n "$st" ] && [ -n "$et" ] || { echo "test binaries not found"; exit 1; }
cp "$st" "$OUT/s2/server-tests" && cp "$et" "$OUT/s2/engine-tests" || exit 1
[ "$(git rev-parse HEAD)" = "$TIP" ] || { echo "HEAD moved during the build"; exit 1; }
rm -rf "$SCR/wt"
git worktree add -q --detach "$SCR/wt" "$G4" || exit 1
(cd "$SCR/wt" && q env CARGO_TARGET_DIR="$SCR/target" cargo build --release -p memra-server 2>&1 | tail -1)
cp "$SCR/target/release/memra-server" "$OUT/g4/memra-server" || exit 1
(cd "$SCR/wt" && git checkout -q --detach "$GPP") || exit 1
(cd "$SCR/wt" && q env CARGO_TARGET_DIR="$SCR/target" cargo build --release -p memra-server 2>&1 | tail -1)
cp "$SCR/target/release/memra-server" "$OUT/gpp/memra-server" || exit 1
git worktree remove --force "$SCR/wt"
rm -rf "$SCR"
echo "$TIP" > "$OUT/tree.sha"; echo "$G4" > "$OUT/tree-g4.sha"; echo "$GPP" > "$OUT/tree-gpp.sha"
sha256sum "$OUT"/s2/* "$OUT"/g4/* "$OUT"/gpp/* | tee "$OUT/binaries.sha256"
for b in s2 g4 gpp; do
    echo "$b span-receipt-sealed wording: $(grep -ac 'span receipt sealed on the copy stream' "$OUT/$b/memra-server") span-refusal wording: $(grep -ac 'span landed bytes differ from' "$OUT/$b/memra-server")"
done | tee "$OUT/markers.txt"
echo BUILD-DONE
