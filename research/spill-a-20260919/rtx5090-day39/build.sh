#!/usr/bin/env bash
# WP-A day 39 on the local rig (DAY39 section 5): the FT, F1 binaries and the tip's test binaries from ONE tree. FT is the tip
# as built; F1 is the tip plus f1.patch (fill_threads fixed at 1), built, then the patch reversed and the tree checked back at
# the tip. Every cargo step under the CPU quota. usage: build.sh <out>   (writes <out>/{ft,f1}/memra-server, <out>/ft/*-tests)
set -uo pipefail
OUT=$1
cd "$(dirname "$0")/../../.." || exit 1
TIP=$(git rev-parse HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
q() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice "$@"; }
mkdir -p "$OUT/ft" "$OUT/f1"
q cargo build --release -p memra-server 2>&1 | tail -2 || exit 1
cp target/release/memra-server "$OUT/ft/memra-server" || exit 1
st=$(q cargo test -p memra-server --lib --no-run --message-format=json 2>/dev/null | python3 -c 'import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]=="memra_server": print(m["executable"])' | tail -1)
et=$(q cargo test -p memra-engine --lib --no-run --message-format=json 2>/dev/null | python3 -c 'import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]=="memra_engine": print(m["executable"])' | tail -1)
[ -n "$st" ] && [ -n "$et" ] || { echo "test binaries not found"; exit 1; }
cp "$st" "$OUT/ft/server-tests" && cp "$et" "$OUT/ft/engine-tests" || exit 1
git apply research/spill-a-20260919/rtx5090-day39/f1.patch || exit 1
q cargo build --release -p memra-server 2>&1 | tail -2
rc=${PIPESTATUS[0]}
cp target/release/memra-server "$OUT/f1/memra-server"
git apply -R research/spill-a-20260919/rtx5090-day39/f1.patch || exit 1
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ] || { echo "tree did not return to the tip"; exit 1; }
echo "$TIP" > "$OUT/tree.sha"
sha256sum "$OUT"/ft/* "$OUT"/f1/* | tee "$OUT/binaries.sha256"
exit $rc
