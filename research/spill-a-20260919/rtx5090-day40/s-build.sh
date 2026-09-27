#!/usr/bin/env bash
# WP-A day 40 design S on the local rig (DAY40 section 3), outside any hold: the S tip's release server and its test
# binaries (the G4 arm is g4-build.sh's /tmp/wt-a-d38g4/g4, crates equal to G4). Under the CPU quota. usage: s-build.sh <out>
set -uo pipefail
OUT=$1
cd "$(dirname "$0")/../../.." || exit 1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
q() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice "$@"; }
exe() { python3 -c 'import sys,json
name=sys.argv[1]
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]==name: print(m["executable"])' "$1" | tail -1; }
mkdir -p "$OUT/s"
q cargo build --release -p memra-server 2>&1 | tail -1
cp target/release/memra-server "$OUT/s/memra-server" || exit 1
st=$(q cargo test -p memra-server --lib --no-run --message-format=json 2>/dev/null | exe memra_server)
et=$(q cargo test -p memra-engine --lib --no-run --message-format=json 2>/dev/null | exe memra_engine)
[ -n "$st" ] && [ -n "$et" ] || { echo "test binaries not found"; exit 1; }
cp "$st" "$OUT/s/server-tests" && cp "$et" "$OUT/s/engine-tests" || exit 1
git rev-parse HEAD > "$OUT/tree.sha"
sha256sum "$OUT"/s/* | tee "$OUT/binaries.sha256"
echo "s span-flip-landed wording: $(grep -ac 'span landed bytes differ from' "$OUT/s/memra-server")" | tee "$OUT/markers.txt"
echo BUILD-DONE
