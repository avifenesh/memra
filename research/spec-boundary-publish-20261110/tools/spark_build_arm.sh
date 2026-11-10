#!/usr/bin/env bash
# Runs ON a GB10 host, CPU only, while no timed cell runs on the unit: one more arm's sm_121a
# binaries from the tree and target directory spark_build.sh built in (the tree is the base
# commit; an arm is that tree with the arm's crates/memra-server/src/worker.rs), so only the
# server crate recompiles. KIND: server (memra-server), test (the server lib's test binary, as
# server-lib-test). The tree is left on the last arm built.
# usage: spark_build_arm.sh OUT_NAME WORKER_FILE [KIND...]   (the file is under $R/arms/; default: server)
set -u
OUT=$1; WF=$2; shift 2
[ $# -gt 0 ] || set -- server
R=$HOME/.cache/memra-specpub
L=$R/logs/build-$OUT.log; mkdir -p "$R/logs" "$R/bins/$OUT"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
W=$R/src/crates/memra-server/src/worker.rs
cd "$R/src" || exit 1
cp "$R/arms/$WF" "$W"; touch "$W"
: > "$L"
sha256sum "$R/arms/$WF" | sed "s|$HOME|~|" >> "$L"
for kind in "$@"; do
  echo "== $OUT $kind ($WF) $(date -Is)" >> "$L"
  if [ "$kind" = server ]; then
    nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 || { echo "BUILD-FAIL $OUT $kind" >> "$L"; exit 1; }
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$OUT/"
  else
    nice -n 10 cargo test --release --locked -j 16 -p memra-server --lib --no-run --message-format=json 2>> "$L" \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m.get("profile",{}).get("test") and m["target"]["name"]=="memra_server":
        print(m["executable"])' > "$R/bins/$OUT/test-exe.txt" || { echo "BUILD-FAIL $OUT $kind" >> "$L"; exit 1; }
    exe=$(tail -n 1 "$R/bins/$OUT/test-exe.txt"); [ -x "$exe" ] || { echo "BUILD-FAIL $OUT $kind no executable" >> "$L"; exit 1; }
    cp -a "$exe" "$R/bins/$OUT/server-lib-test"
  fi
done
sha256sum "$R/bins/$OUT"/memra-server "$R/bins/$OUT"/server-lib-test 2>/dev/null | sed "s|$HOME|~|" >> "$L"
sha256sum "$R"/bins/*/* | grep -v test-exe | sed "s|$HOME|~|" > "$R/bins/sha256.txt"
echo BUILD-DONE >> "$L"
