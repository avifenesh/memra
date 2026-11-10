#!/usr/bin/env bash
# Runs ON a GB10 host, unleased, CPU only, while no timed cell runs on the unit: the sm_121a
# binaries of both arms from ONE source tree and ONE target dir. The tree is the base commit
# copied with git archive; an arm is that tree with the arm's crates/memra-server/src/worker.rs
# (the only file the change touches), so the engine is compiled once and each arm recompiles
# the server crate only.
#   base server: the base's worker.rs.  base test binary: tools/red_worker.py's file (the base's
#   production code plus the head's tests).  head server and test binary: the head's worker.rs.
# usage: spark_build.sh BASE_SHA10 HEAD_SHA10   (files under $R/arms/: worker.base.rs worker.red.rs worker.head.rs)
set -u
BASE=$1; HEAD=$2
R=$HOME/.cache/memra-specpub
L=$R/logs/build.log; mkdir -p "$R/logs" "$R/bins"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target
W=$R/src/crates/memra-server/src/worker.rs
cd "$R/src" || exit 1
: > "$L"
arm() { # arm NAME WORKER_FILE KIND(server|test)
  local out=$R/bins/$1; mkdir -p "$out"
  cp "$R/arms/$2" "$W"; touch "$W"
  echo "== $1 $3 ($2) $(date -Is)" >> "$L"
  if [ "$3" = server ]; then
    nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 || { echo "BUILD-FAIL $1 $3" >> "$L"; exit 1; }
    cp -a "$CARGO_TARGET_DIR/release/memra-server" "$out/"
  else
    nice -n 10 cargo test --release --locked -j 16 -p memra-server --lib --no-run --message-format=json 2>> "$L" \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m.get("profile",{}).get("test") and m["target"]["name"]=="memra_server":
        print(m["executable"])' > "$out/test-exe.txt" || { echo "BUILD-FAIL $1 $3" >> "$L"; exit 1; }
    exe=$(tail -n 1 "$out/test-exe.txt"); [ -x "$exe" ] || { echo "BUILD-FAIL $1 $3 no executable" >> "$L"; exit 1; }
    cp -a "$exe" "$out/server-lib-test"
  fi
  sha256sum "$out"/memra-server "$out"/server-lib-test 2>/dev/null >> "$L"
}
arm "base-$BASE" worker.base.rs server
arm "base-$BASE" worker.red.rs test
arm "head-$HEAD" worker.head.rs server
arm "head-$HEAD" worker.head.rs test
cp "$R/arms/worker.head.rs" "$W"
grep -c 'Compiling memra-server' "$L" >> "$L"
sha256sum "$R"/bins/*/* | grep -v test-exe > "$R/bins/sha256.txt"
echo BUILD-DONE >> "$L"
