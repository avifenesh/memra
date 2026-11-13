#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease), memra#1266: the sm_121a memra-server
# arms from one tree (src, unpacked once from src-lite.tar.gz: the fix branch's crates, docs,
# tools and the research files the crates include) and one target directory. Arm F = the tree as
# shipped; arm M = the tree with main's (11b368f7d7) spec.rs, worker.rs and server lib.rs, the
# only crate files the fix changes. Stops at CAP_MIN so it never outlives its lease; the next
# call resumes. usage: build_gb10.sh CAP_MIN ARM...   -> bins/ARM/memra-server, logs/build.log
set -u
R=<lane>; CAP=${1:-25}; shift
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
L=$R/logs/build.log; mkdir -p "$R/logs"
end=$(( $(date +%s) + CAP * 60 ))
if [ ! -d "$R/src/crates" ]; then mkdir -p "$R/src" && tar -xzf "$R/src-lite.tar.gz" -C "$R/src" || exit 1; fi
cd "$R/src" || exit 1
E=crates/memra-engine/src/spec.rs; W=crates/memra-server/src/worker.rs; S=crates/memra-server/src/lib.rs
[ -f "$R/fix.spec.rs" ] || { cp -a "$E" "$R/fix.spec.rs"; cp -a "$W" "$R/fix.worker.rs"; cp -a "$S" "$R/fix.server-lib.rs"; }
for a in "$@"; do
  [ -x "$R/bins/$a/memra-server" ] && continue
  left=$(( end - $(date +%s) )); [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before $a" >> "$L"; exit 1; }
  case $a in
    M) cp "$R/main.spec.rs" "$E"; cp "$R/main.worker.rs" "$W"; cp "$R/main.server-lib.rs" "$S" ;;
    F) cp "$R/fix.spec.rs" "$E"; cp "$R/fix.worker.rs" "$W"; cp "$R/fix.server-lib.rs" "$S" ;;
    *) echo "unknown arm $a" >> "$L"; exit 2 ;;
  esac
  touch "$E" "$W" "$S"
  echo "== $a $(date -Is)" >> "$L"; sha256sum "$E" "$W" "$S" | sed "s|$HOME|~|" >> "$L"
  timeout -s INT "$left" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP $a (rc=$?)" >> "$L"; exit 1; }
  mkdir -p "$R/bins/$a"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$a/"
  sha256sum "$R/bins/$a/memra-server" | sed "s|$HOME|~|" >> "$L"
done
echo "$(date -Is) BUILD-DONE $*" >> "$L"
