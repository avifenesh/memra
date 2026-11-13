#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease), memra#1266 after memra#1256's landing:
# arm F3 = the fix rebased onto main with memra#1256 (src3, unpacked from src3-lite.tar.gz), arm
# M3 = the same tree with main's four changed crate files and env registry (main3/). Stops at
# CAP_MIN; the next call resumes. usage: build_gb10_r2.sh CAP_MIN   -> bins/{F3,M3}/memra-server
set -u
R=<lane>; CAP=${1:-25}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target
L=$R/logs/build.log; mkdir -p "$R/logs"
end=$(( $(date +%s) + CAP * 60 ))
if [ ! -d "$R/src3/crates" ]; then mkdir -p "$R/src3" && tar -xzf "$R/src3-lite.tar.gz" -C "$R/src3" || exit 1; fi
cd "$R/src3" || exit 1
E=crates/memra-engine/src; V=crates/memra-server/src; K=$R/f3-keep; mkdir -p "$K"
[ -f "$K/spec.rs" ] || { cp $E/spec.rs "$K/spec.rs"; cp $E/lib.rs "$K/elib.rs"; cp $V/lib.rs "$K/slib.rs"; cp $V/worker.rs "$K/worker.rs"; cp crates/memra-engine/env-registry.txt "$K/reg.txt"; }
for a in F3 M3; do
  [ -x "$R/bins/$a/memra-server" ] && continue
  left=$(( end - $(date +%s) )); [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before $a" >> "$L"; exit 1; }
  if [ $a = M3 ]; then
    cp "$R/main3/memra-engine-spec.rs" $E/spec.rs; cp "$R/main3/memra-engine-lib.rs" $E/lib.rs
    cp "$R/main3/memra-server-lib.rs" $V/lib.rs; cp "$R/main3/memra-server-worker.rs" $V/worker.rs
    cp "$R/main3/crates-env-registry.txt" crates/memra-engine/env-registry.txt
  else
    cp "$K/spec.rs" $E/spec.rs; cp "$K/elib.rs" $E/lib.rs; cp "$K/slib.rs" $V/lib.rs; cp "$K/worker.rs" $V/worker.rs; cp "$K/reg.txt" crates/memra-engine/env-registry.txt
  fi
  echo "== $a $(date -Is)" >> "$L"; sha256sum $E/spec.rs $V/worker.rs $V/lib.rs | sed "s|$HOME|~|" >> "$L"
  timeout -s INT "$left" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP $a (rc=$?)" >> "$L"; exit 1; }
  mkdir -p "$R/bins/$a"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$a/"
  sha256sum "$R/bins/$a/memra-server" | sed "s|$HOME|~|" >> "$L"
done
echo "$(date -Is) BUILD-DONE F3 M3" >> "$L"
