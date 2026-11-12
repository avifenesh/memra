#!/usr/bin/env bash
# Runs ON u1 (CPU only, under the caller's lease): the probe arms of memra#1221's record campaign,
# sm_121a memra-server from the 782dc471b7 tree (src2, unpacked once from src782-lite.tar.gz: the
# tree without research/ except the files the crates include) with arm I's trace-only probe in
# worker.rs (worker.i782.rs: pool reserved/used at publisher entry and exit, per plane allocation
# time and pool growth) and the arm's spec.rs:
#   bi = main's trim (spec.main.rs), wi = the head's (spec.head.rs), ni = trim skipped (spec.n.rs).
# The build stops at CAP_MIN (default 25) so it never outlives its lease; the next call resumes it.
# usage: spark_build_probe.sh [CAP_MIN]   -> bins/{bi,wi,ni}-782dc471b7/memra-server; exit 0 when all three exist
set -u
R=$HOME/.cache/memra-1221; CAP=${1:-25}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target2
L=$R/logs/build-probe.log; mkdir -p "$R/logs"
end=$(( $(date +%s) + CAP * 60 ))
[ -d "$R/src2/crates" ] || { mkdir -p "$R/src2" && tar -xzf "$R/src782-lite.tar.gz" -C "$R/src2"; } || exit 1
cd "$R/src2" || exit 1
for a in "bi spec.main.rs" "wi spec.head.rs" "ni spec.n.rs"; do
  set -- $a; OUT=$1-782dc471b7
  [ -x "$R/bins/$OUT/memra-server" ] && continue
  left=$(( end - $(date +%s) )); [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before $OUT" >> "$L"; exit 1; }
  cp "$R/arms/worker.i782.rs" crates/memra-server/src/worker.rs
  cp "$R/arms/$2" crates/memra-engine/src/spec.rs; touch crates/memra-engine/src/spec.rs
  echo "== $OUT ($2) $(date -Is)" >> "$L"; sha256sum "$R/arms/worker.i782.rs" "$R/arms/$2" | sed "s|$HOME|~|" >> "$L"
  timeout -s INT "$left" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP $OUT (rc=$?)" >> "$L"; exit 1; }
  mkdir -p "$R/bins/$OUT"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$OUT/"
  sha256sum "$R/bins/$OUT/memra-server" | sed "s|$HOME|~|" >> "$L"
done
echo "$(date -Is) BUILD-DONE" >> "$L"
