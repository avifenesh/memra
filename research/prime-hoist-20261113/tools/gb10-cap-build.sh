#!/usr/bin/env bash
# Runs ON GB10 unit-b (memra#1258 capacity A/B), CPU only, inside a sparklock lease: the diagnostic
# memra-server for sm_121a of one arm, from tree-cap (202e1411fe base + the [diag1255] worker,
# diag-worker.py) with that arm's engine files (cap-arms/ARM): dmain = main, ds2 = step 2,
# ds2r = step 2 with the reclaimable workspace, ds2qf = the parked workspace with the trim and
# calibration fixes (its own worker). usage: gb10-cap-build.sh ARM
set -u
ARM=$1
R=$HOME/.cache/memra-1258
T=$R/tree-cap
L=$R/logs/cap-build-$ARM.log; mkdir -p "$R/logs" "$R/bins"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$R/target-cap
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
echo "$$" >> "$R/pids.txt"
[ -d "$CARGO_TARGET_DIR" ] || cp -a "$R/target" "$CARGO_TARGET_DIR"
cd "$T" || exit 1
grep -q 'diag1255\] gate-pass' crates/memra-server/src/worker.rs || { echo "worker not diag-patched"; exit 1; }
for f in hybrid_forward.rs lib.rs mmq_ffi.rs qwen_prime_graph.rs; do
  cmp -s "$R/cap-arms/$ARM/$f" "crates/memra-engine/src/$f" || cp "$R/cap-arms/$ARM/$f" "crates/memra-engine/src/$f"
done
# An arm with its own diag worker (ds2qf: the calibration fix) uses it; the others use the base
# diag worker, saved the first time an arm brings its own.
WK=crates/memra-server/src/worker.rs
if [ -e "$R/cap-arms/$ARM/worker.rs" ]; then
  [ -e "$R/worker-diag-base.rs" ] || cp "$WK" "$R/worker-diag-base.rs"
  cmp -s "$R/cap-arms/$ARM/worker.rs" "$WK" || cp "$R/cap-arms/$ARM/worker.rs" "$WK"
elif [ -e "$R/worker-diag-base.rs" ]; then
  cmp -s "$R/worker-diag-base.rs" "$WK" || cp "$R/worker-diag-base.rs" "$WK"
fi
sha256sum crates/memra-engine/src/{hybrid_forward,lib,mmq_ffi,qwen_prime_graph}.rs crates/memra-server/src/worker.rs > "$L"
nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1
rc=$?
if [ $rc = 0 ]; then
  cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/memra-server-$ARM-121a"
  sha256sum "$R/bins/memra-server-$ARM-121a" >> "$L"
fi
echo "CAP-BUILD-DONE $ARM rc=$rc $(date -Is)" | tee -a "$L"
exit $rc
