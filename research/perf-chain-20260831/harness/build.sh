#!/usr/bin/env bash
# perf-chain: build one memra commit on the bench box and bank the binary as
# bin/memra-server-<sha12>. Guards the checkout-attribution trap: records
# `git log -1` AFTER checkout, and refuses a suspiciously instant "Finished".
set -uo pipefail
SHA=${1:?commit-ish}
D=/home/ubuntu/perf-chain
R=$D/memra
. "$HOME/.cargo/env"

FULL=$(git -C "$R" rev-parse "$SHA") || exit 2
S12=${FULL:0:12}
OUT=$D/bin/memra-server-$S12
LOG=$D/logs/build-$S12.log

if [ -x "$OUT" ]; then
  echo "BUILD_CACHED $S12 md5=$(md5sum "$OUT" | cut -d' ' -f1)"
  exit 0
fi

# A live runner must never have its own executable yanked out from under it. The precise
# hazard is a server RUNNING FROM the checkout's target dir; boot.sh only ever launches a
# COPY out of bin/, so a checkout is safe while a measurement runs. Assert exactly that,
# then cap the build so an overlapped compile cannot starve the measured server's host
# threads (48 cores; the server is GPU-bound and uses a handful).
OVERLAP=no
for p in $(pgrep -f "^/home/ubuntu/perf-chain/bin/memra-server" 2>/dev/null); do
  EXE=$(readlink "/proc/$p/exe" 2>/dev/null)
  case "$EXE" in
    "$R"/target/*) echo "REFUSE: live server $p runs FROM $R/target - checkout would yank it"; exit 3;;
  esac
  OVERLAP=yes
done
JOBS=44; NICE=""
if [ "$OVERLAP" = yes ]; then JOBS=8; NICE="nice -n 19 ionice -c3"; fi

{
  echo "=== build $S12 ($FULL) started $(date -u +%FT%TZ) ==="
  git -C "$R" checkout -q --detach "$FULL" 2>&1 || { echo "CHECKOUT_FAIL"; exit 4; }
  git -C "$R" clean -qxdf -e target 2>&1
  echo "--- git log -1 AFTER checkout (attribution receipt) ---"
  git -C "$R" log -1 --format='%H %ci %s' 2>&1
  echo "--- toolchain ---"
  rustc --version; cargo --version; /usr/local/cuda-13.2/bin/nvcc --version | tail -2
  echo "--- cargo build ---"
} >> "$LOG" 2>&1

T0=$(date +%s)
( cd "$R" && $NICE env MEMRA_NVCC=/usr/local/cuda-13.2/bin/nvcc MEMRA_CUDA_ARCH=120a \
    CARGO_BUILD_JOBS=$JOBS cargo build --release -p memra-server ) >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
ELAPSED=$((T1-T0))
echo "build_seconds=$ELAPSED rc=$RC" >> "$LOG"

if [ "$RC" != 0 ]; then
  echo "BUILD_FAIL $S12 rc=$RC elapsed=${ELAPSED}s"; tail -25 "$LOG"; exit 5
fi
# Attribution alarm: a real recompile of a changed commit is never ~instant.
if [ "$ELAPSED" -lt 5 ]; then
  echo "BUILD_SUSPECT $S12: finished in ${ELAPSED}s (checkout may not have taken)"; exit 6
fi
cp "$R/target/release/memra-server" "$OUT" || exit 7
{
  echo "sha=$FULL"
  echo "sha12=$S12"
  echo "bin_md5=$(md5sum "$OUT" | cut -d' ' -f1)"
  echo "bin_fingerprint=$(grep -aom1 'memra-[0-9a-f]\{12\}' "$OUT")"
  echo "build_seconds=$ELAPSED"
  echo "built_under_measurement_overlap=$OVERLAP (jobs=$JOBS nice=${NICE:-none})"
  echo "git_log_1=$(git -C "$R" log -1 --format='%H %s' | cut -c1-120)"
} > "$D/receipts/build-$S12.receipt"
echo "BUILD_OK $S12 elapsed=${ELAPSED}s md5=$(md5sum "$OUT" | cut -d' ' -f1)"
cat "$D/receipts/build-$S12.receipt"
