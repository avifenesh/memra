#!/usr/bin/env bash
# door-free binary build for the step37 degen re-run lane
set -uo pipefail
D=/home/ubuntu/degen-rerun
R=$D/memra
LOG=$D/logs/build-3999a92a6.log
. "$HOME/.cargo/env"
{
  echo "=== build started $(date -u +%FT%TZ) ==="
  echo "--- git log -1 (attribution receipt) ---"
  git -C "$R" log -1 --format='%H %ci %s'
  echo "--- toolchain ---"
  rustc --version; cargo --version; /usr/local/cuda-13.2/bin/nvcc --version | tail -2
} > "$LOG" 2>&1
T0=$(date +%s)
( cd "$R" && env MEMRA_NVCC=/usr/local/cuda-13.2/bin/nvcc MEMRA_CUDA_ARCH=120a \
    CARGO_BUILD_JOBS=44 cargo build --release -p memra-server ) >> "$LOG" 2>&1
RC=$?
T1=$(date +%s); EL=$((T1-T0))
echo "build_seconds=$EL rc=$RC" >> "$LOG"
if [ "$RC" != 0 ]; then echo "BUILD_FAIL rc=$RC elapsed=${EL}s"; tail -30 "$LOG"; exit 5; fi
if [ "$EL" -lt 5 ]; then echo "BUILD_SUSPECT finished in ${EL}s (checkout may not have taken)"; exit 6; fi
cp "$R/target/release/memra-server" "$D/bin/memra-server-3999a92a6" || exit 7
{
  echo "sha=$(git -C "$R" rev-parse HEAD)"
  echo "bin_md5=$(md5sum "$D/bin/memra-server-3999a92a6" | cut -d' ' -f1)"
  echo "bin_sha256=$(sha256sum "$D/bin/memra-server-3999a92a6" | cut -d' ' -f1)"
  echo "bin_bytes=$(stat -c%s "$D/bin/memra-server-3999a92a6")"
  echo "bin_fingerprint_sha12=$(grep -aom1 'memra-[0-9a-f]\{12\}' "$D/bin/memra-server-3999a92a6" | cut -d- -f2)"
  echo "build_seconds=$EL"
  echo "git_log_1=$(git -C "$R" log -1 --format='%H %s' | cut -c1-140)"
  echo "rustc=$(rustc --version)"
  echo "nvcc=$(/usr/local/cuda-13.2/bin/nvcc --version | tail -2 | tr '\n' ' ')"
  echo "cuda_arch=120a"
  echo "door_refusal_marker_present=$(grep -acq 'MEMRA_NVFP4_BANK_V2=1 is' "$D/bin/memra-server-3999a92a6" && echo yes || echo no)"
} > "$D/receipts/build-3999a92a6.receipt"
echo "BUILD_OK elapsed=${EL}s"
cat "$D/receipts/build-3999a92a6.receipt"
