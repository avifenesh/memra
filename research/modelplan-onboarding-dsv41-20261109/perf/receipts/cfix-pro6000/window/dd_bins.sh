#!/usr/bin/env bash
# Runs ON the dev desk: sm_120a release binaries of one commit for wrx (dsv41-gate, dsv41-tpbench, dsv41-run,
# memra-server, memra) from the tarball src-TAG.tar that cfix2_bins.sh staged. One fresh tree per commit and this
# lane's own release target dir (~/.cache/dsv41-cfix2/target-rel); every source touched after the untar, since memra
# trees share one crate version.
# usage: dd_bins.sh TAG SHA   -> ~/.cache/dsv41-cfix2/bins/TAG/, marker "BUILD-DONE TAG rc N"
set -u
TAG=$1 SHA=$2
W=$HOME/.cache/dsv41-cfix2
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$W/target-rel
OUT=$W/bins/$TAG
SRC=$W/src-$TAG-${SHA:0:10}
mkdir -p "$OUT" "$SRC"
cd "$SRC" || exit 1
tar -xf "$W/src-$TAG.tar" || { echo "BUILD-DONE $TAG rc 9 (untar)"; exit 1; }
find crates Cargo.toml Cargo.lock -type f -exec touch {} +
echo "$TAG $SHA build start $(date -Is)"
nice -n 10 cargo build --release --locked -j 16 -p memra-engine --bin dsv41-gate --bin dsv41-tpbench --bin dsv41-run \
  -p memra-server --bin memra-server -p memra-cli --bin memra > "$OUT/build.log" 2>&1
rc=$?
grep -E "Compiling memra|Finished|^error" "$OUT/build.log" | tail -n 8
if [ $rc = 0 ]; then
  for b in dsv41-gate dsv41-tpbench dsv41-run memra-server memra; do cp -a "$CARGO_TARGET_DIR/release/$b" "$OUT/"; done
  echo "$SHA built $(date -Is) on the dev desk, cuda 13.1, sm_120a" > "$OUT/SOURCE"
  (cd "$OUT" && sha256sum dsv41-gate dsv41-tpbench dsv41-run memra-server memra)
else
  grep -E "^error" -A10 "$OUT/build.log" | head -30
fi
echo "BUILD-DONE $TAG rc $rc $(date +%T)"
