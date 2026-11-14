#!/usr/bin/env bash
# GB10 build lease: for each TAG, dsv41-gate (and memra-server when SERVER=1) for 121a from the staged tarball
# src-TAG.tar, one target dir, one after another. usage: build.sh TAG[:server] ...
# Done-file out/build/runs.txt, marker BUILD-DONE.
set -u
R=$HOME/.cache/dsv41-fu1113
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" RUSTUP_TOOLCHAIN=1.97.1
export MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR=$R/target MEMRA_NVCC=/usr/local/cuda/bin/nvcc
O=$R/out/build; mkdir -p $O; : > $O/runs.txt
for item in "$@"; do
  TAG=${item%%:*}
  echo "build $TAG start $(date +%T)" >> $O/runs.txt
  rm -rf $R/src && mkdir -p $R/src $R/bins/$TAG && cd $R/src && tar -xf $R/src-$TAG.tar || { echo "$TAG untar failed" >> $O/runs.txt; continue; }
  find crates/memra-engine/cu crates/memra-engine/src crates/memra-engine/build.rs crates/memra-server/src -type f -exec touch {} +
  PKG="-p memra-engine --bin dsv41-gate"
  case $item in *:server) PKG="$PKG -p memra-server --bin memra-server" ;; esac
  nice -n 19 timeout 80m cargo build --release --locked -j 16 $PKG > $O/build-$TAG.log 2>&1
  rc=$?
  echo "$TAG build rc $rc $(date +%T) compiling-lines $(grep -c 'Compiling' $O/build-$TAG.log)" >> $O/runs.txt
  if [ $rc -ne 0 ]; then
    grep -E "^error" -A12 $O/build-$TAG.log | head -40 >> $O/runs.txt
  else
    cp -a $R/target/release/dsv41-gate $R/bins/$TAG/
    case $item in *:server) cp -a $R/target/release/memra-server $R/bins/$TAG/ ;; esac
    sha256sum $R/bins/$TAG/* >> $O/runs.txt
  fi
done
echo "BUILD-DONE $(date +%T)" >> $O/runs.txt
