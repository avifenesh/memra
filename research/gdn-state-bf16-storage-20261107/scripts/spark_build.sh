#!/usr/bin/env bash
# Runs ON a GB10 unit: build the bf16 GDN state storage tree (sm_121a) at low priority.
#   spark_build.sh TAG
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
R=$HOME/.cache/memra-gdnbf16-real
TAG=${1:?tag}
T=$R/target
cd "$R/src"
CARGO_TARGET_DIR=$T nice -n 10 ionice -c2 -n7 cargo build --release -j 6 \
  -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin decode-batch-bench \
  --bin argmax-margin-probe -p memra-server --bin memra-server > "$R/logs/build-$TAG.log" 2>&1
rc=$?
echo "BUILD=$rc" >> "$R/logs/build-$TAG.log"
if [ $rc = 0 ]; then
  mkdir -p "$R/bins/$TAG"
  for b in kernel-check run-gen run-spec decode-batch-bench argmax-margin-probe memra-server; do
    cp "$T/release/$b" "$R/bins/$TAG/"
  done
  sha256sum "$R/bins/$TAG"/* > "$R/bins/$TAG/sha256.txt"
fi
echo "SPARK_BUILD_DONE $TAG rc=$rc"
