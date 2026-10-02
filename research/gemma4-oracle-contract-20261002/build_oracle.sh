#!/bin/bash
# Run this build phase under the caller's resource controller, before any GPU lease.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: build_oracle.sh <pinned-llama.cpp-source> <receipt-directory>' >&2
  exit 64
fi
src=$(realpath "$1")
receipt=$(realpath "$2")
[[ $(git -C "$src" rev-parse HEAD) == f3f1a8f2760f28325a5ec20c05b171e5b7c83a29 ]]
gzip -dc "$receipt/oracle-instrumentation.patch.gz" | git -C "$src" apply --check -
gzip -dc "$receipt/oracle-instrumentation.patch.gz" | git -C "$src" apply -
export SCCACHE_DISABLE=1 RUSTC_WRAPPER=
cmake -S "$src" -B "$src/build" -G Ninja -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CCACHE=OFF
cmake --build "$src/build" --target llama-server -j 2
c++ -O2 -std=c++17 -I"$src/include" -I"$src/ggml/include" -I"$src/common" -I"$src/vendor" \
  "$receipt/oracle_probe_v2.cpp" -L"$src/build/bin" -Wl,-rpath,"$src/build/bin" \
  -lllama-common -lllama -lggml -lggml-base -lggml-cpu -o "$src/build/bin/oracle-probe"
sha256sum "$src/build/bin/llama-server" "$src/build/bin/oracle-probe" "$src/build/bin/"*.so* > "$receipt/binaries.sha256"
