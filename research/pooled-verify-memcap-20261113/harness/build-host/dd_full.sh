#!/usr/bin/env bash
# CPU build host (CPU build host, no GPU): the local-ci CPU chain of memra#1266 on its own tree.
# clippy -D warnings, coalescer contracts, memra-server suite, memra-engine lib suite, memra-gguf
# lib suite (models absent here: its artifact skips are counted, not a census), portable suites.
# usage: dd_full.sh SRC_DIR  -> $R/full.txt (last line FULL-DONE)
set -u
R=<lane>
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target CARGO_BUILD_JOBS=12
cd "$1" || exit 1
run() { n=$1; shift; echo "# $(date -Is) $n"; taskset -c 20-31 nice -n 15 ionice -c2 -n7 "$@" > "$R/full-$n.log" 2>&1; echo "$n rc=$?"; grep -E "^test result" "$R/full-$n.log" | awk '{p+=$4; f+=$6; i+=$8} END {if (NR) print "  passed " p ", failed " f ", ignored " i " over " NR " binaries"}'; }
{
  echo "# $(date -Is) full CPU chain of memra#1266"
  run fmt cargo fmt --all -- --check
  run clippy cargo clippy --release --locked --all-targets -- -D warnings
  run coalescer python3 tools/check-coalescer-contract.py --out "$R/target/coalescer-contract"
  run server cargo test --release --locked -p memra-server
  run kv cargo test --release --locked -p memra-kv --lib
  run engine cargo test --release --locked -p memra-engine --lib
  run gguf cargo test --release --locked -p memra-gguf --lib
  run portable bash tools/portable-suites.sh
  echo FULL-DONE
} > "$R/full.txt" 2>&1
