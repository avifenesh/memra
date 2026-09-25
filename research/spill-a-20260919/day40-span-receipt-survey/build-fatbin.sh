#!/usr/bin/env bash
# The engine's tier_receipt fatbin, compiled with the engine build's flags (crates/memra-engine/build.rs: -gencode for the
# card's arch, -O3, --fatbin). usage: build-fatbin.sh <arch, e.g. 120a> <out.fatbin>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
nvcc -gencode "arch=compute_$1,code=sm_$1" -O3 --fatbin "$HERE/../../../crates/memra-engine/cu/tier_receipt.cu" -o "$2"
sha256sum "$HERE/../../../crates/memra-engine/cu/tier_receipt.cu" "$2"
