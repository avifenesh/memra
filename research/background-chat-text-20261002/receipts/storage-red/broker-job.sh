#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target/B914" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a RUST_TEST_THREADS=1
out="$PWD/research/background-chat-text-20261002/receipts/storage-red"
mkdir -p "$out"
set +e
cargo test --release -p memra-server --lib background_chat_text_storage_refusal -- --test-threads=1 2>&1 | tee "$out/old-order-red.log"
code=${PIPESTATUS[0]}
set -e
python3 - "$out/old-order-red.log" "$code" <<'VERIFY'
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
assert int(sys.argv[2])==101 and 'Complete {' in text and 'background_storage_failed' in text,text
VERIFY
