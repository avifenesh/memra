#!/bin/bash
set -euo pipefail
export CARGO_TARGET_DIR="$PWD/target/B914" CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RUST_TEST_THREADS=2
out="$PWD/research/background-chat-text-20261002/receipts/build-v8"
mkdir -p "$out"
git rev-parse HEAD > "$out/source.txt"
git diff HEAD --binary > "$out/source.diff"
rustc -Vv > "$out/rustc.txt"
sha256sum /data/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf > "$out/model.sha256"
python3 - "$out/model.sha256" <<'VERIFY'
import sys
from pathlib import Path
assert Path(sys.argv[1]).read_text().split()[0] == '0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c'
VERIFY
cargo test --release -p memra-server --lib background_ -- --test-threads=2 2>&1 | tee "$out/background-tests.log"
python3 - "$out/background-tests.log" <<'VERIFY'
import re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
rows=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
assert len(rows)==1 and int(rows[0][0])>=6 and tuple(map(int,rows[0][1:]))==(0,0),rows
VERIFY
cargo test --release -p memra-server --lib -- --test-threads=2 2>&1 | tee "$out/server-suite.log"
python3 tools/validation_plan.py cargo clippy --packages memra-server 2>&1 | tee "$out/clippy.log"
cargo build --release -p memra-server --bin memra-server --example background_accounting_gate 2>&1 | tee "$out/build.log"
sha256sum "$CARGO_TARGET_DIR/release/memra-server" "$CARGO_TARGET_DIR/release/examples/background_accounting_gate" > "$out/binaries.sha256"
cargo fmt --all -- --check
git diff --check
tools/check-flags.sh
