#!/usr/bin/env bash
# Reproducible day-1 CPU check without modifying any lead-owned Cargo/lib/contracts file.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
LANE="$ROOT/research/spill-d-20260919"
SCRATCH=$(mktemp -d "$LANE/.cpu-check.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
cat > "$SCRATCH/Cargo.toml" <<EOF
[package]
name = "memra-tier"
version = "0.0.0"
edition = "2021"
publish = false
[workspace]
[lib]
path = "lib.rs"
[[test]]
name = "peer"
path = "$ROOT/crates/memra-tier/tests/peer/mod.rs"
[[test]]
name = "placement"
path = "$ROOT/crates/memra-tier/tests/placement/mod.rs"
EOF
cat > "$SCRATCH/lib.rs" <<EOF
#[path = "$ROOT/crates/memra-tier/src/peer/mod.rs"]
pub mod peer;
#[path = "$ROOT/crates/memra-tier/src/placement/mod.rs"]
pub mod placement;
EOF
cargo check --offline --manifest-path "$SCRATCH/Cargo.toml"
cargo test --offline --manifest-path "$SCRATCH/Cargo.toml" --tests
python3 -B crates/memra-tier/tests/battery/test_receipts.py
# Compile the proposal's real Rust facade instead of trusting prose signatures.
python3 - "$LANE/CONTRACTS-PROPOSAL.md" "$SCRATCH/facade.rs" <<'PY'
import sys
from pathlib import Path
body = Path(sys.argv[1]).read_text().split('```rust\n', 1)[1].split('```', 1)[0]
Path(sys.argv[2]).write_text(body)
PY
RLIB=$(find "$SCRATCH/target/debug/deps" -name 'libmemra_tier-*.rlib' -print)
[ -n "$RLIB" ] && [ -f "$RLIB" ]
rustc --edition=2021 --crate-type=lib --extern "memra_tier=$RLIB" "$SCRATCH/facade.rs" -o "$SCRATCH/facade.rlib"
printf '%s\n' 'D CPU verification: 8 peer + 5 placement Rust tests, Python receipt teeth, proposal facade compiled. No GPU execution.'
