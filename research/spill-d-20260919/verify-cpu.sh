#!/usr/bin/env bash
# Day-2 uses the actual frozen workspace, not a competing scratch contract crate.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo check -p memra-tier --offline --all-targets
cargo test -p memra-tier --offline
python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p 'test_*.py'
printf '%s\n' 'D CPU verification complete. No GPU execution or hardware qualification.'
