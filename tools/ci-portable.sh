#!/usr/bin/env bash
# CPU execution, including integration tests and compile-fail doctests. No GPU claim.
set -euo pipefail
cd "$(dirname "$0")/.."
echo 'portable CI: tier, KV and onboarding CLI suites; GPU qualification is separate'
cargo test --release --locked -p memra-tier -p memra-kv -p memra-cli
