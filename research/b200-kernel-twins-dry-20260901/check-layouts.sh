#!/usr/bin/env bash
# CPU-only proof that the exact SM100 operand/scale address helpers are bijective and bounded.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CXX=${CXX:-c++}
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

cd "$ROOT"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror \
  research/b200-kernel-twins-dry-20260901/layout_contract_test.cpp \
  -o "$SCRATCH/layout-contract-test"
"$SCRATCH/layout-contract-test"
