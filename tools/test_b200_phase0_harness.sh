#!/usr/bin/env bash
# CPU-only refusal/plan teeth for the first real-B200 harness. Never opens CUDA.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="$ROOT/research/b200-kernel-twins-dry-20260901/run-box-phase0.sh"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

cat > "$SCRATCH/nvidia-smi" <<EOF
#!/bin/sh
touch "$SCRATCH/nvidia-called"
exit 99
EOF
chmod +x "$SCRATCH/nvidia-smi"

PATH="$SCRATCH:$PATH" "$SCRIPT" --plan > "$SCRATCH/plan.txt"
grep -q 'MEMRA_B200_HOST_ROLE=research-non-production' "$SCRATCH/plan.txt"
grep -q 'NVFP4 exact gate' "$SCRATCH/plan.txt"
[ ! -e "$SCRATCH/nvidia-called" ] || { echo "FAIL: --plan called nvidia-smi" >&2; exit 1; }

if PATH="$SCRATCH:$PATH" "$SCRIPT" run > "$SCRATCH/no-role.txt" 2>&1; then
    echo "FAIL: run accepted a missing host-role acknowledgement" >&2
    exit 1
fi
grep -q 'MEMRA_B200_HOST_ROLE must be exactly research-non-production' "$SCRATCH/no-role.txt"
[ ! -e "$SCRATCH/nvidia-called" ] || { echo "FAIL: no-role refusal called nvidia-smi" >&2; exit 1; }

if PATH="$SCRATCH:$PATH" MEMRA_B200_HOST_ROLE=research-non-production \
    "$SCRIPT" run > "$SCRATCH/no-sha.txt" 2>&1; then
    echo "FAIL: run accepted a missing expected SHA" >&2
    exit 1
fi
grep -q 'MEMRA_B200_EXPECTED_SHA must be 40 lowercase hex' "$SCRATCH/no-sha.txt"
[ ! -e "$SCRATCH/nvidia-called" ] || { echo "FAIL: no-SHA refusal called nvidia-smi" >&2; exit 1; }

echo 'b200-phase0-harness fixture: 3 arms PASS'
