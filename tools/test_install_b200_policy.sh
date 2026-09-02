#!/usr/bin/env bash
# B200 is runtime-qualified from source but has no published prebuilt; install.sh must fail before
# network access for both auto-detected and explicitly requested sm_100a.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT

cat > "$SCRATCH/nvidia-smi" <<'EOF'
#!/bin/sh
printf '10.0\n'
EOF
cat > "$SCRATCH/curl" <<'EOF'
#!/bin/sh
echo 'UNEXPECTED NETWORK ACCESS' >&2
exit 99
EOF
chmod +x "$SCRATCH/nvidia-smi" "$SCRATCH/curl"

run_refusal() {
    local label=$1
    shift
    local log="$SCRATCH/$label.log"
    if PATH="$SCRATCH:/usr/bin:/bin" MEMRA_VERSION=v-test "$@" sh "$ROOT/tools/install.sh" \
        >"$log" 2>&1; then
        echo "FAIL: install.sh accepted $label B200 target" >&2
        exit 1
    fi
    grep -q 'runtime-qualified backend has no published prebuilt' "$log"
    grep -q 'MEMRA_CUDA_ARCH=100a cargo build --release --bins' "$log"
    if grep -q 'UNEXPECTED NETWORK ACCESS' "$log"; then
        echo "FAIL: install.sh touched the network before refusing $label B200 target" >&2
        exit 1
    fi
}

run_refusal auto env
run_refusal explicit env MEMRA_CUDA_ARCH=100a

echo 'install-b200-policy fixture: 2 arms PASS'
