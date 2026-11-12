#!/usr/bin/env bash
# B200 builds and runs from source but has no published prebuilt; install.sh must fail before
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
    grep -q 'no sm_100a' "$log"
    grep -q 'prebuilt is published' "$log"
    grep -q 'MEMRA_CUDA_ARCH=100a cargo build --release --bins' "$log"
    if grep -q 'UNEXPECTED NETWORK ACCESS' "$log"; then
        echo "FAIL: install.sh touched the network before refusing $label B200 target" >&2
        exit 1
    fi
}

# The CI 100a cell exports MEMRA_CUDA_ARCH=100a; every detection arm clears it.
run_refusal auto env -u MEMRA_CUDA_ARCH
run_refusal explicit env MEMRA_CUDA_ARCH=100a

# A capability with no build target (Ampere 8.6) is refused before network access.
printf '#!/bin/sh\nprintf "8.6\\n"\n' > "$SCRATCH/nvidia-smi"
log="$SCRATCH/ampere.log"
if env -u MEMRA_CUDA_ARCH PATH="$SCRATCH:/usr/bin:/bin" MEMRA_VERSION=v-test sh "$ROOT/tools/install.sh" >"$log" 2>&1; then
    echo "FAIL: install.sh accepted compute capability 8.6" >&2
    exit 1
fi
grep -q 'compute capability 8.6, and memra has no build target' "$log"
if grep -q 'UNEXPECTED NETWORK ACCESS' "$log"; then
    echo "FAIL: install.sh touched the network before refusing 8.6" >&2
    exit 1
fi

# No driver loaded: nvidia-smi prints an error on stdout and exits non-zero. That is not a
# capability; the installer keeps the 120a fallback and goes on to resolve the release (the curl
# stub then stops it, which is how this arm sees that detection passed).
printf '#!/bin/sh\necho "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver."\nexit 9\n' \
    > "$SCRATCH/nvidia-smi"
log="$SCRATCH/nodriver.log"
env -u MEMRA_CUDA_ARCH PATH="$SCRATCH:/usr/bin:/bin" MEMRA_VERSION=v-test sh "$ROOT/tools/install.sh" >"$log" 2>&1 || true
if grep -q 'no build target' "$log"; then
    echo "FAIL: install.sh refused a host whose driver is not loaded" >&2
    cat "$log" >&2
    exit 1
fi
grep -q 'UNEXPECTED NETWORK ACCESS' "$log"

echo 'install-b200-policy fixture: 4 arms PASS'
