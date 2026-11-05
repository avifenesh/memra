#!/usr/bin/env bash
# moe-prefetch Max-Q kit: build the three c60 binaries (DAY59.md section 1a: run-gen, run-spec, memra-server from
# da649107c, default features) on the laptop and copy them into the kit inside the same build-lock hold.
# Run only as:
#   flock /home/avifenesh/.cache/memra-rig-build.lock nice -n 15 env MEMRA_CUDA_ARCH=120a \
#     CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-maxq/target bash /home/avifenesh/.cache/memra-maxq/moe-prefetch/build.sh
# Mirrors research/spill-c-20260919/day61-box-build.sh (label c60) minus the on-box checkout.
set -uo pipefail
K=/home/avifenesh/.cache/memra-maxq/moe-prefetch
WT=$K/wt-c60
SHA=da649107c7e85f7681ddfd179650543b8400270d
OUT=$K/kit/bins/c60
LOG=$K/kit/build-c60.log
: "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}"
# Both registered c60 builds (Workstation box, 5090 queue v9) used CUDA 13.1 (pro-single-day61/build-c60.log); pin it.
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc
[ "${MEMRA_CUDA_ARCH:-}" = 120a ] || { echo "MEMRA_CUDA_ARCH must be 120a"; exit 1; }
mkdir -p "$OUT"
cd "$WT" || exit 1
{
    echo "label=c60 sha=$SHA start=$(date -u +%FT%TZ) host=laptop"
    head=$(git rev-parse HEAD)
    echo "head=$head"
    [ "$head" = "$SHA" ] || { echo "worktree is not at $SHA"; echo "rc=1"; exit 1; }
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "tree not clean"; git status --short | head; echo "rc=1"; exit 1; fi
    echo "MEMRA_CUDA_ARCH=$MEMRA_CUDA_ARCH CARGO_TARGET_DIR=$CARGO_TARGET_DIR MEMRA_NVCC=${MEMRA_NVCC:-auto}"
    cargo --version
    rustc --version
    cargo build --release -j2 -p memra-engine --bin run-gen --bin run-spec 2>&1 | grep -E 'nvcc|MEMRA_CUDA_ARCH|^error|Finished|Compiling memra' | tail -n 20
    brc=${PIPESTATUS[0]}
    [ "$brc" -eq 0 ] || { echo "rc=$brc"; exit 1; }
    cargo build --release -j2 -p memra-server 2>&1 | grep -E 'nvcc|^error|Finished|Compiling memra' | tail -n 20
    brc=${PIPESTATUS[0]}
    [ "$brc" -eq 0 ] || { echo "rc=$brc"; exit 1; }
    for b in run-gen run-spec memra-server; do
        cp --no-preserve=links "$CARGO_TARGET_DIR/release/$b" "$OUT/$b-c60.tmp" && mv -f "$OUT/$b-c60.tmp" "$OUT/$b-c60" || { echo "copy $b failed"; echo "rc=1"; exit 1; }
        echo "binary=$b-c60 tree=$head sha256=$(sha256sum "$OUT/$b-c60" | cut -d' ' -f1) bytes=$(stat -c %s "$OUT/$b-c60")"
    done
    echo "end=$(date -u +%FT%TZ)"
    echo "rc=0"
} > "$LOG" 2>&1
grep -q '^rc=0$' "$LOG" || { echo "build c60 failed (see $LOG)"; exit 1; }
echo "built c60"
