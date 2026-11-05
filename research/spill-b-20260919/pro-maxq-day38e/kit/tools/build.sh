#!/usr/bin/env bash
# park-compact Max-Q kit: build the three memra-server arms of the DAY38 addendum D cell on the laptop
# and copy them into the kit inside the same build-lock hold (the target dir is shared between kits).
# Run only as:
#   flock /home/avifenesh/.cache/memra-rig-build.lock nice -n 15 env MEMRA_CUDA_ARCH=120a \
#     CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-maxq/target bash /home/avifenesh/.cache/memra-maxq/park-compact/build.sh
# Arms (kit/DEVIATIONS.md D1): off = the fix-stack tip unpatched (the Max-Q parks whole), on = tip plus
# kit/patches/maxq-arm.patch (the candidate Max-Q key), red = on plus day38-red.patch. Each is
# build-arms.sh's line: cargo build --release -p memra-server --bin memra-server, default features.
# One worktree, the patches applied in order, so each arm rebuilds only the crates its patch touches;
# the worktree is restored to the clean tip at the end.
set -uo pipefail
K=/home/avifenesh/.cache/memra-maxq/park-compact
WT=$K/wt-tip
SHA=2ccb038d1c16358edde65582597e7b840e3f63fb
KIT=$K/kit
LOG=$KIT/build.log
: "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}"
[ "${MEMRA_CUDA_ARCH:-}" = 120a ] || { echo "MEMRA_CUDA_ARCH must be 120a"; exit 1; }
# The deciding cell's arms (box-d bins/green/build.log) and CI both build with CUDA 13.1; pin it.
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc
cd "$WT" || exit 1
restore() { git -C "$WT" checkout -q -- crates; }
trap restore EXIT
build_arm() {  # build_arm LABEL PATCH...: patches are already applied by the caller
    local label=$1; shift
    local out=$KIT/bins/$label
    mkdir -p "$out"
    echo "== arm $label start=$(date -u +%FT%TZ) patches=[$*]"
    cargo build --release -j2 -p memra-server --bin memra-server > "$out/build.log" 2>&1
    local brc=$?
    echo "exit=$brc" >> "$out/build.log"
    grep -E 'nvcc|MEMRA_CUDA_ARCH|^error|Finished|Compiling memra' "$out/build.log" | tail -n 20
    [ "$brc" -eq 0 ] || { echo "arm $label: build failed rc=$brc"; return 1; }
    cp --no-preserve=links "$CARGO_TARGET_DIR/release/memra-server" "$out/memra-server.tmp" \
        && mv -f "$out/memra-server.tmp" "$out/memra-server" || { echo "arm $label: copy failed"; return 1; }
    echo "$SHA" > "$out/source.commit"
    : > "$out/source.patches"
    local p
    for p in "$@"; do (cd "$KIT/patches" && sha256sum "$p") >> "$out/source.patches"; done
    git diff --stat -- crates >> "$out/source.patches"
    ldd "$out/memra-server" > "$out/ldd.txt" 2>&1
    echo "binary=$label/memra-server tree=$SHA patches=[$*] sha256=$(sha256sum "$out/memra-server" | cut -d' ' -f1) bytes=$(stat -c %s "$out/memra-server")"
}
{
    echo "park-compact build start=$(date -u +%FT%TZ) host=laptop sha=$SHA"
    head=$(git rev-parse HEAD)
    echo "head=$head"
    [ "$head" = "$SHA" ] || { echo "worktree is not at $SHA"; echo "rc=1"; exit 1; }
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "tree not clean"; git status --short | head; echo "rc=1"; exit 1; fi
    echo "MEMRA_CUDA_ARCH=$MEMRA_CUDA_ARCH CARGO_TARGET_DIR=$CARGO_TARGET_DIR MEMRA_NVCC=$MEMRA_NVCC"
    "$MEMRA_NVCC" --version | tail -n 1
    cargo --version
    rustc --version
    # The shared target dir hashes workspace crates without their worktree path, so a crate another kit built
    # from another tree after this worktree's checkout looks fresh here (first attempt: memra-reference against
    # a stale memra-gguf without mimo_v2, and a stale memra-engine build script). Touch every tracked file so
    # each workspace crate rebuilds from THIS tree, then prove it did from the off arm's log.
    git ls-files -z -- crates Cargo.toml Cargo.lock | xargs -0 touch -c
    build_arm off || { echo "rc=1"; exit 1; }
    for c in memra-engine memra-gguf memra-kv memra-lanes memra-net-guard memra-reference memra-runtime \
             memra-sampling memra-server memra-tier memra-tokenizer memra-validate; do
        grep -qF "Compiling $c v0.138.0 ($WT/crates/$c)" "$KIT/bins/off/build.log" \
            || { echo "arm off: $c was not compiled from $WT (stale artifact)"; echo "rc=1"; exit 1; }
    done
    echo "arm off: all 12 workspace crates of memra-server compiled from $WT"
    git apply "$KIT/patches/maxq-arm.patch" || { echo "maxq-arm.patch does not apply"; echo "rc=1"; exit 1; }
    build_arm on maxq-arm.patch || { echo "rc=1"; exit 1; }
    for c in memra-engine memra-server; do
        grep -qF "Compiling $c v0.138.0 ($WT/crates/$c)" "$KIT/bins/on/build.log" || { echo "arm on: $c not rebuilt"; echo "rc=1"; exit 1; }
    done
    git apply "$KIT/patches/day38-red.patch" || { echo "day38-red.patch does not apply on top of maxq-arm.patch"; echo "rc=1"; exit 1; }
    build_arm red maxq-arm.patch day38-red.patch || { echo "rc=1"; exit 1; }
    grep -qF "Compiling memra-server v0.138.0 ($WT/crates/memra-server)" "$KIT/bins/red/build.log" || { echo "arm red: memra-server not rebuilt"; echo "rc=1"; exit 1; }
    [ "$(sha256sum < "$KIT/bins/off/memra-server")" != "$(sha256sum < "$KIT/bins/on/memra-server")" ] \
        && [ "$(sha256sum < "$KIT/bins/on/memra-server")" != "$(sha256sum < "$KIT/bins/red/memra-server")" ] \
        || { echo "two arms produced the same binary"; echo "rc=1"; exit 1; }
    restore
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree not restored"; echo "rc=1"; exit 1; }
    (cd "$KIT/bins" && sha256sum off/memra-server on/memra-server red/memra-server) > "$KIT/bins/SHA256SUMS"
    cat "$KIT/bins/SHA256SUMS"
    echo "end=$(date -u +%FT%TZ)"
    echo "rc=0"
} > "$LOG" 2>&1
grep -q '^rc=0$' "$LOG" || { echo "park-compact build failed (see $LOG)"; exit 1; }
echo "built off on red"
