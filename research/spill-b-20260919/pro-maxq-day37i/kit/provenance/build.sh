#!/usr/bin/env bash
# Build the bins the kv-vmm Max-Q cell needs (DAY37 O1 replay at the r4 tree), and copy them into the kit while the
# shared build lock is still held (the target dir is shared between kits). Run only as:
#   flock /home/avifenesh/.cache/memra-rig-build.lock nice -n 15 env MEMRA_CUDA_ARCH=120a \
#     CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-maxq/target \
#     bash /home/avifenesh/.cache/memra-maxq/kv-vmm/build.sh
# Labels:
#   lane  c6f9282c26e7  memra-server, kv-tier-gate, vmm-call-cost  (the cargo line of chain.sh:48-49, one invocation)
#   main  17dceb9817d0  memra-server                               (the cargo line of chain.sh:60)
set -euo pipefail
: "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}" "${MEMRA_CUDA_ARCH:?set MEMRA_CUDA_ARCH}"
BASE=/home/avifenesh/.cache/memra-maxq/kv-vmm
KIT=$BASE/kit
export PATH=/usr/local/cuda/bin:$PATH
# The registered builds (pro-single-day37/box*/bins/main/build.log) auto-detected nvcc 13.1 on the rented box; CI pins
# 13.1 the same way (build.rs resolve_nvcc). The laptop's CUDA_HOME points at 13.4, so pin 13.1 explicitly.
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc
[ -x "$MEMRA_NVCC" ] || { echo "no $MEMRA_NVCC" >&2; exit 2; }
build_one() { # label worktree want_sha bins... ; then the cargo args after --
  local label=$1 wt=$2 want=$3; shift 3
  local bins=() ; while [ "$1" != -- ]; do bins+=("$1"); shift; done; shift
  cd "$wt"
  local have; have=$(git rev-parse HEAD)
  [ "$have" = "$want" ] || { echo "$label: worktree at $have, want $want" >&2; exit 2; }
  git diff --quiet HEAD -- crates Cargo.toml Cargo.lock || { echo "$label: worktree crates are dirty" >&2; exit 2; }
  echo "$label build start $(date -u +%FT%TZ) src=$have arch=$MEMRA_CUDA_ARCH target=$CARGO_TARGET_DIR"
  cargo build --release -j2 "$@"
  local out=$KIT/bins/$label
  rm -rf "$out"; mkdir -p "$out"
  for b in "${bins[@]}"; do cp "$CARGO_TARGET_DIR/release/$b" "$out/$b"; done
  (cd "$out" && sha256sum "${bins[@]}") > "$out/SHA256SUMS"
  echo "$have" > "$out/source.commit"
  {
    echo "label=$label"
    echo "source_sha=$have"
    echo "built_utc=$(date -u +%FT%TZ)"
    echo "MEMRA_CUDA_ARCH=$MEMRA_CUDA_ARCH"
    echo "cargo_features=default"
    echo "profile=release"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "MEMRA_NVCC=$MEMRA_NVCC"
    echo "nvcc=$("$MEMRA_NVCC" --version | tail -1)"
    echo "cmd=cargo build --release -j2 $*"
  } > "$out/BUILD-INFO.txt"
  for b in "${bins[@]}"; do ldd "$out/$b" > "$out/ldd-$b.txt" 2>&1 || true; done
  cat "$out/SHA256SUMS"
  echo "$label build end $(date -u +%FT%TZ)"
}
build_one lane "$BASE/wt-lane" c6f9282c26e71349c0263b88dc7beb59c4a57797 memra-server kv-tier-gate vmm-call-cost -- \
  -p memra-server --bin memra-server -p memra-engine --bin kv-tier-gate --bin vmm-call-cost
build_one main "$BASE/wt-main" 17dceb9817d050d0960a8e2d0be372921bfbee80 memra-server -- \
  -p memra-server --bin memra-server
echo "all builds done $(date -u +%FT%TZ)"
