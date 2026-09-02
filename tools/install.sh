#!/bin/sh
# memra installer — downloads the prebuilt self-contained binaries from the latest
# GitHub release (or $MEMRA_VERSION, e.g. MEMRA_VERSION=v0.69.0) into $MEMRA_INSTALL_DIR
# (default ~/.local/bin) and verifies the sha256 against the release's SHA256SUMS.
#
#   curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
#
# Arch selection: sm_120a (RTX 50-series, default) / sm_100a (B200 source build) /
# sm_90a (Hopper) / sm_89 (Ada portable). B200 is runtime-qualified from source but has no
# published prebuilt, so the release installer refuses it before network access. Other targets
# auto-detect from nvidia-smi; override with MEMRA_CUDA_ARCH.
# Requirements at RUN time: Linux x86_64, NVIDIA driver >= 580 (CUDA 13
# runtime support), and the CUDA 13 runtime libraries (cudart, cublas, cublasLt). The glibc
# floor is NOT stated here on purpose — it is read out of the release's own SHA256SUMS and
# matched against this host (see the block below). Model weights are NOT bundled —
# run-gen/memra-server auto-download from Hugging Face via hf:owner/repo:QUANT specs.
set -eu

REPO="avifenesh/memra"
INSTALL_DIR="${MEMRA_INSTALL_DIR:-$HOME/.local/bin}"
BINS="memra-server run-gen run-spec kernel-check"

err() { echo "install.sh: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar  >/dev/null 2>&1 || err "tar is required"
[ "$(uname -s)" = "Linux" ]   || err "prebuilt binaries are Linux-only; build from source: cargo install memra-server"
[ "$(uname -m)" = "x86_64" ]  || err "prebuilt binaries are x86_64-only; build from source: cargo install memra-server"

# Resolve CUDA arch: explicit MEMRA_CUDA_ARCH, else nvidia-smi compute cap, else 120a.
# B200 is recognized but refused before any network call because no sm_100a release asset is
# published yet. Source builds auto-detect it and carry their own hardware-qualified gates.
ARCH="${MEMRA_CUDA_ARCH:-}"
if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ') || cap=""
    case "$cap" in
        12.0|12.1) ARCH=120a ;;
        10.0)      ARCH=100a ;;
        9.0)       ARCH=90a  ;;
        8.9)       ARCH=89   ;;
    esac
fi
ARCH="${ARCH:-120a}"
[ "$ARCH" != "100a" ] || err "B200 sm_100a is source-only in the release installer: the
runtime-qualified backend has no published prebuilt yet. Build from source on the B200:
    MEMRA_CUDA_ARCH=100a cargo build --release --bins
(`MEMRA_CUDA_ARCH` is optional on the B200 because source builds auto-detect compute capability
10.0). Use only model and kernel paths covered by the B200 receipt matrix."

# Resolve version only after arch admission, so a known-unpublished B200 target fails locally
# instead of making a GitHub API request and later blaming a missing release asset.
VERSION="${MEMRA_VERSION:-}"
if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4) || err "could not resolve latest release"
fi

BASE="https://github.com/$REPO/releases/download/$VERSION"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# THE GLIBC FLOOR IS DISCOVERED FROM THE RELEASE, NEVER HARDCODED.
#
# This script used to build the asset name around a literal `glibc2.35`, while
# .github/workflows/release.yml DERIVES that number at build time from
# `ldd --version` on the runner. They agreed only for as long as GitHub's ubuntu-22.04 image
# reported 2.35 — and that image is on the deprecation path. The day it moved, every asset
# would be published as glibc2.39-* and this line would 404 for EVERY USER, with an error
# message that blamed the sm_ARCH matrix instead. A published name must be read from the
# release, not restated here.
#
# SHA256SUMS is authenticated independently with the release workflow's GitHub OIDC identity.
# A release-asset attacker can replace the tarball, sums, and bundle together; cosign still
# refuses because it verifies the signed payload and transparency proof against that identity.
curl -fsSL -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
    || err "download failed: $BASE/SHA256SUMS (is $VERSION a real release?)"
curl -fsSL -o "$TMP/SHA256SUMS.sigstore.json" "$BASE/SHA256SUMS.sigstore.json" \
    || err "release $VERSION has no signed checksum bundle"

COSIGN_VERSION=v3.1.3
COSIGN_SHA256=4629c757b7618056f8ddd7e2625ae9fdd94c0372a65049520bc7d9df9efc7f71
curl -fsSL -o "$TMP/cosign" \
    "https://github.com/sigstore/cosign/releases/download/$COSIGN_VERSION/cosign-linux-amd64" \
    || err "could not download the pinned cosign verifier $COSIGN_VERSION"
printf '%s  %s\n' "$COSIGN_SHA256" "$TMP/cosign" | sha256sum -c - >/dev/null \
    || err "cosign verifier digest mismatch"
chmod 755 "$TMP/cosign"
"$TMP/cosign" verify-blob \
    --bundle "$TMP/SHA256SUMS.sigstore.json" \
    --certificate-identity "https://github.com/avifenesh/memra/.github/workflows/release.yml@refs/tags/$VERSION" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
    "$TMP/SHA256SUMS" >/dev/null \
    || err "checksum signature verification failed for $VERSION"

# Every floor published for our arch, ascending.
floors=$(sed -n "s/.*memra-$VERSION-linux-x86_64-glibc\([0-9.]*\)-sm$ARCH\.tar\.gz\$/\1/p" \
         "$TMP/SHA256SUMS" | sort -t. -k1,1n -k2,2n -u)
[ -n "$floors" ] || err "release $VERSION publishes no sm_$ARCH build.
Build from source instead: cargo install memra-server (needs the CUDA toolkit).
Available assets in $VERSION:
$(sed -n 's/.*  //p' "$TMP/SHA256SUMS" | sed 's/^/    /')"

# Highest floor the host can actually satisfy: a 2.39-floor binary will not run on a 2.35 host.
host_glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$') || host_glibc=""
FLOOR=""
for f in $floors; do
    if [ -n "$host_glibc" ]; then
        # sort -V puts the smaller first; keep f while f <= host_glibc.
        [ "$(printf '%s\n%s\n' "$f" "$host_glibc" | sort -V | head -1)" = "$f" ] || continue
    fi
    FLOOR="$f"
done
[ -n "$FLOOR" ] || err "this host's glibc ${host_glibc:-unknown} is older than every published
floor for sm_$ARCH ($(echo $floors | tr '\n' ' ')). Build from source: cargo install memra-server"

PKG="memra-$VERSION-linux-x86_64-glibc$FLOOR-sm$ARCH"
echo "memra $VERSION (sm_$ARCH, glibc floor $FLOOR, host ${host_glibc:-unknown}) -> $INSTALL_DIR"

curl -fsSL -o "$TMP/$PKG.tar.gz" "$BASE/$PKG.tar.gz" \
    || err "download failed: $BASE/$PKG.tar.gz"
( cd "$TMP" && grep " $PKG.tar.gz\$" SHA256SUMS | sha256sum -c - >/dev/null ) \
    || err "sha256 verification FAILED for $PKG.tar.gz"

tar -C "$TMP" -xzf "$TMP/$PKG.tar.gz"
mkdir -p "$INSTALL_DIR"
for b in $BINS; do
    install -m 755 "$TMP/$PKG/$b" "$INSTALL_DIR/$b"
done

echo "installed: $BINS"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) echo "NOTE: $INSTALL_DIR is not on your PATH" ;;
esac
echo "verify:   $INSTALL_DIR/kernel-check   # expect: ALL GREEN"
