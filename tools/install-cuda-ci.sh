#!/usr/bin/env bash
# Install the CUDA 13.1 compiler packages on GitHub's Ubuntu runners from an authenticated,
# digest-pinned repository bootstrap. Keep every privileged workflow on this one manifest.
set -euo pipefail

. /etc/os-release
case "${VERSION_ID:-}" in
  22.04)
    repo=ubuntu2204
    keyring_sha=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df
    ;;
  24.04)
    repo=ubuntu2404
    keyring_sha=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
    ;;
  *)
    echo "unsupported CI Ubuntu version: ${VERSION_ID:-missing}" >&2
    exit 2
    ;;
esac

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
package="$scratch/cuda-keyring_1.1-1_all.deb"
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" -o "$package"
printf '%s  %s\n' "$keyring_sha" "$package" | sha256sum -c -
sudo dpkg -i "$package"
sudo apt-get update -q
sudo apt-get install -y -q cuda-nvcc-13-1 cuda-cudart-dev-13-1 libcublas-dev-13-1 "$@"
