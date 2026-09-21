#!/usr/bin/env bash
# Provision the Hy3 lockstep A/B box: toolchain, repo, artifact. Log: /root/lane/provision.log
set -euo pipefail
mkdir -p /root/lane /data/hy3
export DEBIAN_FRONTEND=noninteractive
echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf
apt-get update -qq >/dev/null
apt-get install -y -qq build-essential cmake pkg-config libssl-dev git curl python3-pip python3-venv ripgrep >/dev/null
echo "APT_DONE $(date -u +%T)"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.97.1 >/dev/null
fi
source "$HOME/.cargo/env"; rustc --version
echo "RUST_DONE $(date -u +%T)"
python3 -m venv /root/venv && /root/venv/bin/pip install -q 'huggingface_hub[hf_transfer]' >/dev/null
echo "PIP_DONE $(date -u +%T)"
# artifact download in background (public repo, no token), verified against SHA256SUMS afterwards
( cd /data/hy3 && HF_HUB_ENABLE_HF_TRANSFER=1 /root/venv/bin/hf download Tiyuvta/Hy3-NVFP4 --revision 0af425172b7a --local-dir /data/hy3 --max-workers 16 > /root/lane/hf-download.log 2>&1 && echo "HF_DONE $(date -u +%T)" >> /root/lane/hf-download.log || echo "HF_FAIL $(date -u +%T)" >> /root/lane/hf-download.log ) &
if [ ! -d /root/lane/memra ]; then git clone -q https://github.com/avifenesh/memra /root/lane/memra; fi
cd /root/lane/memra && git fetch -q origin main && git checkout -q origin/main && git log --oneline -1 && git config --global user.email memra-lane@tiyuvta.ai && git config --global user.name memra-lane && rustup show active-toolchain >/dev/null 2>&1
echo "CLONE_DONE $(date -u +%T)"
wait
echo "PROVISION_DONE $(date -u +%T)"
