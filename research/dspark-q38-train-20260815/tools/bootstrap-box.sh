#!/usr/bin/env bash
# One-shot bring-up for a fresh sbox DL-image box (spot-safe, idempotent, all attempt-1 fixes baked).
# Usage on box: bash bootstrap-box.sh   (expects ~/.hf_token pushed first)
# Logs: ~/bootstrap.log. After this: tmux sessions `serve` (GPU0 DSPARK server) ready to start.
set -ux
exec >> /home/ubuntu/bootstrap.log 2>&1
export PATH="$HOME/.local/bin:$PATH"

# scratch -> DL-image LVM NVMe (do NOT mkfs nvme1n1 — it is LVM-owned, mounted at /opt/dl-image/nvme)
sudo chown ubuntu:ubuntu /opt/dl-image/nvme
[ -L /scratch ] || { sudo rm -rf /scratch; sudo ln -s /opt/dl-image/nvme /scratch; }
mkdir -p /scratch/{models,repos,ckpt,receipts,corpus,venvs} /scratch/receipts/g1

sudo apt-get update -y && sudo apt-get install -y tmux jq git-lfs ninja-build
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export HF_TOKEN=$(cat /home/ubuntu/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
uv tool install --with hf_transfer "huggingface_hub[cli,hf_transfer]" --force || true

cd /scratch/repos
[ -d dflash ]    || git clone --depth 1 https://github.com/z-lab/dflash
[ -d SpecForge ] || git clone --depth 1 https://github.com/sgl-project/SpecForge
[ -d memra ]     || git clone --depth 1 https://github.com/avifenesh/memra

HF=$HOME/.local/bin/hf
$HF download Qwen/Qwen3.8-27B-FP8 --local-dir /scratch/models/qwen38-27b-fp8 &
$HF download RadixArk/Qwen3.8-27B-DSpark --local-dir /scratch/models/radixark-q38-dspark &
$HF download Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF --local-dir /scratch/models/q38-nvfp4-mtp-gguf &

# eval venv: sglang >= 0.5.17 (DSPARK + qwen3.8; fa3 is SM<=90 — use --attention-backend flashinfer)
if [ ! -d /scratch/venvs/eval ]; then
  uv venv -p 3.11 /scratch/venvs/eval
  VIRTUAL_ENV=/scratch/venvs/eval uv pip install "sglang[all]>=0.5.17" ninja
  VIRTUAL_ENV=/scratch/venvs/eval uv pip install -e /scratch/repos/dflash
fi
# train venv: SpecForge + patched capture sglang 0.5.14 (patch script needs `python` on PATH)
if [ ! -d /scratch/venvs/train ]; then
  uv venv -p 3.11 /scratch/venvs/train
  cd /scratch/repos/SpecForge
  VIRTUAL_ENV=/scratch/venvs/train uv pip install -v . --prerelease=allow
  PATH=/scratch/venvs/train/bin:$PATH bash scripts/apply_sglang_spec_capture_patch.sh
fi
wait  # downloads

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "BOOTSTRAP DONE $(date -u +%FT%TZ)"
