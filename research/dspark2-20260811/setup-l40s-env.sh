#!/usr/bin/env bash
set -euo pipefail

root=/home/ubuntu/dspark2
venv="$root/venv"

mkdir -p "$root"/{checkpoints,corpus,logs,src,tmp}

if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
fi

source "$venv/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

python -m pip install --upgrade \
    pip==26.2.1 \
    setuptools==80.9.0 \
    wheel==0.47.0

python -m pip install \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
    accelerate==1.14.0 \
    bitsandbytes==0.50.0 \
    datasets==4.3.0 \
    huggingface_hub==1.26.0 \
    numpy==2.4.6 \
    packaging==26.3 \
    psutil==7.2.2 \
    safetensors==0.8.0 \
    scikit-learn==1.7.2 \
    sentencepiece==0.2.2 \
    tensorboard==2.20.0 \
    tqdm==4.70.0 \
    transformers==5.5.0 \
    ninja==1.13.0

set +e
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 \
    timeout 900 python -m pip install flash-attn==2.7.8 --no-build-isolation
flash_rc=$?
set -e

if [[ $flash_rc -eq 0 ]]; then
    printf 'FLASH_BACKEND=flash-attn\n'
else
    printf 'FLASH_BACKEND=sdpa\n'
    printf 'FLASH_ATTN_INSTALL_RC=%s\n' "$flash_rc"
fi

python -m pip check
df -h /

