set -eux
. /root/.cargo/env
mkdir -p /data/ai-ml/hf-models
ln -sfn /root/dp/models/qwen35-9b-nvfp4-gguf /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf
cd /root/dp/memra
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu
export MEMRA_CUDA_ARCH=120a
export MEMRA_MODELS_DIR=/root/dp/models
export MEMRA_KC_MODELS_DIR=/root/dp/models/qwen35-9b-nvfp4-gguf
export MEMRA_CI_KC_SKIP_BUDGET=13   # measured on this box, see RESULTS.md
export MEMRA_CI_GGUF=0              # census run separately at measured budget 12
export MEMRA_CI_RELEASE_CARD=0      # red arm unsatisfiable: roster models absent on a 1-model box
export MEMRA_CI_OVERLAP=0            # serial: overlapped CPU chain flaked the cache-meter gate
export MEMRA_CI_LOCK=/tmp/memra-gpu.lock
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock.inner
time tools/local-ci.sh
echo LOCALCI_DONE
