set -eu
cd /root/dp
arm=$1                 # off | on
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export MEMRA_COMPAT=openai
export MEMRA_MODELS="qwen/qwen3.5-9b=/root/dp/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
export MEMRA_MODEL_METADATA=/root/dp/scripts/qwen35-9b.models.toml
export MEMRA_PORT=8080 PORT=8080
export MEMRA_NONSTREAM_DEADLINE_GATE=0 MEMRA_TIMEOUT_MS_MAX=1800000
# THE ONLY DIFFERENCE BETWEEN ARMS. Unset is the shipped default (door OFF); =1 is the door.
case "$arm" in
  off) ;;                                   # env absent, exactly as the default resolves it
  on)  export MEMRA_SERVE_DEVPENALTY=1 ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
exec /root/dp/memra/target/release/memra-server
