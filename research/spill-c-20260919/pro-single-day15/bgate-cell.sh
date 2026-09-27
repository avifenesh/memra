#!/usr/bin/env bash
# lane B's prefix-evict-reclaim-gate.py / prefix-newest-turn-fits-gate.py on the day-15 binary; arm off|on sets the door.
# usage: bgate-cell.sh <evict|newest> <off|on> <lockfd>
set -uo pipefail
gate=$1; arm=$2; fd=$3
R=/root/spill-receipts/c-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
env_extra=(MEMRA_GPU_LOCK=/tmp/memra-gpu.lock CUDA_VISIBLE_DEVICES=0)
[ "$arm" = on ] && env_extra+=(MEMRA_KV_HOST_CONTRACTS=1)
case $gate in
  evict) script=tools/prefix-evict-reclaim-gate.py ;;
  newest) script=tools/prefix-newest-turn-fits-gate.py ;;
  *) echo "unknown gate $gate" >&2; exit 2 ;;
esac
sha256sum $R/bins/memra-server
env "${env_extra[@]}" python3 $script --external-lock "$fd" --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --bin $R/bins/memra-server --out $R/b$gate-$arm/ev
