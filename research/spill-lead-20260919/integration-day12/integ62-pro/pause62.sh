set -u
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd /root/wt-integ62
P=/root/integ62-pro/pause-demote-rerun; mkdir -p $P/ev
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "lock busy" > $P/result.txt; exit 1; }
{ date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv; } > $P/card.before.txt 2>&1
bash tools/kv-host-pause-demote-gate.sh --external-lock 9 /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /root/wt-integ62/target/release/memra-server $P/ev > $P/gate.log 2>&1; echo "rc=$?" >> $P/gate.log
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > $P/card.after.txt 2>&1
sha256sum /root/wt-integ62/target/release/memra-server > $P/binary.sha256
grep -E 'GATE: ' $P/gate.log | tail -n 1 > $P/result.txt
flock -u 9
echo PAUSE-DONE >> $P/result.txt
