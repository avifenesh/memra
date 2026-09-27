set -u
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd /root/wt-integ66
{ cargo build --release -p memra-server; echo build_rc=$?; } > /root/integ66-pro/build.log 2>&1
grep -q build_rc=0 /root/integ66-pro/build.log || { echo BUILD-FAILED >> /root/integ66-pro/build.log; exit 1; }
bash research/spill-lead-20260919/integration-day12/integ66-pro/run-held.sh /root/integ66-pro /root/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf /root/wt-integ66/target/release/memra-server > /root/integ66-pro/driver.out 2>&1
# The pause-demote gate (A day 47, design V) needs a tool-calling model: the 27B, as lane A ran it.
P=/root/integ66-pro/pause-demote; mkdir -p $P/ev
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "lock busy" > $P/result.txt; exit 1; }
{ date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv; } > $P/card.before.txt 2>&1
bash tools/kv-host-pause-demote-gate.sh --external-lock 9 /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /root/wt-integ66/target/release/memra-server $P/ev > $P/gate.log 2>&1; echo "rc=$?" >> $P/gate.log
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > $P/card.after.txt 2>&1
sha256sum /root/wt-integ66/target/release/memra-server > $P/binary.sha256
grep -E 'GATE: ' $P/gate.log | tail -n 1 > $P/result.txt
flock -u 9
echo DRIVE-DONE >> /root/integ66-pro/driver.out
