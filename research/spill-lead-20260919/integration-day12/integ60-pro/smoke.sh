set -u
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH MEMRA_NVCC=/usr/local/cuda/bin/nvcc
R=/root/integ60-pro/serve-smoke-rerun; mkdir -p $R
cd /root/wt-integ60
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "lock busy" > $R/result.txt; exit 1; }
{ date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; } > $R/apps-before.txt 2>&1
bash tools/serve-smoke.sh > $R/gate.log 2>&1; echo "rc=$?" >> $R/gate.log
cp /tmp/serve-smoke.log $R/server.log 2>/dev/null
sha256sum target/release/memra-server > $R/binary-after.sha256
grep -E 'serve-smoke: ' $R/gate.log | tail -n 1 > $R/result.txt
flock -u 9
echo SMOKE-DONE >> $R/result.txt
