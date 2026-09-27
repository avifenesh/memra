set -u
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/wt-integ60
{ cargo build --release -p memra-server; echo build_rc=$?; } > /root/integ60-pro/build.log 2>&1
grep -q build_rc=0 /root/integ60-pro/build.log || { echo BUILD-FAILED >> /root/integ60-pro/build.log; exit 1; }
bash research/spill-lead-20260919/integration-day12/integ60-pro/run-held.sh /root/integ60-pro /root/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf /root/wt-integ60/target/release/memra-server > /root/integ60-pro/driver.out 2>&1
echo DRIVE-DONE >> /root/integ60-pro/driver.out
