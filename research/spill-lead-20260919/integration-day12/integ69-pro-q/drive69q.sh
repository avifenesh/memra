set -u
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
cd /root/wt-integ69
{ cargo build --release -p memra-server && cargo build --release -p memra-engine --bin tier-transfer-gate --bin kv-tier-gate; echo build_rc=$?; } > /root/integ69-pro-q/build.log 2>&1
grep -q build_rc=0 /root/integ69-pro-q/build.log || { echo BUILD-FAILED >> /root/integ69-pro-q/build.log; exit 1; }
bash research/spill-lead-20260919/integration-day12/integ69-pro-q/run-held.sh /root/integ69-pro-q /root/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf /root/wt-integ69/target/release/memra-server > /root/integ69-pro-q/driver.out 2>&1
# The pause-demote gate (A day 47, design V) needs a tool-calling model: the 27B, as lane A ran it.
P=/root/integ69-pro-q/pause-demote; mkdir -p $P/ev
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "lock busy" > $P/result.txt; exit 1; }
{ date -u +%FT%TZ; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv; } > $P/card.before.txt 2>&1
bash tools/kv-host-pause-demote-gate.sh --external-lock 9 /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /root/wt-integ69/target/release/memra-server $P/ev > $P/gate.log 2>&1; echo "rc=$?" >> $P/gate.log
nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > $P/card.after.txt 2>&1
sha256sum /root/wt-integ69/target/release/memra-server > $P/binary.sha256
grep -E 'GATE: ' $P/gate.log | tail -n 1 > $P/result.txt
# Lane A's DAY70 section 5: the tier gate binaries under the same hold (neither takes the lock), each with its own
# receipt; kv-tier-gate refuses any MEMRA_* name but three and needs a new --out per arm.
G=/root/integ69-pro-q/tier-gates; mkdir -p $G
TB=/root/wt-integ69/target/release
sha256sum $TB/tier-transfer-gate $TB/kv-tier-gate > $G/binaries.sha256
CLEAN=$(env | awk -F= '/^MEMRA_/ && $1!="MEMRA_GPU_LOCK" && $1!="MEMRA_NVCC" && $1!="MEMRA_CUDA_ARCH" {printf "-u %s ", $1}')
for c in conformance roundtrip; do
    timeout 700 env NVIDIA_TF32_OVERRIDE=0 $TB/tier-transfer-gate $c > $G/ttg-$c.log 2>&1; echo "$?" > $G/ttg-$c.exit
done
for arm in cancel-demote cancel-restore corrupt-host missing-host host-budget-short device-short require-resident; do
    timeout 1800 env $CLEAN NVIDIA_TF32_OVERRIDE=0 $TB/kv-tier-gate --artifact /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
        --case active --context 8192 --tiers host --same-program --kv-allocator pooled \
        --fault "$arm" --out $G/ktg-$arm > $G/ktg-$arm.log 2>&1
    echo "$?" > $G/ktg-$arm.exit
done
for f in $G/*.exit; do echo "$(basename $f .exit) exit=$(cat $f) $(tail -n 1 ${f%.exit}.log)"; done > $G/result.txt
flock -u 9
echo DRIVE-DONE >> /root/integ69-pro-q/driver.out
