set -eux
. /root/.cargo/env
cd /root/dp/memra
git worktree add /root/dp/base HEAD~1 2>/dev/null || true
cd /root/dp/base
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu
export MEMRA_CUDA_ARCH=120a
git log --oneline -1
cargo test --release -p memra-engine --lib -j16 --no-run 2>&1 | tail -3
cargo test --release -p memra-engine --lib -j16 -- --ignored --test-threads=1 2>&1 | tail -12
echo BASE_CONTROL_DONE
