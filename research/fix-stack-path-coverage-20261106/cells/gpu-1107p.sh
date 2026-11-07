#!/bin/bash
# After gpu-1107n.sh, same worktree, tip and binaries: the two steps that run refused there.
# fast-gate's named probes without the four whose model is not on this box (g12, g12d: no 12B QAT;
# k27: no Qwen3.6-27B NVFP4; kat: no KAT-Coder; a named probe with no model refuses the whole run),
# and the probe recovery gate standalone with local-ci's inner-lock seam (it takes the 5090 lock
# itself, which gpulock holds).
set -u
W=<work>/engine-dsv4-glm
cd $W || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1107n; S=$L/status
H=$(git rev-parse HEAD); BIN=$L/memra-server-${H:0:12}
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix-stack final tip 1107 (onto cf85a23ff5): $why" --wait 3h -- "$@"; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
note "gpu-1107p start at head ${H:0:12}"
G 30m "GPU probe recovery gate (fixed), inner lock" env MEMRA_GPU_LOCK=/tmp/memra-5090.lock.inner python3 tools/gpu-probe-recovery-gate.py --model <hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf --bin $BIN --out $L/probe-gate-2 > $L/probe-gate-2.log 2>&1
note "probe-gate-2 rc=$? $(grep -E 'A recover|B latch|C fatal|GPU-PROBE-RECOVERY|REFUSED' $L/probe-gate-2.log | tr '\n' ' ' | cut -c1-500) :: $(grep -h armed $L/probe-gate-2/*/fake-smi-trace.txt 2>/dev/null | tr '\n' ' ')"
P=q9,q35,q35slru,q35spec,g31,g26,e4b,chunkinv,chunkinvc,ptick,ptickc,pbg9,pbg9c,amargin,amarginc,amarginb,amarginbc,gwstress,isogap,isogapc,accept,sstress
G 2h "fast-gate named probes (prime, argmax, spec)" nice -n 15 bash tools/fast-gate/fast-gate.sh --tier 1 --probes $P > $L/fastgate-2.log 2>&1
note "fast-gate-2 rc=$? $(grep -E 'kernel-check \(|: (PASS|FAIL|SKIP)|REFUSED' $L/fastgate-2.log | sed 's/(gates green + golden token-identical, [0-9]*s)//' | tr '\n' ' ' | cut -c1-1400)"
note "gpu-1107p done"
