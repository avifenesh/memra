#!/bin/bash
# RTX 5090 check of the argmax-margin gate as merged at the restack onto main a5a84b3373 (#1081's
# per-prompt flip budget beside the stack's batch decode arm), at the stack top. The top differs from
# the 1a6b6eb671 runs by tools/ and the four DSv4 sibling gates only, so this re-runs the gate's own
# consumers: fast-gate's four argmax-margin probes and the release battery (argmax + run-spec rows).
set -u
TIP=$1
W=<work>/engine-dsv4-glm
cd $W || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty" >&2; exit 2; }
git fetch -q <work>/restack-1105 fix4/gates || exit 2
git checkout -q --detach $TIP || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1107s; mkdir -p $L; S=$L/status
H=$(git rev-parse HEAD); echo "head $H start $(date '+%m-%d %H:%M')" > $S
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix-stack top (onto a5a84b3373): $why" --wait 2h -- "$@"; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
B bash -c "git ls-files -z | xargs -0 touch -c && cargo build --release -p memra-engine --bins -p memra-server --bin memra-server" > $L/build.log 2>&1; note "build rc=$?"
G 1h "fast-gate argmax-margin probes" nice -n 15 bash tools/fast-gate/fast-gate.sh --tier 1 --probes amargin,amarginb,amarginbc,amarginc > $L/fastgate.log 2>&1
note "fast-gate rc=$? $(grep -E ': (PASS|FAIL|SKIP)' $L/fastgate.log | tr '\n' ' ' | cut -c1-500)"
G 2h "release battery" bash tools/release-battery.sh --evidence-dir $L/battery-${H:0:12} > $L/battery.log 2>&1
note "battery rc=$? $(grep -m1 -oE 'RELEASE BATTERY (PASS|FAIL)' $L/battery.log) :: $(grep -a -h -o 'flip budget: [0-9]* (row: [^)]*)' $L/battery.log $L/battery-${H:0:12}/*/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-300)"
note "done"
