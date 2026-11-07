#!/bin/bash
# gpu-1106d: the coverage rows still owed on the RTX 5090 after gpu-1106b/c (workflow wf_9c65ec76-bc2,
# reannot-1106/rows-vs-cells.json), on the fix-stack tip 99c65ab0b2 binaries. Each cell runs code a row
# names with the model and env that execute it, and records the engagement line that proves it.
set -u
W=<work>/engine-dsv4-glm
cd $W || exit 2
L=<work>/logs-gpu-1106d; mkdir -p $L; S=$L/status
C=<work>/cells-1106
BIN=<work>/logs-gpu-1106/memra-server-99c65ab0b20e
B6=<work>/logs-gpu-1106b
DBB=$B6/decode-batch-bench; DBG=$B6/decode-batch-gate
Q9=<hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra tip 1106 owed coverage rows: $why" --wait 2h -- "$@"; }
note() { echo "$*" >> $S; }
echo "start $(date '+%m-%d %H:%M')" > $S

# 1. engine cells (decode-batch-bench / decode-batch-gate from the tip build)
eng() { local tag=$1 ttl=$2; shift 2; G "$ttl" "$tag" "$@" > $L/$tag.log 2>&1; local rc=$?
  note "$tag rc=$rc :: $(grep -a -E 'BITWISE|bitwise|DIFFER|NOT POOLABLE|VERIFY T=|ENGAGED|gate[0-9]+ .*: (PASS|FAIL)|ALL GREEN|out of memory|arms \[' $L/$tag.log | sort | uniq -c | sort -rn | head -n 6 | tr '\n' '|' | cut -c1-400)"; }
eng C4-eager-sampled 20m $DBB $Q9 --pooled-draft 3,3,2,4 --temp 0.8 --all-sampled --pen
eng C4-eager-mixed 20m $DBB $Q9 --pooled-draft 3,3,2,4 --temp 0.8 --pen
eng C12-taps-4x4 30m $DBB $Q38 --pooled 4,4,4,4 --taps 5,19,33,47,61
eng C12-taps-8x8 30m $DBB $Q38 --pooled 8,8 --taps 5,19,33,47,61 --ctx 2048
eng C12-verify-t 20m $DBB $Q9 --verify-t 1,2,4,8,16
eng D2-q38-b24-tc1 30m env MEMRA_MMVQ_TC=1 MEMRA_DEBUG=1 $DBG $Q38 --mode config --batch 24 --steps 8
eng D4-q38-b24-tc0 30m env MEMRA_MMVQ_TC=0 MEMRA_DEBUG=1 $DBG $Q38 --mode config --batch 24 --steps 8
eng D2-q9-b24-tc1 30m env MEMRA_MMVQ_TC=1 MEMRA_DEBUG=1 $DBG $Q9 --mode config --batch 24 --steps 8
eng D2-q9-b32-tc1 30m env MEMRA_MMVQ_TC=1 MEMRA_DEBUG=1 $DBG $Q9 --mode config --batch 32 --steps 8

# 2. memra-probe target arms (B15): 121a must refuse a 12.0 card before the fatbin load; 100a warns
# and builds 120a. target-cuda is shared, so 120a is rebuilt at the end.
PB() { flock <cache>/memra-rig-build.lock nice -n 15 env CARGO_TARGET_DIR=<work>/target-cuda MEMRA_NVCC=/usr/local/cuda/bin/nvcc "$@"; }
for arch in 121a 100a 120a; do
  PB env MEMRA_CUDA_ARCH=$arch cargo build --release -p memra-probe -vv > $L/probe-build-$arch.log 2>&1
  note "probe-build-$arch rc=$? :: $(grep -a -o -E 'MEMRA_PROBE_CUDA_TARGET=[0-9a-z]+|builds only the sm_120 family[^"]*' $L/probe-build-$arch.log | sort -u | tr '\n' '|' | cut -c1-240)"
  [ $arch = 120a ] && break
  G 10m "memra-probe built for $arch" <work>/target-cuda/release/memra-probe > $L/probe-run-$arch.log 2>&1
  note "probe-run-$arch rc=$? :: $(grep -a -E 'compute capability|built for sm_121a|GREEN|panicked' $L/probe-run-$arch.log | head -n 3 | tr '\n' '|' | cut -c1-260)"
done

# 3. install.sh's 12.1 arm (D24), CPU only, stubbed nvidia-smi and curl
ST=$L/install-stub; mkdir -p $ST/bin $ST/dir
printf '#!/bin/sh\necho 12.1\n' > $ST/bin/nvidia-smi; printf '#!/bin/sh\nexit 22\n' > $ST/bin/curl; chmod +x $ST/bin/*
PATH=$ST/bin:$PATH MEMRA_VERSION=v0.0.0-test MEMRA_INSTALL_DIR=$ST/dir sh -x tools/install.sh > $L/install-121.log 2>&1
note "D24-install-121 rc=$? :: $(grep -a -E '^\+ ARCH=|download failed|has no memra build target' $L/install-121.log | head -n 3 | tr '\n' '|' | cut -c1-260)"

# 4. serving cells (serve_cells.py scenarios; each prints CELL lines into OUT/cells.txt)
srv() { local sc=$1 ttl=$2; shift 2; G "$ttl" "serve $sc" env "$@" python3 $C/serve_cells.py $sc $BIN $L/serve > $L/serve-$sc.log 2>&1
  note "serve-$sc rc=$? :: $(grep -a 'CELL' $L/serve-$sc.log | cut -c1-150 | tr '\n' '|' | cut -c1-900)"; }
srv leftbehind 1h X=1
srv pinned-sampled 1h X=1
srv bandemote 40m X=1
srv dflash-high 40m X=1
srv shed 40m X=1
srv restore-sampled 40m X=1
srv calib-nopool 40m X=1
G 1h "riders under a K pin (C2)" env MEMRA_SPEC_K=3 python3 $C/serve_cells.py riders $BIN $L/serve-c2pin > $L/serve-c2pin.log 2>&1
note "C2-riders-pin rc=$? :: $(grep -a -E 'CELL|INFO' $L/serve-c2pin.log | cut -c1-150 | tr '\n' '|' | cut -c1-500) calib-skipped=$(grep -a -c 'calibration skipped' $L/serve-c2pin/server-riders.log) pin=$(grep -a -c 'source=operator-pin' $L/serve-c2pin/server-riders.log)"
G 30m "riders plain ref, greedy rows (C2)" env RIDER_KEYS=c3/r0,c3/r2,c4/r0,c4/r2,c6/r0,c6/r2 python3 $C/serve_cells.py riders-plain-ref $BIN $L/serve-c2pin > $L/serve-c2pin-ref.log 2>&1
note "C2-riders-pin-plain-ref rc=$? :: $(grep -a 'CELL' $L/serve-c2pin-ref.log | cut -c1-110 | tr '\n' '|' | cut -c1-700)"
mkdir -p $L/serve-tc0; cp <work>/logs-gpu-1106b/serve/mtp-solo.json $L/serve-tc0/ 2>/dev/null
G 1h "wave32 with the tensor-core tier off (A12)" env MEMRA_MMVQ_TC=0 python3 $C/serve_cells.py wave32 $BIN $L/serve-tc0 > $L/serve-wave32-tc0.log 2>&1
note "A12-wave32-tc0 rc=$? :: $(grep -a 'CELL' $L/serve-wave32-tc0.log | cut -c1-140 | tr '\n' '|' | cut -c1-500)"
# D14: the vision directory's own checkpoint arm, through a same-filesystem hard link
VD=<models>/q38-vision/dirarm
mkdir -p $VD && { [ -e $VD/model.safetensors ] || ln <models>/q38-vision/outside.safetensors $VD/model.safetensors; }
G 1h "vision dir arm (D14)" env VIS_DIR=$VD python3 $C/serve_cells.py vision-ab $BIN $L/serve-vision-dir $BIN > $L/serve-vision-dir.log 2>&1
note "D14-vision-dir rc=$? :: loaded=$(grep -a -h -o 'tower loaded from [^ ]*' $L/serve-vision-dir/server-vision-tip.log | head -n 1) :: $(python3 -c "
import json; a=json.load(open('$L/serve-vision-dir/vision.json')); b=json.load(open('<work>/logs-gpu-1106b/serve/vision.json')); print('tip equal to the outside.safetensors run:', a['tip']==b['tip'])" 2>&1)"
rm -f $VD/model.safetensors; rmdir $VD 2>/dev/null

# 5. DSpark pool cell (C5) against the stored serial reference, and the engine tap twin ran in step 1
G 1h "DSpark pool cell (C5)" python3 research/batch-spec-20261101/battery/dspark_pool_cell.py $BIN $L/dspark-pool/tip > $L/dspark-pool.log 2>&1
note "C5-dspark-pool rc=$? :: cohort-sizes=$(grep -a -h -o 'pooled DFlash2 cohort engaged: [0-9]* sessions' $L/dspark-pool/tip/*.log 2>/dev/null | sort -u | grep -o '[0-9]* sessions' | tr '\n' ',') cap=$(grep -a -h -c 'capped by VRAM headroom' $L/dspark-pool/tip/*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null)"
python3 research/batch-spec-20261101/battery/compare_pool_runs.py research/batch-spec-20261101/dflash-pooled/cells/cell-serial tip=$L/dspark-pool/tip > $L/dspark-pool-compare.txt 2>&1
note "C5-dspark-pool-compare rc=$? :: $(grep -a -c -w SAME $L/dspark-pool-compare.txt) SAME, $(grep -a -c -w DIFF $L/dspark-pool-compare.txt) DIFF"

# 6. mixed rider cell without a cost table (A7), twice: engagement and run-to-run identity
for i in 1 2; do
  G 1h "mixed cell, no cost table, run $i (A7)" python3 research/batch-spec-20261101/battery/mixed_cell.py $BIN $L/mixed-$i > $L/mixed-$i.log 2>&1
  note "A7-mixed-$i rc=$? :: unified=$(grep -a -c 'unified step engaged' $L/mixed-$i/server.log) spec-cost-calibration=$(grep -a -c '\[spec-cost\] model="q9" depth=' $L/mixed-$i/server.log)"
done
python3 - $L/mixed-1/summary.json $L/mixed-2/summary.json >> $S 2>&1 <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
ka = {k: v["sha256"] for k, v in a.items() if isinstance(v, dict) and "sha256" in v}
kb = {k: v["sha256"] for k, v in b.items() if isinstance(v, dict) and "sha256" in v}
diff = sorted(k for k in ka if ka[k] != kb.get(k))
print(f"A7-mixed-reproducible :: {len(ka) - len(diff)}/{len(ka)} streams equal across runs; differ: {diff}")
PY

# 7. local-ci correctness with the GPU-only lib tests and the SKIP report (D23), last: it rebuilds
sccache --start-server > /dev/null 2>&1
PB env MEMRA_CUDA_ARCH=120a cargo build --release > $L/d23-prebuild.log 2>&1; note "D23-prebuild rc=$?"
PB env MEMRA_CUDA_ARCH=120a cargo test --release -p memra-engine --lib --no-run > $L/d23-prebuild-test.log 2>&1; note "D23-prebuild-test rc=$?"
G 3h "local-ci correctness (D23)" env MEMRA_CI_LOCK_HELD=1 CARGO_TARGET_DIR=<work>/target-cuda \
  MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_BUILD_JOBS=2 \
  MEMRA_CI_CLIPPY=0 MEMRA_CI_SAMPCHECK=0 MEMRA_CI_CONTGATE=0 MEMRA_CI_RUNSPEC=0 MEMRA_CI_PRIME_EXACT=0 MEMRA_CI_GWSTRESS=0 \
  MEMRA_CI_GRAPH=0 MEMRA_CI_SERVE=0 MEMRA_CI_HEALTH_FAULT=0 MEMRA_CI_STRESS=0 MEMRA_CI_FAULTGATE=0 MEMRA_CI_SPEC_CTX_EDGE=0 \
  MEMRA_CI_ADMIT_MEM_BURST=0 MEMRA_CI_FAIRGATE=0 MEMRA_CI_GPUPROBEGATE=0 MEMRA_CI_ACCEPT=0 MEMRA_CI_HITGATE=0 \
  bash tools/local-ci.sh --correctness > $L/local-ci.log 2>&1
note "D23-local-ci rc=$? :: $(grep -a -E 'local-ci: SKIP [0-9]+ GPU test|SKIP\[MIMO|GPU-only lib tests FAILED|local-ci: (PASS|FAIL|ALL)' $L/local-ci.log | head -n 4 | tr '\n' '|' | cut -c1-300)"
note "done $(date '+%m-%d %H:%M')"
