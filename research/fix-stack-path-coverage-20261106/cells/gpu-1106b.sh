#!/bin/bash
# Path-driven 5090 cells for the fix-stack tip 99c65ab0b2, from the 2026-11-06 coverage review
# (reannot-1106/coverage-report.txt): each cell below executes changed code that no gpu-1106 cell
# reached or checked, or replaces a gpu-1106 cell whose env could not execute its path after #1069.
# Starts when gpu-1106.sh has finished its serving step (census line), then stops gpu-1106.sh before
# its stale DSpark and gemma cells.
set -u
W=<work>/engine-dsv4-glm
cd $W || exit 2
[ "$(git rev-parse --short=10 HEAD)" = 99c65ab0b2 ] || { echo "worktree not at the tip" >&2; exit 2; }
L=<work>/logs-gpu-1106b; mkdir -p $L; S=$L/status
C=<work>/cells-1106
export CARGO_TARGET_DIR=<work>/target-cuda MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_BUILD_JOBS=2
REL=$CARGO_TARGET_DIR/release
L1=<work>/logs-gpu-1106
BIN=$L1/memra-server-99c65ab0b20e
BIN_OLD=<work4>/logs-top/memra-server-196596e5d7e3
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra tip 1106 path cells: $why" --wait 3h -- "$@"; }
res() { grep -o 'test result: [A-Za-z]*\. [0-9]* passed; [0-9]* failed' "$1" | awk '{p+=$4; f+=$6} END {printf "passed=%d failed=%d", p, f}'; }
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
Q9=<hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q35=<hf-models>/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ORN=<hf-models>/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
DRAFT=<work4>/dflash2-q38
DOORS3="MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1"

echo "wait for gpu-1106 serving step $(date +%H:%M)" > $S
waitfor --grep 'census-e4b' --file $L1/status --timeout 6h --every 30s > /dev/null 2>&1 || { echo "gpu-1106 did not reach the census step" >> $S; exit 1; }
for p in $(pgrep -P 3878634); do kill $p; done; kill 3878634 2>/dev/null
echo "gpu-1106 stopped after census $(date +%H:%M); its steps 6-7 are replaced below" >> $S
waitfor --until 'gpulock status | grep -q "5090: free"' --timeout 10m --every 10s > /dev/null 2>&1
cp $REL/decode-batch-bench $REL/decode-batch-gate $REL/kernel-check $REL/f8f4_check $L/
echo "bins copied from target-cuda (built at the tip by gpu-1106 step 1)" >> $S

# 1. DSpark cells with the env #1069 needs on the 5090 (the doors default OFF here)
G 30m "DSpark fault matrix, three doors" env $DOORS3 cargo test --release -p memra-server --lib worker::tests::dflash_retained_gpu_plan_fault_matrix -- --ignored --exact > $L/fault-matrix.log 2>&1
echo "fault-matrix rc=$? $(res $L/fault-matrix.log)" >> $S
G 20m "kda onorm zq8 (not an ignored test)" cargo test --release -p memra-engine --test kda_onorm_zq8_gpu > $L/kda-onorm.log 2>&1
echo "kda-onorm rc=$? $(res $L/kda-onorm.log)" >> $S
G 50m "DSpark strict-prefix pair, three doors" env EV=$L/dspark-cell BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell.log 2>&1
echo "dspark-cell rc=$? $(grep -E 'DSPARK-PAIR CELL|strict-prefix restores' $L/dspark-cell.log | tr '\n' ' ' | cut -c1-200)" >> $S
G 1h "DSpark cold-drafter gate" bash research/dflash-restorable-hit-20261103/cold-drafter-gate.sh $BIN $L/cold-drafter > $L/cold-drafter.log 2>&1
echo "cold-drafter rc=$? $(grep -E 'PASS|FAIL|identical|cold-drafter restore' $L/cold-drafter.log | tail -n 3 | tr '\n' ' ' | cut -c1-300)" >> $S
G 1h "DSpark serve smoke" bash tools/dspark-serve-smoke.sh $Q38 $DRAFT $BIN $L/dspark-smoke > $L/dspark-smoke.log 2>&1
echo "dspark-smoke rc=$? $(grep -E 'FAIL|on-r1' $L/dspark-smoke.log | tr '\n' ' ' | cut -c1-200)" >> $S
python3 $C/dspark_post.py $L/dspark-smoke >> $S 2>&1
G 1h "DSpark serve smoke, no MTP head" env MEMRA_MTP_SKIP=1 bash tools/dspark-serve-smoke.sh $Q38 $DRAFT $BIN $L/dspark-smoke-mtpskip > $L/dspark-smoke-mtpskip.log 2>&1
echo "dspark-smoke-mtpskip rc=$? admit lines: $(grep -a -h -E '\[admit\]' $L/dspark-smoke-mtpskip/serve-on.log 2>/dev/null | head -n 2 | cut -c1-160 | tr '\n' ' ')" >> $S
G 40m "host spill identity (sized)" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q9 $BIN $L/identity > $L/identity.log 2>&1
echo "identity rc=$? $(grep -E 'sized:|GATE' $L/identity.log | tr '\n' ' ' | cut -c1-300)" >> $S
G 40m "host spill identity, drafter arm, three doors" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT $DOORS3 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark > $L/identity-dspark.log 2>&1
echo "identity-dspark rc=$? $(grep -E 'sized:|GATE' $L/identity-dspark.log | tr '\n' ' ' | cut -c1-300)" >> $S
G 40m "host spill failure (sized)" bash tools/kv-host-spill-failure-gate.sh --external-lock 9 $Q9 $BIN $L/failure > $L/failure.log 2>&1
echo "failure rc=$? $(grep -E 'GATE|FAIL' $L/failure.log | tr '\n' ' ' | cut -c1-300)" >> $S

# 2. serving cells for the MTP cost gate, pooled cohort, riders, 32-row waves, sampled demotion, vision
for sc in mtp-pool plain-ref wave32 dflash-demote; do
  G 1h "serve cells $sc" python3 $C/serve_cells.py $sc $BIN $L/serve > $L/serve-$sc.log 2>&1
  echo "serve-$sc rc=$? $(grep -c ' PASS' $L/serve-$sc.log) pass, $(grep -c ' FAIL' $L/serve-$sc.log) fail: $(grep ' FAIL' $L/serve-$sc.log | cut -c1-120 | tr '\n' ' ')" >> $S
done
G 1h "serve cells vision a1e9 vs tip" python3 $C/serve_cells.py vision-ab $BIN $L/serve $BIN_OLD > $L/serve-vision-ab.log 2>&1
echo "serve-vision-ab rc=$? $(grep 'CELL' $L/serve-vision-ab.log | cut -c1-200 | tr '\n' ' ')" >> $S
sed "s|^BIN = .*|BIN = \"$BIN\"|" research/batch-spec-20261101/sampled_demote_cell.py > $C/sampled_demote_cell.tip.py
G 1h "MTP sampled demotion" env DEMOTE_AT=40 python3 $C/sampled_demote_cell.tip.py > $L/sampled-demote.log 2>&1
echo "sampled-demote rc=$? $(tail -n 3 $L/sampled-demote.log | tr '\n' ' ' | cut -c1-300)" >> $S

# 3. engine bitwise cells: pooled verify and draft, unified step, pooled-plain, exact 32-row tier
dbb() { local tag=$1; shift; G 30m "decode-batch-bench $tag" $L/decode-batch-bench "$@" > $L/dbb-$tag.log 2>&1; local rc=$?
  local eq=$(grep -c 'BITWISE EQUAL' $L/dbb-$tag.log) bad=$(grep -c -E 'MISMATCH|NOT POOLABLE' $L/dbb-$tag.log)
  echo "dbb-$tag rc=$rc equal=$eq bad=$bad $([ "$eq" -gt 0 ] && [ "$bad" = 0 ] && [ $rc = 0 ] && echo PASS || echo FAIL)" >> $S; }
dbb q9-pooled-4x4 $Q9 --pooled 4,4,4,4
dbb q9-pooled-ragged $Q9 --pooled 3,5,2,6
dbb q9-pooled-draft $Q9 --pooled-draft 3,3,2,4
dbb q9-pooled-draft-sampled $Q9 --pooled-draft 3,3,2,4 --temp 0.8 --all-sampled --top-p 0.9
dbb q9-unified-43 $Q9 --unified 4,3 --plain 6 --kinds gtfph
dbb q9-unified-333 $Q9 --unified 3,3,3 --plain 4
dbb q9-pooled-plain $Q9 --pooled-plain 8
dbb q35-pooled $Q35 --pooled 4,4,4,4
dbb q38-pooled $Q38 --pooled 4,4,4,4
for m in q9:$Q9 q38:$Q38; do
  for tc in 0 1; do for ch in 32 16; do
    G 30m "exact tier ${m%%:*} tc=$tc chunk=$ch" env MEMRA_DBB_SAMP=greedy MEMRA_MMVQ_TC=$tc $L/decode-batch-bench ${m#*:} --seqs 32 --steps 32 --chunk $ch > $L/tokfp-${m%%:*}-tc$tc-c$ch.log 2>&1
    echo "tokfp ${m%%:*} tc=$tc chunk=$ch rc=$? $(grep -m1 -o 'TOKFP.*' $L/tokfp-${m%%:*}-tc$tc-c$ch.log | sed 's/chunk=[0-9]* //' | sha256sum | cut -c1-12)" >> $S
  done; done
done
for spec in q9:$Q9:24 q9:$Q9:32 q38:$Q38:8 q38:$Q38:16 q38:$Q38:24 q38:$Q38:32; do
  IFS=: read -r tag path b <<< "$spec"
  G 30m "decode-batch-gate $tag B=$b" $L/decode-batch-gate $path --mode config --batch $b > $L/dbg-$tag-b$b.log 2>&1
  echo "dbg $tag B=$b rc=$? FAIL-lines=$(grep -c ' FAIL' $L/dbg-$tag-b$b.log)" >> $S
done

# 4. kernels and tools the review found unexercised
G 1h "kernel-check tensor-core b32" env MEMRA_MMVQ_TC=1 $L/kernel-check $ORN > $L/kernel-check-tc.log 2>&1
echo "kernel-check-tc rc=$? $(grep -m1 -E 'ALL GREEN|FAIL' $L/kernel-check-tc.log | cut -c1-160)" >> $S
for f8 in unset 1; do
  if [ $f8 = unset ]; then G 20m "f8f4-check default" $L/f8f4_check $Q9 > $L/f8f4-$f8.log 2>&1; else G 20m "f8f4-check forced" env MEMRA_MMQ_F8F4=1 $L/f8f4_check $Q9 > $L/f8f4-$f8.log 2>&1; fi
  echo "f8f4-check $f8 rc=$? $(tail -n 2 $L/f8f4-$f8.log | tr '\n' ' ' | cut -c1-200)" >> $S
done
B cargo build --release -p memra-probe > $L/probe-build.log 2>&1; echo "probe-build rc=$?" >> $S
G 10m "memra-probe 120a" env MEMRA_CUDA_ARCH=120a $REL/memra-probe > $L/probe-120a.log 2>&1
echo "probe-120a rc=$? $(grep -m1 -o 'compute capability: [0-9.]*' $L/probe-120a.log)" >> $S
B env -u MEMRA_CUDA_ARCH cargo build --release -p memra-probe -vv > $L/probe-build-auto.log 2>&1; echo "probe-build-auto rc=$? $(grep -m1 -o -E 'MEMRA_CUDA_ARCH[^"]*(auto|unset)[^"]{0,80}|sm_12[01]a' $L/probe-build-auto.log | head -n 1)" >> $S
G 10m "memra-probe auto-detected arch" $REL/memra-probe > $L/probe-auto.log 2>&1
echo "probe-auto rc=$? $(grep -m1 -o 'compute capability: [0-9.]*' $L/probe-auto.log)" >> $S

# 5. load: the only cell past 16 served rows with pooled cohorts and sampled demotion (liveness)
G 1h "serve stress c=64" bash tools/serve-stress-gate.sh > $L/sstress.log 2>&1
echo "sstress rc=$? $(tail -n 2 $L/sstress.log | tr '\n' ' ' | cut -c1-200)" >> $S
echo "done $(date +%H:%M)" >> $S
