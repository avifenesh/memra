#!/bin/bash
# RTX 5090 re-validation of the fix stack tip after the 2026-11-06 restack onto main 5d70118b99
# (fix4/gates 99c65ab0b2). Same cells as the wave-4 top run (memra-fix4/top-cells.sh plus the
# sized host-spill gates of gates-cells.sh), with the #1069 door checks in place of the
# retired-names step: on the 5090 the three DSpark restore doors default OFF and =1 arms them.
# Each step appends one line to $S; logs and evidence stay under $L.
set -u
W=<work>/engine-dsv4-glm
TIP=99c65ab0b2
cd $W || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty" >&2; exit 2; }
git checkout -q --detach $TIP || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1106; mkdir -p $L; S=$L/status
H=$(git rev-parse HEAD); echo "head $H start $(date +%H:%M)" > $S
REL=$CARGO_TARGET_DIR/release
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix-stack tip 1106: $why" --wait 3h -- "$@"; }
res() { grep -o 'test result: [A-Za-z]*\. [0-9]* passed; [0-9]* failed' "$1" | awk '{p+=$4; f+=$6} END {printf "passed=%d failed=%d", p, f}'; }
Q38=<hf-models>/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
Q9=<hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
E4B=<hf-models>/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf
G26=<hf-models>/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf
G31=<hf-models>/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf
DRAFT=<work4>/dflash2-q38

# 1. builds. target-cuda also holds crates built by the Max-Q lane worktrees (same 0.138.0 names),
# so touch every tracked file inside the lock hold, then require a Compiling line from this
# worktree for every workspace crate the binaries link.
B bash -c "git ls-files -z | xargs -0 touch -c && cargo build --release -p memra-engine --bins -p memra-server --bin memra-server -p memra-tokenizer --bin tok-parity" > $L/build.log 2>&1; echo "build rc=$?" >> $S
need=$(cargo tree --offline -p memra-server -p memra-engine -p memra-tokenizer -e normal,build --prefix none 2>/dev/null | grep -o -E '^memra-[a-z0-9-]+' | sort -u)
miss=""; for c in $need; do grep -q -E "Compiling $c v[0-9.]+ \($W/crates/$c\)" $L/build.log || miss="$miss $c"; done
echo "closure: $(echo $need | wc -w) crates, not compiled from this worktree:${miss:- none}" >> $S
echo "stamp server=$(grep -a -c "${H:0:12}" $REL/memra-server)" >> $S
cp $REL/memra-server $L/memra-server-${H:0:12}
BIN=$L/memra-server-${H:0:12}
CUDA_VISIBLE_DEVICES= B cargo test --release -p memra-engine --lib > $L/engine-lib.log 2>&1; echo "engine-lib rc=$? $(res $L/engine-lib.log)" >> $S
CUDA_VISIBLE_DEVICES= B cargo test --release -p memra-server > $L/server.log 2>&1; echo "server rc=$? $(res $L/server.log)" >> $S
B cargo test --release -p memra-gguf > $L/gguf.log 2>&1; echo "gguf rc=$? $(res $L/gguf.log)" >> $S

# 2. GPU tests (one lease; each test file reports on its own)
GT="e4m3_batch_r2_gpu q80_batch_r2_gpu mla_kpool_select_v2_gpu glm5_kpool_indexer_gpu glm5_matvec_doors_gpu glm5_decode_graph_capture_gpu kda_onorm_zq8_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu tp_ar_gpu gemma4_chunked_prime_gpu tcol_twins_gpu mla_decode_split_gpu glm5_moe_loc_doors_gpu mla_gpu_forward"
args=(); for t in $GT; do args+=(--test "$t"); done
B cargo test --release -p memra-engine "${args[@]}" --no-run > $L/gpu-tests-build.log 2>&1; echo "gpu-tests-build rc=$?" >> $S
G 2h "GPU tests" env NVIDIA_TF32_OVERRIDE=0 cargo test --release --no-fail-fast -p memra-engine "${args[@]}" -- --ignored --test-threads 1 > $L/gpu-tests.log 2>&1
echo "gpu-tests rc=$? $(res $L/gpu-tests.log) failed-files: $(grep -E '^error: test failed' $L/gpu-tests.log | tr '\n' ' ' | cut -c1-300)" >> $S
G 30m "DSpark fault matrix" env MEMRA_DSPARK_PREFIX_RESTORE=1 cargo test --release -p memra-server --lib worker::tests::dflash_retained_gpu_plan_fault_matrix -- --ignored --exact > $L/dspark-fault-matrix.log 2>&1
echo "dspark-fault-matrix rc=$? $(res $L/dspark-fault-matrix.log)" >> $S

# 3. release battery and fast-gate goldens
G 4h "release battery" bash tools/release-battery.sh --evidence-dir $L/battery-${H:0:12} > $L/battery.log 2>&1
echo "battery rc=$? $(grep -m1 -oE 'RELEASE BATTERY (PASS|FAIL)' $L/battery.log) $(grep -m1 -oE 'kernel-check +[A-Z]+ +[^(]*\([^)]*\)' $L/battery.log)" >> $S
G 1h "fast-gate goldens" nice -n 15 bash tools/fast-gate/fast-gate.sh --tier 1 --probes q9,q35,q35slru,q35spec,g31,g26,e4b > $L/fastgate.log 2>&1
echo "fast-gate rc=$? $(grep -E 'kernel-check \(|: (PASS|FAIL)' $L/fastgate.log | sed 's/(gates green + golden token-identical, [0-9]*s)//' | tr '\n' ' ' | cut -c1-400)" >> $S

# 4. serving: default serve-smoke, then the gemma4 arm on E4B, 26B-A4B and 31B
G 1h "serve-smoke" bash tools/serve-smoke.sh > $L/serve-smoke.log 2>&1
echo "serve-smoke rc=$? $(grep -c '  ok: ' $L/serve-smoke.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke.log) FAIL" >> $S
for gm in e4b:$E4B g26:$G26 g31:$G31; do
    t=${gm%%:*}
    G 1h "serve-smoke gemma4 $t" env GEMMA_MODEL=${gm#*:} bash tools/serve-smoke.sh > $L/serve-smoke-$t.log 2>&1
    echo "serve-smoke-$t rc=$? $(grep -c '  ok: ' $L/serve-smoke-$t.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke-$t.log) FAIL, gemma4 FAIL lines: $(grep -E 'gemma4' $L/serve-smoke-$t.log | grep -c 'FAIL:')" >> $S
done

# 5. the E4B thought-stream census on this binary
census() {
    env MEMRA_COMPAT=openai "MEMRA_MODELS=e4b=$E4B" MEMRA_ADDR=127.0.0.1:18173 MEMRA_CTX=8192 $BIN > $L/census-server.log 2>&1 &
    pid=$!
    waitfor --http http://127.0.0.1:18173/v1/models --timeout 10m > /dev/null 2>&1 && python3 <work4>/gemma-census-client.py 18173 $L/census > $L/census.log 2>&1
    kill $pid; wait $pid 2>/dev/null
}
export -f census; export L E4B BIN
G 40m "E4B census" bash -c census; echo "census-e4b $(grep SUMMARY $L/census.log | cut -c1-400)" >> $S

# 6. DSpark (#1069 doors): the strict-prefix cell with only the prefix restore armed; the smoke on
# the 5090 default (doors OFF) with its boot line; the sized host-spill gates and the drafter arm.
G 45m "DSpark strict-prefix cell" env EV=$L/dspark-cell BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=1" bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell.log 2>&1
echo "dspark-cell rc=$? $(grep -E 'DSPARK-PAIR CELL|strict-prefix restores' $L/dspark-cell.log | tr '\n' ' ') boot-line: $(grep -a -h -m1 -o 'prefix restore doors default [^\\]*' $L/dspark-cell/warm-server.log | cut -c1-200)" >> $S
G 1h "DSpark serve smoke" bash tools/dspark-serve-smoke.sh $Q38 $DRAFT $BIN $L/dspark-smoke > $L/dspark-smoke.log 2>&1
echo "dspark-smoke rc=$? $(grep -E 'FAIL|on-r1' $L/dspark-smoke.log | tr '\n' ' ' | cut -c1-200) boot-line: $(grep -a -r -h -m1 -o 'prefix restore doors default [^\\]*' $L/dspark-smoke | head -n 1 | cut -c1-200)" >> $S
G 40m "host spill identity (sized)" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q9 $BIN $L/identity > $L/identity.log 2>&1
echo "identity rc=$? $(grep -E 'sized:|GATE' $L/identity.log | tr '\n' ' ' | cut -c1-300)" >> $S
G 40m "host spill identity, drafter arm (sized)" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_DSPARK_PREFIX_RESTORE=1 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark > $L/identity-dspark.log 2>&1
echo "identity-dspark rc=$? $(grep -E 'sized:|GATE' $L/identity-dspark.log | tr '\n' ' ' | cut -c1-300)" >> $S
G 40m "host spill failure (sized)" bash tools/kv-host-spill-failure-gate.sh --external-lock 9 $Q9 $BIN $L/failure > $L/failure.log 2>&1
echo "failure rc=$? $(grep -E 'GATE|FAIL' $L/failure.log | tr '\n' ' ' | cut -c1-300)" >> $S

# 7. E4B engine cells and the health-fault gate (arm j must SKIP on this card)
G 1h "E4B cells" bash <cache>/memra-fix3/e4b-fixlogs/scripts/e4b_gpu_cells.sh $W $CARGO_TARGET_DIR <cache>/memra-fix3/e4b-fixlogs/inputs $L/e4b > $L/e4b.log 2>&1
echo "e4b rc=$?" >> $S
grep -E '^(rungen|argmax|spec K=(1|8)|verify|e4b-graph)' $L/e4b/summary.txt | cut -c1-140 | sed 's/^/  e4b /' >> $S
G 1h "health-fault gate" bash tools/health-fault-gate.sh > $L/health-fault.log 2>&1
echo "health-fault rc=$? $(grep -c -E 'PASS' $L/health-fault.log) PASS lines, $(grep -c -E 'FAIL' $L/health-fault.log) FAIL lines, $(grep -c SKIP $L/health-fault.log) SKIP lines" >> $S
echo "done $(date +%H:%M)" >> $S
