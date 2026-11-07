#!/bin/bash
# RTX 5090 re-validation of the fix-stack tip after the 2026-11-07 restacks onto main c2e04719b6 and 65d5887596 (#1078)
# (#1073 GB10 prefill levers incl. the h2 prime twin and the prime FFN changes, #1074 cold-drafter
# capture, #1075 wave-probe trim + fit checks, #1076 n>1 extended-stream deadline), and the rows
# that waited on #1074 (A1, A2, A17: DSpark restorable-hit preference, retained-plan route).
# The cells follow the changed paths: fast-gate's own plan for the delta from the pre-restack tip
# (bbcfec5517) picks the probes; the hand-resolved prime glue (hybrid_forward.rs fo_owned) and
# the flash_attn.cu template rewrite run in the GPU tests, kernel-check with the h2 twin armed,
# the prime gates and local-ci's full battery.
#   gpu-1107.sh <tip sha>
set -u
TIP=$1
W=<work>/engine-dsv4-glm
OLDTIP=bbcfec55173a4e850514bc391831e529c66e7796
cd $W || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty" >&2; exit 2; }
git fetch -q <work>/restack-1105 fix4/gates || exit 2
git checkout -q --detach $TIP || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1107; mkdir -p $L; S=$L/status
C=<work>/cells-1106
H=$(git rev-parse HEAD); echo "head $H start $(date '+%m-%d %H:%M')" > $S
REL=$CARGO_TARGET_DIR/release
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix-stack tip 1107 (restack onto 65d5887596): $why" --wait 3h -- "$@"; }
res() { grep -o 'test result: [A-Za-z]*\. [0-9]* passed; [0-9]* failed' "$1" | awk '{p+=$4; f+=$6} END {printf "passed=%d failed=%d", p, f}'; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
Q9=<hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=<cache>/memra-fix4/dflash2-q38
OLDBIN=<work>/logs-gpu-1106/memra-server-99c65ab0b20e

# 1. builds (touch inside the lock hold: target-cuda is shared with the lane worktrees), closure, stamp
B bash -c "git ls-files -z | xargs -0 touch -c && cargo build --release -p memra-engine --bins -p memra-server --bin memra-server -p memra-tokenizer --bin tok-parity" > $L/build.log 2>&1; note "build rc=$?"
need=$(cargo tree --offline -p memra-server -p memra-engine -p memra-tokenizer -e normal,build --prefix none 2>/dev/null | grep -o -E '^memra-[a-z0-9-]+' | sort -u)
miss=""; for c in $need; do grep -q -E "Compiling $c v[0-9.]+ \($W/crates/$c\)" $L/build.log || miss="$miss $c"; done
note "closure: $(echo $need | wc -w) crates, not compiled from this worktree:${miss:- none}"
note "stamp server=$(grep -a -c "${H:0:12}" $REL/memra-server)"
cp $REL/memra-server $L/memra-server-${H:0:12}; cp $REL/kernel-check $L/kernel-check-${H:0:12}
BIN=$L/memra-server-${H:0:12}
CUDA_VISIBLE_DEVICES= B cargo test --release -p memra-engine --lib > $L/engine-lib.log 2>&1; note "engine-lib rc=$? $(res $L/engine-lib.log)"
CUDA_VISIBLE_DEVICES= B cargo test --release -p memra-server > $L/server.log 2>&1; note "server rc=$? $(res $L/server.log)"

# 2. GPU tests: the prime twins and the 1106 list; the h2 byte gate alone with its door set
GT="gemma4_chunked_prime_gpu tcol_twins_gpu e4m3_batch_r2_gpu q80_batch_r2_gpu mla_kpool_select_v2_gpu glm5_kpool_indexer_gpu glm5_matvec_doors_gpu glm5_decode_graph_capture_gpu kda_onorm_zq8_gpu kda_fused_proj_gpu kda_fused_proj_bf16_gpu glm5_tparallel_verify_gpu glm5_spec_session_gpu glm5_dflash_session_gpu tp_ar_gpu mla_decode_split_gpu glm5_moe_loc_doors_gpu mla_gpu_forward"
args=(); for t in $GT prime_kv_h2_gpu; do args+=(--test "$t"); done
B cargo test --release -p memra-engine "${args[@]}" --no-run > $L/gpu-tests-build.log 2>&1; note "gpu-tests-build rc=$?"
args=(); for t in $GT; do args+=(--test "$t"); done
G 2h "GPU tests" env NVIDIA_TF32_OVERRIDE=0 cargo test --release --no-fail-fast -p memra-engine "${args[@]}" -- --ignored --test-threads 1 > $L/gpu-tests.log 2>&1
note "gpu-tests rc=$? $(res $L/gpu-tests.log) failed-files: $(grep -E '^error: test failed' $L/gpu-tests.log | tr '\n' ' ' | cut -c1-300)"
G 20m "h2 prime twin byte gate" env MEMRA_PRIME_KV_H2=1 NVIDIA_TF32_OVERRIDE=0 cargo test --release -p memra-engine --test prime_kv_h2_gpu -- --ignored --test-threads 1 > $L/h2-gate.log 2>&1
note "h2-byte-gate rc=$? $(res $L/h2-gate.log)"
G 30m "kernel-check FULL, h2 twin armed" env MEMRA_PRIME_KV_H2=1 $L/kernel-check-${H:0:12} > $L/kc-h2.log 2>&1
note "kernel-check-h2 rc=$? $(grep -a -E 'ALL GREEN|FAIL' $L/kc-h2.log | tail -n 2 | tr '\n' ' ' | cut -c1-200)"

# 3. fast-gate's plan for the delta from the pre-restack tip (tier 1 probes, goldens)
G 2h "fast-gate plan probes for the restack delta" nice -n 15 bash tools/fast-gate/fast-gate.sh --diff $OLDTIP --tier 1 > $L/fastgate.log 2>&1
note "fast-gate rc=$? $(grep -E 'kernel-check \(|: (PASS|FAIL|SKIP)' $L/fastgate.log | sed 's/(gates green + golden token-identical, [0-9]*s)//' | tr '\n' ' ' | cut -c1-900)"

# 4. DSpark rows that waited on #1074, and the #1069 cells
srv() { local sc=$1 ttl=$2; shift 2; G "$ttl" "serve $sc" env "$@" python3 $C/serve_cells.py $sc $BIN $L/serve > $L/serve-$sc.log 2>&1
  note "serve-$sc rc=$? :: $(grep -a 'CELL' $L/serve-$sc.log | cut -c1-200 | tr '\n' '|' | cut -c1-1200)"; }
srv restorable 1h X=1
G 1h "cold-drafter gate (A2)" env GATE=$W/tools/dspark-cold-drafter-gate.py DF=$DRAFT bash $C/cold-drafter-gate-1107.sh $BIN $L/cold-drafter > $L/cold-drafter.log 2>&1
note "cold-drafter rc=$? :: $(grep -a -E 'GATE|identical|=[0-9]+$|SHAPE|refusing|another process' $L/cold-drafter.log | tr '\n' ' ' | cut -c1-500)"
retained() { # tag doors(1|0): rig5090_ab.sh's agentic workload at c1, T=0
  local tag=$1 d=$2 D=$L/retained-$1
  mkdir -p $D
  if ss -ltn | grep -q ':8093 '; then echo "port 8093 busy" > $D/refused; return 4; fi
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$Q38" MEMRA_ADDR=127.0.0.1:8093 MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=1 \
    MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT \
    MEMRA_FRSPEC_TRIM=<models>/q38-gguf/q38-ranks-sxc32768.gguf.txt MEMRA_PREFIX_STABLE_BOUNDARY=0 \
    MEMRA_DSPARK_PREFIX_RESTORE=$d MEMRA_DSPARK_PARTIAL_RESTORE=$d MEMRA_DSPARK_BOUNDARY_CAPTURE=$d \
    $BIN > $D/serve.log 2>&1 &
  local sp=$!
  waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet && grep -a -q listening $D/serve.log \
    && (cd <cache>/model-scout/harness && CLIENT_MAX_TIME=1800 BENCH_BASE=http://127.0.0.1:8093 BENCH_MODEL=q38 \
        BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 timeout 3600 python3 ab_agent.py run \
        <cache>/memra-1025/ab-workload-5090.json $D $tag --users 1 --think 1 --temperature 0 --seed 1234 > $D/cells.log 2>&1)
  local rc=$?
  kill -TERM $sp; waitfor --gone $sp --timeout 90s --quiet || kill -KILL $sp; wait $sp 2>/dev/null
  return $rc
}
export -f retained; export L Q38 DRAFT BIN
G 40m "retained-plan route, doors on (A1/A17c)" bash -c 'retained doors1 1'; r1=$?
G 40m "retained-plan route, doors off (A1/A17c)" bash -c 'retained doors0 0'; r0=$?
note "retained rc=$r1/$r0 :: plan=$(grep -a -c 'retained prefix plan' $L/retained-doors1/serve.log) result=$(grep -a -h -o 'retained prefix result: route=[A-Za-z]*[^)]*) admitted=[a-z]*' $L/retained-doors1/serve.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200) dspark-restore=$(grep -a -c 'DSPARK restore' $L/retained-doors1/serve.log) off-arm-restores=$(grep -a -c 'DSPARK restore' $L/retained-doors0/serve.log)"
python3 - $L/retained-doors1 $L/retained-doors0 >> $S 2>&1 <<'PY'
import glob, json, os, sys
def texts(d):
    f = glob.glob(os.path.join(d, "*.texts.json"))
    if not f:
        return {}
    return {(r["session"], r["turn"]): json.dumps([r.get("reasoning"), r.get("content"), r.get("tool_calls")]) for r in json.load(open(f[0]))}
a, b = (texts(d) for d in sys.argv[1:3])
common = sorted(set(a) & set(b), key=str)
diff = [k for k in common if a[k] != b[k]]
print(f"retained-bytes :: {len(common) - len(diff)}/{len(common)} turns equal doors-on vs doors-off (on={len(a)} off={len(b)}); first differing: {diff[:3]}")
PY
G 45m "DSpark strict-prefix cell" env EV=$L/dspark-cell BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=1" bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell.log 2>&1
note "dspark-cell rc=$? $(grep -E 'DSPARK-PAIR CELL|strict-prefix restores' $L/dspark-cell.log | tr '\n' ' ')"
G 1h "DSpark serve smoke" bash tools/dspark-serve-smoke.sh $Q38 $DRAFT $BIN $L/dspark-smoke > $L/dspark-smoke.log 2>&1
note "dspark-smoke rc=$? $(grep -E 'FAIL|on-r1' $L/dspark-smoke.log | tr '\n' ' ' | cut -c1-200)"

# 5. #1075 on the stack: the 27B + DFlash2 floor and admission, old tip binary vs this tip
G 1h "floor A/B (#1075)" python3 $C/serve_cells.py floor-ab $BIN $L/serve-floor $OLDBIN > $L/serve-floor.log 2>&1
note "floor-ab rc=$? :: $(grep -a 'CELL' $L/serve-floor.log | cut -c1-200 | tr '\n' '|' | cut -c1-600) :: $(grep -a -h -o 'transient floor [0-9]*MB' $L/serve-floor/server-floor-*.log 2>/dev/null | tr '\n' ' ')"

# 5b. #1078 on the stack: a default Ornith-1.5 (qwen35_moe + MTP) boot, the 1106 binary (panicked) vs this tip
G 40m "Ornith default boot A/B (#1078)" python3 $C/serve_cells.py orn-boot-ab $BIN $L/serve-orn $OLDBIN > $L/serve-orn.log 2>&1
note "orn-boot-ab rc=$? :: $(grep -a 'CELL' $L/serve-orn.log | cut -c1-220 | tr '\n' '|' | cut -c1-600)"

# 6. host spill identity (sized): restore = prime on the prime path main changed
G 40m "host spill identity (sized)" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q9 $BIN $L/identity > $L/identity.log 2>&1
note "identity rc=$? $(grep -E 'sized:|GATE' $L/identity.log | tr '\n' ' ' | cut -c1-300)"
G 40m "host spill identity, drafter arm (sized)" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_DSPARK_PREFIX_RESTORE=1 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark > $L/identity-dspark.log 2>&1
note "identity-dspark rc=$? $(grep -E 'sized:|GATE' $L/identity-dspark.log | tr '\n' ' ' | cut -c1-300)"

# 7. the full battery the plan asks for (fast-gate tier 2 = local-ci.sh), then the release battery
G 4h "local-ci full battery (fast-gate tier 2)" nice -n 15 env MEMRA_CI_LOCK_HELD=1 bash tools/local-ci.sh > $L/local-ci.log 2>&1
note "local-ci rc=$? :: $(grep -a -E 'local-ci: (PASS|FAIL|ALL|SKIP [0-9]+)|FAILED' $L/local-ci.log | head -n 6 | tr '\n' '|' | cut -c1-500)"
G 4h "release battery" bash tools/release-battery.sh --evidence-dir $L/battery-${H:0:12} > $L/battery.log 2>&1
note "battery rc=$? $(grep -m1 -oE 'RELEASE BATTERY (PASS|FAIL)' $L/battery.log) $(grep -m1 -oE 'kernel-check +[A-Z]+ +[^(]*\([^)]*\)' $L/battery.log)"
G 1h "serve-smoke" bash tools/serve-smoke.sh > $L/serve-smoke.log 2>&1
note "serve-smoke rc=$? $(grep -c '  ok: ' $L/serve-smoke.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke.log) FAIL"
note "done"
