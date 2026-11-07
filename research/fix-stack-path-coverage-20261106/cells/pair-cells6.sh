#!/bin/bash
# DSv4 pair window 2026-11-07 on the pair box GPUs 0,1, on binaries from the fix-stack tip 99c65ab0b2
# (bin6/, sha256 in bin6/bin6.sha256). Untimed pass/fail cells, in priority order:
#   A. hardware proof owed since 2026-11-05: the AR-epoch fix (67c578d28b; the four SIB-7 dense
#      gates and the full-token replay), the ep-pair gate with the drafter it needs, and the
#      finish_reason fix (1be8f19ab8; a 48-token cut must read "length").
#   B. the 2026-11-05 passes again on the restacked tip (split C4 digest 8e376825c9, replicated
#      planes a89c57deac included).
#   C. two single-card Max-Q serving gates on GPU 1: the DSpark restore doors' default-ON boot
#      line (#1069) with the DSpark serve smoke, and spec-cache-mixed (coverage row D21).
# Fence: starts at 07:30 at the earliest, on idle cards 0,1, after stage6 verified the model; no new
# cell after 10:30, every cell over by 10:50; the model and drafter are deleted at the end, and
# pair-guard6.sh kills and cleans at 11:00 regardless.
set -u
W=<pair-kit>
B=$W/bin6; K=$W/tools6; M=$W/dsv4-nvfp4; T=$W/tape.txt; O=$W/out6
Q38=<maxq-kit>/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DF=$W/dflash2-q38
WF=<box-home>/.local/bin/waitfor
export PATH=<box-home>/.local/bin:$PATH
mkdir -p $O; S=$O/status.txt
echo "cells6 armed $(date '+%m-%d %H:%M:%S')" >> $S
START=$(date -d '2026-11-07 07:30' +%s); DEADLINE=$(date -d '2026-11-07 10:30' +%s); END=$(date -d '2026-11-07 10:50' +%s)
$WF --file $W/stage6.ok --timeout 16h --every 1m > $O/wait.log 2>&1 || { echo "cells6: model never staged ($(date '+%m-%d %H:%M'))" >> $S; exit 3; }
$WF --until "test \$(date +%s) -ge $START" --timeout 16h --every 1m >> $O/wait.log 2>&1
[ "$(date +%s)" -lt "$DEADLINE" ] || { echo "cells6: too late to start ($(date '+%m-%d %H:%M'))" >> $S; exit 3; }
$WF --until 'test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1)" -lt 2000' \
    --timeout 3h --every 30s >> $O/wait.log 2>&1 || { echo "cards 0,1 busy at $(date +%H:%M)" >> $S; exit 3; }
[ "$(date +%s)" -lt "$DEADLINE" ] || { echo "cells6: cards freed too late ($(date +%H:%M))" >> $S; exit 3; }
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "memra-gpu.lock busy" >> $S; exit 3; }
( cd $B && sha256sum -c --quiet bin6.sha256 ) > $O/bin-check.log 2>&1 || { echo "cells6: bin6 fails its manifest" >> $S; exit 3; }
export CUDA_VISIBLE_DEVICES=0,1 MEMRA_DSV4_DECODE_PATH=device
TPEP="MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
SIB="MEMRA_DSV4_DOTS_ARM=f32x MEMRA_DSV4_DRAFTER=off MEMRA_MOE_F16G=2 MEMRA_F16G_SK=32 MEMRA_DSV4_SAMPLER=device"
echo "start $(date +%H:%M:%S) bins verified; model $(du -sh $M | cut -f1)" >> $S
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,fan.speed,power.draw,clocks.sm,memory.used --format=csv -i 0,1 -lms 1000 > $O/thermal.csv 2>&1 &
TPID=$!

cell() {  # cell NAME TIMEOUT_S ENV... -- CMD...
    local name=$1 to=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then echo "$name SKIP deadline" >> $S; return; fi
    [ $((END - now)) -lt "$to" ] && to=$((END - now))
    env "${envs[@]}" timeout "$to" "$@" > $O/$name.log 2>&1
    local rc=$?
    local v=$(grep -a -E -m4 'PASS|FAIL|REFUSE|_OK|DIGEST|panicked|assertion|out of memory|retired|required|requires' $O/$name.log | cut -c1-160 | tr '\n' '|')
    echo "$name rc=$rc $(( $(date +%s) - now ))s placing=$(grep -c '\[load\] placing' $O/$name.log) $(grep -a -m1 -o '\[load\] placing [a-z_]*' $O/$name.log) :: $v" >> $S
}

serve() {  # serve NAME ENV...: two greedy 48-token requests; both answered, identical, finish "length"
    local name=$1; shift
    local now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then echo "$name SKIP deadline" >> $S; return; fi
    local port=18300
    env "$@" MEMRA_COMPAT=openai MEMRA_MODELS="dsv4=$M" MEMRA_ADDR=127.0.0.1:$port $B/memra-server > $O/$name.log 2>&1 &
    local pid=$! ok
    # waitfor --http does not notice a server that exited: poll the pid alongside
    if $WF --until "curl -sf -o /dev/null http://127.0.0.1:$port/v1/models || ! kill -0 $pid" --timeout 15m --every 5s > $O/$name.wait 2>&1 \
        && kill -0 $pid 2>/dev/null; then
        for i in 1 2; do
            curl -s -m 300 http://127.0.0.1:$port/v1/chat/completions -H 'content-type: application/json' \
                -d '{"model":"dsv4","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"Name three prime numbers and say why each is prime."}]}' \
                > $O/$name.req$i.json 2>&1
        done
        ok=$(python3 - $O/$name.req1.json $O/$name.req2.json <<'PY' 2>&1 | tail -n 1
import json, sys
r = [json.load(open(f)) for f in sys.argv[1:]]
t = [(x["choices"][0]["message"].get("reasoning") or "") + "\x1f" + (x["choices"][0]["message"].get("content") or "") for x in r]
fin = [x["choices"][0].get("finish_reason") for x in r]
n = [x.get("usage", {}).get("completion_tokens") for x in r]
print("answered" if all(s.strip("\x1f") for s in t) else "EMPTY", "identical" if len(set(t)) == 1 else "DIFFER",
      "finish=" + ",".join(map(str, fin)), "tokens=" + ",".join(map(str, n)),
      "PASS" if all(s.strip("\x1f") for s in t) and len(set(t)) == 1 and fin == ["length", "length"] else "FAIL")
PY
)
    else
        ok="no-ready $(tail -n 2 $O/$name.log | cut -c1-200 | tr '\n' '|')"
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "$name $(( $(date +%s) - now ))s $(grep -a -m1 -o '\[load\] placing [a-z_]*' $O/$name.log) :: $ok" >> $S
}

# A. owed proof
cell SIB-7-dense-fast 1200 $TPEP $SIB -- $B/dsv4_dense_fast_gate $M $T $O/dense-fast
cell SIB-7-dense-exact-tail 1200 $TPEP $SIB -- $B/dsv4_dense_exact_tail_gate $M $T $O/dense-exact-tail
cell SIB-7-compose-cadence 1200 $TPEP $SIB -- $B/dsv4_compose_cadence_dense_gate $M $T $O/compose-cadence --defaults
cell SIB-7-hcdot-defaults 1200 $TPEP $SIB -- $B/dsv4_hc_dot_split_gate $M $T $O/hcdot --defaults
cell FIX-6-full-replay 1200 $TPEP MEMRA_DSV4_SAMPLER=device -- $B/dsv4_tp_ep_sampled_perf_gate $M $T --placement=tp_ep --full-token-replay
cell PP6-7-ep-pair 1200 MEMRA_DSV4_DRAFTER=dspark -- $B/dsv4_ep_pair_gate $M $T
serve SERVE6-1-tpep
# B. the 2026-11-05 passes on the restacked tip
cell TP_EP-3-rows 1500 -- $B/dsv4_rows_gate $M $T 24 64 --placement=tp_ep
cell TP_EP-4-verify 1200 -- $B/dsv4_tp_ep_verify_gate $M $T --placement=tp_ep
cell TP_EP-5-tpep 1200 $TPEP -- $B/dsv4_tp_ep_gate $M $T --placement=tp_ep
cell TP_EP-6-sampled-plain 1200 $TPEP -- $B/dsv4_tp_ep_sampled_perf_gate $M $T --placement=tp_ep
cell REPL-3-tpep 1200 $TPEP -- $B/dsv4_tp_ep_gate $M $T --placement=tp_ep_replicated
cell REPL-2-verify 1200 -- $B/dsv4_tp_ep_verify_gate $M $T --placement=tp_ep_replicated
cell REPL-4-sampled 1200 $TPEP -- $B/dsv4_tp_ep_sampled_perf_gate $M $T --placement=tp_ep_replicated
cell PULL-REPL-3-tpep 1200 MEMRA_DSV4_AR_PUSH=0 $TPEP -- $B/dsv4_tp_ep_gate $M $T --placement=tp_ep_replicated
cell TP_EP-7-kvsplit 1200 $TPEP -- $B/dsv4_kv_split_gate $M $T 3000 300
cell TP_EP-7-replaylong 1200 $TPEP -- $B/dsv4_tp_replay_long_gate $M $T 304
cell TP_EP-8-park 1200 MEMRA_DSV4_DRAFTER=dspark -- $B/dsv4_park_boundary_gate $M $T
serve SERVE6-1b-tpep-dspark MEMRA_DSV4_DRAFTER=dspark
cell PP-2-rows 1500 -- $B/dsv4_rows_gate $M $T 24 64 --placement=pp
cell PP-3-verify 1200 -- $B/dsv4_tp_ep_verify_gate $M $T --placement=pp
cell PP-7-pipeline 1200 -- $B/dsv4_pipeline_gate $M $T
serve SERVE6-2-pp MEMRA_DSV4_TOPOLOGY=pp
cell PP-4-c4-host 1200 MEMRA_DSV4_DRAFTER=dspark -- $B/dsv4_c4_host_gate $M $T
cell PP-4-c4-direct 1200 MEMRA_DSV4_DRAFTER=dspark -- $B/dsv4_c4_direct_gate $M $T capacity-1m
cell PP-7-graph-probe 1200 MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_EP=off MEMRA_DSV4_GRAPH_PROBE=1 -- $B/dsv4_graph_probe_gate $M $T graph-probe
cell PP-6-matrix-distribution 1200 MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_PREFILL_MOE=reference MEMRA_DSV4_EP=off -- $B/dsv4_matrix_distribution_gate $M $T $O/matrix-dist
cell PP-4-plain-perf 1200 MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_SAMPLE_SORT=radix MEMRA_DSV4_PREFILL_MOE=reference -- $B/dsv4_plain_perf_gate $M $T all
# C. single-card Max-Q serving gates on GPU 1 (the lock on FD 9 is reused by both gates)
if [ -f "$DF/READY" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; then
    CUDA_VISIBLE_DEVICES=1 MEMRA_GPU_LOCK=/tmp/memra-gpu.lock timeout 1800 bash $K/dspark-serve-smoke.sh $Q38 $DF $B/memra-server $O/dspark-smoke > $O/dspark-smoke.log 2>&1
    rc=$?
    boot=$(grep -a -h -m1 -o '\[dspark\] prefix restore doors default [^"]*"[^"]*" ([^)]*): restore=[a-z]* partial=[a-z]* capture=[a-z]*' $O/dspark-smoke/serve-on.log 2>/dev/null)
    case "$boot" in *"default ON"*"restore=true partial=true capture=true"*) b=PASS ;; *) b=FAIL ;; esac
    echo "DSPARK6-default-on rc=$rc boot-line $b :: $boot :: $(grep -a -E 'ALL GREEN|FAIL' $O/dspark-smoke.log | tail -n 2 | tr '\n' '|' | cut -c1-200)" >> $S
else
    echo "DSPARK6-default-on SKIP (no drafter at $DF or past the deadline)" >> $S
fi
if [ "$(date +%s)" -lt "$DEADLINE" ]; then
    CUDA_VISIBLE_DEVICES=1 MEMRA_GPU_LOCK=/tmp/memra-gpu.lock timeout 2400 bash $K/spec-cache-mixed-gate.sh $Q38 $B/memra-server $O/spec-cache-mixed > $O/spec-cache-mixed.log 2>&1
    echo "SPEC-CACHE-MIXED6 rc=$? :: $(grep -a -E 'PASS|FAIL|GATE' $O/spec-cache-mixed.log | tail -n 3 | tr '\n' '|' | cut -c1-260)" >> $S
else
    echo "SPEC-CACHE-MIXED6 SKIP deadline" >> $S
fi

kill $TPID 2>/dev/null
rm -rf <pair-kit>/dsv4-nvfp4 <pair-kit>/dflash2-q38 <pair-kit>/stage6.ok
echo "cells6 done $(date +%H:%M:%S); model and drafter deleted; $(df -BG /mnt/Data | tail -1 | awk '{print $4}') free; gpu0,1 MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | tr '\n' ' ')" >> $S
