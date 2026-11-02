#!/usr/bin/env bash
# batch-spec GB10 receipts on spark-cb33 (owner GGUF Qwen3.8-27B, gguf-mtpmask arm), agentic cell
# (ab_agent.py via bench_memra.sh: 48 real coding-agent turns, thinking on, T=1, 768 new tokens)
# at USERS, one server boot per (arm, rep), arms interleaved. Runs on the rig; the server runs on
# cb33 under the memra cb33 lease. Usage: gb10_receipts.sh OUTDIR "ARMS" [REPS]
#   arms: model = batch-spec (pooled + cost-model gate); band = batch-spec with
#   MEMRA_SPEC_GATE_MODEL=0; base = the spark-sm121 binary the batch-spec branch sits on.
set -u
OUT=$1; ARMS=$2; REPS=${3:-1}
NODE=spark-cb33
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)
USERS=${USERS:-1 2 4 8 16 32}
BS_BIN=${BS_BIN:?batch-spec bin dir on cb33}
BASE_BIN=${BASE_BIN:?base bin dir on cb33}
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/receipts.log"; }
for rep in $(seq 1 "$REPS"); do
  for arm in $ARMS; do
    case $arm in
      model) BIN=$BS_BIN; EXTRA="" ;;
      band) BIN=$BS_BIN; EXTRA="MEMRA_SPEC_GATE_MODEL=0" ;;
      base) BIN=$BASE_BIN; EXTRA="" ;;
    esac
    tag="memra-$arm-r$rep"
    log "boot $tag bin=$BIN"
    ssh $NODE "(ARM=gguf-mtpmask BIN=$BIN LEASE=$LEASE EXTRA='$EXTRA' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-batchspec-$tag.out 2>&1 < /dev/null & echo started"
    waitfor --ssh $NODE 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED $tag"; ssh $NODE 'kill $(cat ~/memra-run/serve-bs-gguf-mtpmask.pid)'; continue; }
    ssh -o ExitOnForwardFailure=yes -N -L 8090:127.0.0.1:8090 $NODE &
    TUN=$!
    waitfor --http http://127.0.0.1:8090/v1/models --timeout 60s --quiet
    BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 NODE=$NODE QUALITY=0 USERS="$USERS" \
      BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
      bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/$tag" "$tag"
    log "done $tag: $(grep -h 'cell c' "$OUT/$tag/run.log" | cut -c1-200 | tr '\n' ' ')"
    kill $TUN 2>/dev/null
    PID=$(ssh $NODE 'cat ~/memra-run/serve-bs-gguf-mtpmask.pid')
    ssh $NODE "kill $PID"
    waitfor --ssh $NODE "test ! -e /proc/$PID" --timeout 3m --quiet
    scp -q "$NODE:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/$tag/server.log" 2>/dev/null
  done
done
log "all done"
