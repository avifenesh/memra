#!/usr/bin/env bash
# Rig-side driver for the GB10 cells of the bf16 GDN decode-class lane
# (research/gdn-state-bf16-decode-20261107). The caller holds the unit's sparklock lease and passes
# its id. Every served arm boots fresh, one arm per boot.
#   gb10_chain.sh UNIT LEASE BINTAG MODE OUTDIR
# MODE:
#   build    rsync the lane source to the unit, build sm_121a there (spark_build.sh)
#   gates    kernel-check, fp32 identity vs main, run-spec K=1..8, GPU tests (gb10_gates.sh)
#   quality  the 264-task long-context panel (qual_panel.py) on class QCLASS (0|1), tag rep QREP
#   probes   coherence (greedy + sampled) fp32 then bf16, then TTFT 32K/128K/256K with the
#            MemAvailable floor, A B B A
#   bench    decode-batch-bench decode tick, A B B A A B per cell (no server)
#   cells    ABAB served agentic cells c1..c32 (bench_memra.sh, QUALITY=0): fp32, bf16, fp32,
#            bf16; nothing else is sent to a server before its cells
set -u
UNIT=${1:?unit b23d|cb33}; LEASE=${2:?lease}; BINTAG=${3:?bin tag}; MODE=${4:?mode}; OUT=${5:?outdir}
NODE=spark-$UNIT
R=.cache/memra-gdnbf16-decode
L=$HOME/$R
H=$HOME/.cache/model-scout/harness
PORT=18292; [ "$UNIT" = b23d ] && PORT=18293
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
ssh $NODE "mkdir -p $R/runs $R/logs $R/src"
scp -q "$L/scripts/serve_gdn.sh" "$L/scripts/spark_build.sh" "$L/scripts/gb10_gates.sh" "$L/scripts/gb10_bench.sh" "$NODE:$R/"
boot() { # tag class
  ssh $NODE "(BIN=\$HOME/$R/bins/$BINTAG LEASE=$LEASE CLASS=$2 TAG=$1 nohup bash \$HOME/$R/serve_gdn.sh) > \$HOME/$R/runs/guard-$1.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
  waitfor --ssh $NODE 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 25m --every 15s --quiet || { log "boot FAILED $1"; return 1; }
  if ss -ltn | grep -q ":$PORT "; then log "rig port $PORT busy; refusing"; return 1; fi
  ssh -o ExitOnForwardFailure=yes -N -L $PORT:127.0.0.1:8090 $NODE &
  TUN=$!
  waitfor --http http://127.0.0.1:$PORT/v1/models --timeout 60s --quiet || { log "tunnel FAILED $1"; return 1; }
  log "booted $1 class=$2 $(ssh $NODE "grep -m1 'bf16 GDN decode class' \$HOME/$R/runs/serve-$1.log" | cut -c1-90)"
}
stop() { # tag
  kill $TUN 2>/dev/null
  PID=$(ssh $NODE "cat \$HOME/$R/runs/serve-$1.pid")
  ssh $NODE "kill $PID"
  waitfor --ssh $NODE "test ! -e /proc/$PID" --timeout 3m --quiet
  scp -q "$NODE:$R/runs/serve-$1.log" "$OUT/server-$1.log" 2>/dev/null
  scp -q "$NODE:$R/runs/serve-$1.guard" "$OUT/guard-$1.txt" 2>/dev/null
  log "stopped $1 $(grep -m1 '\[prefix-cache\] on:' "$OUT/server-$1.log" | cut -c1-160)"
}
export BENCH_BASE=http://127.0.0.1:$PORT BENCH_MODEL=q38 NODE BENCH_TOKENIZE_PATH=/v1/tokenize \
  BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800
preflight() { # synthetic kernel-check (incl. the gdn_st16_* cells) before any served arm
  ssh $NODE "SPARKLOCK_LEASE=$LEASE MEMRA_KC_FAST=1 timeout 600 \$HOME/$R/bins/$BINTAG/kernel-check" > "$OUT/kc-fast.log" 2>&1
  log "preflight kernel-check rc=$? $(grep -cE ' OK$' "$OUT/kc-fast.log") OK, $(grep -cE ' FAIL$' "$OUT/kc-fast.log") FAIL; $(grep -E '^gdn_st16' "$OUT/kc-fast.log" | grep -c ' OK$') gdn_st16 OK"
  if grep -qE ' FAIL$' "$OUT/kc-fast.log"; then log "preflight FAILED: stopping"; exit 3; fi
}
case "$MODE" in
  build)
    head=$(git -C "$L/wt" rev-parse HEAD)
    rsync -a --delete --exclude=.git --exclude=research --exclude=target "$L/wt/" "$NODE:$R/src/"
    (cd "$L/wt" && rsync -a --files-from="$L/notes/research-includes.txt" ./ "$NODE:$R/src/")
    ssh $NODE "echo $head > \$HOME/$R/src/.lane-head"
    W=${WHAT:-all}
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE WHAT=$W JOBS=${JOBS:-6} nohup bash \$HOME/$R/spark_build.sh $BINTAG) > \$HOME/$R/runs/build-$BINTAG-$W.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q SPARK_BUILD_DONE \$HOME/$R/logs/build-$BINTAG-$W.log" --timeout 150m --every 60s --quiet || log "build wait failed"
    scp -q "$NODE:$R/logs/build-$BINTAG-$W.log" "$OUT/" 2>/dev/null
    scp -q "$NODE:$R/bins/$BINTAG/sha256.txt" "$OUT/bins-$BINTAG-121a.sha256" 2>/dev/null
    log "build $BINTAG ($W) at $head: $(tail -n 1 "$OUT/build-$BINTAG-$W.log")" ;;
  gates)
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE nohup bash \$HOME/$R/gb10_gates.sh $BINTAG ${MAINBIN:?MAINBIN}) > \$HOME/$R/runs/gates-$BINTAG.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q GATES_DONE \$HOME/$R/runs/gates-$BINTAG/gates.log" --timeout 180m --every 60s --quiet || log "gates wait failed"
    rsync -a "$NODE:$R/runs/gates-$BINTAG/" "$OUT/gates/" 2>/dev/null
    log "gates done: $(tr '\n' ' ' < "$OUT/gates/gates.log" | cut -c1-900)" ;;
  quality)
    preflight
    tag=quality-bf16${QCLASS:?QCLASS}-${QREP:-1}
    boot $tag "$QCLASS" && {
      timeout 12000 /usr/bin/python3 "$H/qual_panel.py" "$OUT/$tag" "$tag" > "$OUT/$tag.log" 2>&1; log "quality rc=$? $(tail -n 3 "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-300)"
      stop $tag
    } ;;
  probes)
    for c in 0 1; do
      tag=coherence-bf16$c
      boot $tag $c || continue
      python3 "$L/scripts/coherence.py" http://127.0.0.1:$PORT q38 "$OUT/$tag.jsonl" >> "$OUT/chain.log" 2>&1
      stop $tag
    done
    i=0
    for c in 0 1 1 0; do
      i=$((i + 1)); tag=ttft-r$i-bf16$c
      boot $tag $c || continue
      ssh -o BatchMode=yes "$NODE" 'while :; do awk -v t=$(date +%s) "/MemAvailable/{printf \"{\\\"unix\\\": %d, \\\"mem_available_gib\\\": %.3f}\\n\", t, \$2/1048576}" /proc/meminfo || exit; sleep 1; done' > "$OUT/$tag-memory.jsonl" &
      MW=$!
      (cd "$H" && timeout 5400 python3 turnbench.py "$OUT/$tag" --plan 32768:1 131072:1 258048:1 --new 1024 --reps 1) > "$OUT/$tag.log" 2>&1
      log "ttft $tag rc=$? $(grep '^{' "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-400)"
      kill $MW 2>/dev/null
      stop $tag
    done ;;
  bench)
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE nohup bash \$HOME/$R/gb10_bench.sh $BINTAG) > \$HOME/$R/runs/bench.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q BENCH_DONE \$HOME/$R/runs/bench.out" --timeout 180m --every 60s --quiet || log "bench wait failed"
    scp -q "$NODE:$R/runs/bench/*" "$OUT/" 2>/dev/null
    log "bench done" ;;
  cells)
    preflight
    i=0
    for c in 0 1 0 1; do
      i=$((i + 1)); tag=cells-r$i-bf16$c
      boot $tag $c || continue
      QUALITY=0 USERS="1 2 4 8 16 32" bash "$HOME/.cache/memra-spark/b23d/bench_memra.sh" "$OUT/$tag" "$tag"
      stop $tag
    done ;;
esac
log "CHAIN_DONE $MODE"
