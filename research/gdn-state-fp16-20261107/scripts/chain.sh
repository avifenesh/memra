#!/usr/bin/env bash
# Rig-side driver for the GB10 cells of the fp16 GDN state lane (research/gdn-state-fp16-20261107).
# The caller holds the unit's sparklock lease and passes its id. Every served arm boots fresh.
#   chain.sh UNIT LEASE BINTAG MODE OUTDIR
# MODE:
#   build    rsync the lane source to the unit and build sm_121a there (spark_build.sh, WHAT=...)
#   census   the GDN state magnitude census (fp32 class, MEMRA_GDN_STATE_FP16=0): the 264-task
#            panel on one boot, then a 131K and a
#            258K cold prime with a 1,024-token turn each (turnbench) on a second boot
#   quality  the 264-task long-context panel (qual_panel.py) on class QCLASS (0|1), tag rep QREP
#   ttft     TTFT 32K/128K/256K, A B B A (fp32, fp16, fp16, fp32)
#   bench    decode-batch-bench decode tick, A B B A A B per cell (no server)
#   gates    kernel-check, fp32 identity vs main, run-spec K=1..8, GPU tests (gates.sh)
set -u
UNIT=${1:?unit b23d|cb33}; LEASE=${2:?lease}; BINTAG=${3:?bin tag}; MODE=${4:?mode}; OUT=${5:?outdir}
NODE=spark-$UNIT
R=.cache/memra-gdnfp16
L=$HOME/$R
H=$HOME/.cache/model-scout/harness
PORT=18294; [ "$UNIT" = b23d ] && PORT=18295
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
ssh $NODE "mkdir -p $R/runs $R/logs $R/src"
scp -q "$L"/scripts/*.sh "$NODE:$R/"
boot() { # tag class [census dir on the unit]
  ssh $NODE "(BIN=\$HOME/$R/bins/$BINTAG LEASE=$LEASE CLASS=$2 TAG=$1 CENSUS=${3:-} nohup bash \$HOME/$R/serve.sh) > \$HOME/$R/runs/guard-$1.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
  waitfor --ssh $NODE 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 25m --every 15s --quiet || { log "boot FAILED $1"; return 1; }
  if ss -ltn | grep -q ":$PORT "; then log "rig port $PORT busy; refusing"; return 1; fi
  ssh -o ExitOnForwardFailure=yes -N -L $PORT:127.0.0.1:8090 $NODE &
  TUN=$!
  waitfor --http http://127.0.0.1:$PORT/v1/models --timeout 60s --quiet || { log "tunnel FAILED $1"; return 1; }
  log "booted $1 class=$2 census=${3:-off} $(ssh $NODE "grep -m1 -E 'gdn-census\\] on|GDN decode class' \$HOME/$R/runs/serve-$1.log" | cut -c1-120)"
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
preflight() { # synthetic kernel-check before any served arm
  ssh $NODE "SPARKLOCK_LEASE=$LEASE MEMRA_KC_FAST=1 timeout 600 \$HOME/$R/bins/$BINTAG/kernel-check" > "$OUT/kc-fast.log" 2>&1
  log "preflight kernel-check rc=$? $(grep -cE ' OK$' "$OUT/kc-fast.log") OK, $(grep -cE ' FAIL$' "$OUT/kc-fast.log") FAIL; $(grep -E '^gdn_st16|^gdn_sth' "$OUT/kc-fast.log" | grep -c ' OK$') gdn 16-bit OK"
  if grep -qE ' FAIL$' "$OUT/kc-fast.log"; then log "preflight FAILED: stopping"; exit 3; fi
}
case "$MODE" in
  build)
    head=$(git -C "$L/wt-arm" rev-parse HEAD)
    rsync -a --delete --exclude=.git --exclude=research --exclude=target "$L/wt-arm/" "$NODE:$R/src/"
    (cd "$L/wt-arm" && rsync -a -R research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja \
      research/spill-b-20260919/fixtures/recompute-load.csv "$NODE:$R/src/")
    ssh $NODE "echo $head > \$HOME/$R/src/.lane-head"
    W=${WHAT:-bins}
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE WHAT=$W JOBS=${JOBS:-6} nohup bash \$HOME/$R/spark_build.sh $BINTAG) > \$HOME/$R/runs/build-$BINTAG-$W.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q SPARK_BUILD_DONE \$HOME/$R/logs/build-$BINTAG-$W.log" --timeout 150m --every 60s --quiet || log "build wait failed"
    scp -q "$NODE:$R/logs/build-$BINTAG-$W.log" "$OUT/" 2>/dev/null
    scp -q "$NODE:$R/bins/$BINTAG/sha256.txt" "$OUT/bins-$BINTAG-121a.sha256" 2>/dev/null
    log "build $BINTAG ($W) at $head: $(tail -n 1 "$OUT/build-$BINTAG-$W.log")" ;;
  census)
    tag=census-panel
    boot $tag 0 "\$HOME/$R/runs/census/panel" && {
      timeout 12000 /usr/bin/python3 "$H/qual_panel.py" "$OUT/$tag" "$tag" > "$OUT/$tag.log" 2>&1; log "panel rc=$? $(tail -n 3 "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-300)"
      stop $tag
    }
    tag=census-long
    boot $tag 0 "\$HOME/$R/runs/census/long" && {
      (cd "$H" && timeout 5400 python3 turnbench.py "$OUT/$tag" --plan 131072:1 258048:1 --new 1024 --reps 1) > "$OUT/$tag.log" 2>&1
      log "turnbench rc=$? $(grep '^{' "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-400)"
      stop $tag
    }
    rsync -a "$NODE:$R/runs/census/" "$OUT/census/" 2>/dev/null
    log "census rows: panel $(wc -l < "$OUT/census/panel/census.jsonl" 2>/dev/null) long $(wc -l < "$OUT/census/long/census.jsonl" 2>/dev/null)" ;;
  quality)
    preflight
    tag=quality-fp16${QCLASS:?QCLASS}-${QREP:-1}
    boot $tag "$QCLASS" && {
      timeout 12000 /usr/bin/python3 "$H/qual_panel.py" "$OUT/$tag" "$tag" > "$OUT/$tag.log" 2>&1; log "quality rc=$? $(tail -n 3 "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-300)"
      stop $tag
    } ;;
  ttft)
    i=0
    for c in 0 1 1 0; do
      i=$((i + 1)); tag=ttft-r$i-fp16$c
      boot $tag $c || continue
      ssh -o BatchMode=yes "$NODE" 'while :; do awk -v t=$(date +%s) "/MemAvailable/{printf \"{\\\"unix\\\": %d, \\\"mem_available_gib\\\": %.3f}\\n\", t, \$2/1048576}" /proc/meminfo || exit; sleep 1; done' > "$OUT/$tag-memory.jsonl" &
      MW=$!
      (cd "$H" && timeout 5400 python3 turnbench.py "$OUT/$tag" --plan 32768:1 131072:1 258048:1 --new 1024 --reps 1) > "$OUT/$tag.log" 2>&1
      log "ttft $tag rc=$? $(grep '^{' "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-400)"
      kill $MW 2>/dev/null
      stop $tag
    done ;;
  bench)
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE nohup bash \$HOME/$R/bench.sh $BINTAG) > \$HOME/$R/runs/bench.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q BENCH_DONE \$HOME/$R/runs/bench.out" --timeout 180m --every 60s --quiet || log "bench wait failed"
    scp -q "$NODE:$R/runs/bench/*" "$OUT/" 2>/dev/null
    log "bench done" ;;
  buildmain)
    # main's tree for the fp32 identity baseline: an archive of origin/main's build inputs, its
    # own source and target dirs on the unit (never the lane target)
    head=$(git -C "$L/wt-arm" merge-base HEAD origin/main)
    ssh $NODE "rm -rf \$HOME/.cache/memra-gdnfp16/main-src && mkdir -p \$HOME/.cache/memra-gdnfp16/main-src"
    git -C "$L/wt-arm" archive "$head" Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml crates \
      research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja \
      research/spill-b-20260919/fixtures/recompute-load.csv | ssh $NODE "tar -x -C \$HOME/$R/main-src"
    ssh $NODE "echo $head > \$HOME/$R/main-src/.lane-head"
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE WHAT=bins JOBS=${JOBS:-6} SRC=\$HOME/$R/main-src TGT=\$HOME/$R/main-target nohup bash \$HOME/$R/spark_build.sh $BINTAG) > \$HOME/$R/runs/build-$BINTAG-bins.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q SPARK_BUILD_DONE \$HOME/$R/logs/build-$BINTAG-bins.log" --timeout 150m --every 60s --quiet || log "build wait failed"
    scp -q "$NODE:$R/logs/build-$BINTAG-bins.log" "$OUT/" 2>/dev/null
    scp -q "$NODE:$R/bins/$BINTAG/sha256.txt" "$OUT/bins-$BINTAG-121a.sha256" 2>/dev/null
    log "build main $BINTAG at $head: $(tail -n 1 "$OUT/build-$BINTAG-bins.log")" ;;
  census-long)
    tag=census-long
    boot $tag 0 "\$HOME/$R/runs/census/long" && {
      (cd "$H" && timeout 5400 python3 turnbench.py "$OUT/$tag" --plan 131072:1 258048:1 --new 1024 --reps 1) > "$OUT/$tag.log" 2>&1
      log "turnbench rc=$? $(grep '^{' "$OUT/$tag.log" | tr '\n' ' ' | cut -c1-400)"
      stop $tag
    }
    rsync -a "$NODE:$R/runs/census/long/" "$OUT/census/long/" 2>/dev/null
    log "census rows: long $(wc -l < "$OUT/census/long/census.jsonl" 2>/dev/null)" ;;
  gates)
    ssh $NODE "(SPARKLOCK_LEASE=$LEASE nohup bash \$HOME/$R/gates.sh $BINTAG ${MAINBIN:?MAINBIN}) > \$HOME/$R/runs/gates-$BINTAG.out 2>&1 < /dev/null & echo started" >> "$OUT/chain.log"
    waitfor --ssh $NODE "grep -q GATES_DONE \$HOME/$R/runs/gates-$BINTAG/gates.log" --timeout 180m --every 60s --quiet || log "gates wait failed"
    rsync -a "$NODE:$R/runs/gates-$BINTAG/" "$OUT/gates/" 2>/dev/null
    log "gates done: $(tr '\n' ' ' < "$OUT/gates/gates.log" | cut -c1-900)" ;;
esac
log "CHAIN_DONE $MODE"
