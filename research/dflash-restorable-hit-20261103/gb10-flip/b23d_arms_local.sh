#!/usr/bin/env bash
# Runs ON spark-b23d (detached), client on the unit: the laptop's Wi-Fi went to 150-350 ms RTT to the
# Sparks and dropped the 12:56 arm-3 tunnel twice (warm tokenize timeout, then the tunnel died 14 min
# into c1), so the bench client runs here and the laptop only monitors.
# Arms, fresh boot each, same serve script and bench steps as arms 1 and 2 (bench_memra.sh with the
# MemAvailable sampler local instead of over ssh):
#   3  dflash-flip-default      flip 6561499516, three restore doors unset (GB10 default ON), STABLE_BOUNDARY=0, c1..c32
#   4  dflash-1031r-off         #1031 ea5ef13e8b, all four doors off, c1..c32
#   4r dflash-1031r-off-c1r     same as 4, c1 only (reverse-order repeat)
#   3r dflash-flip-default-c1r  same as 3, c1 only
#   LEASE=<sparklock lease> bash ~/memra-run/b23d_arms_local.sh   -> ~/memra-run/arms-local/
set -u
LEASE=${LEASE:?lease}
O=$HOME/memra-run/arms-local
mkdir -p "$O"
H=$HOME/.cache/model-scout/harness
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$O/chain.log"; }
OFF="MEMRA_DSPARK_PREFIX_RESTORE=0 MEMRA_DSPARK_PARTIAL_RESTORE=0 MEMRA_DSPARK_BOUNDARY_CAPTURE=0 MEMRA_PREFIX_STABLE_BOUNDARY=0"
DEF="MEMRA_DSPARK_PREFIX_RESTORE= MEMRA_DSPARK_PARTIAL_RESTORE= MEMRA_DSPARK_BOUNDARY_CAPTURE= MEMRA_PREFIX_STABLE_BOUNDARY=0"
bench() { # OUT TAG USERS
  local out=$1 tag=$2 users=$3
  mkdir -p "$out"
  blog(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" >> "$out/run.log"; }
  ( while :; do awk -v t=$(date +%s) '/MemAvailable/{printf "{\"unix\": %d, \"mem_available_gib\": %.3f}\n", t, $2/1048576}' /proc/meminfo; sleep 1; done ) > "$out/spark-b23d-memory.jsonl" &
  local mw=$!
  blog "start $tag base=$BENCH_BASE model=$BENCH_MODEL node=spark-b23d client=local"
  (cd $H && timeout 900 python3 turnbench.py "$out/warm" --plan 8192:1 32768:1 --new 256 --reps 1 >> "$out/turn.log" 2>&1) || blog "warm rc=$?"
  blog "warm $(tail -n 2 "$out/turn.log" | cut -c1-90 | tr '\n' ' ')"
  for u in $users; do
    (cd $H && timeout 3600 python3 ab_agent.py run $HOME/.cache/claude-spark-20261025/ab-workload.json "$out" "$tag" --users $u --think 1 --temperature 1.0 --seed 1234 >> "$out/cells.log" 2>&1) || blog "cell c$u rc=$?"
    blog "cell c$u $(tail -n 1 "$out/cells.log" | cut -c1-160)"
  done
  kill $mw 2>/dev/null
  python3 - "$out/spark-b23d-memory.jsonl" >> "$out/run.log" <<'EOF'
import json, sys
v = [json.loads(l)['mem_available_gib'] for l in open(sys.argv[1]) if l.strip()]
print(f'MemAvailable min {min(v):.2f} GiB, median {sorted(v)[len(v)//2]:.2f} GiB over {len(v)} s')
EOF
  blog end
}
serve() { # tag bindir users extra
  local tag=$1 bin=$2 users=$3 extra=$4
  if ss -ltn | grep -q ':8090 '; then log "port 8090 busy before $tag; refusing"; exit 4; fi
  [ -d "$O/$tag" ] && mv "$O/$tag" "$O/$tag-prev-$(date +%H%M%S)"
  (ARM=gguf-dflash BIN=$bin LEASE=$LEASE EXTRA="$extra" nohup bash $HOME/memra-run/serve-batchspec.sh) > $HOME/memra-run/serve-$tag.out 2>&1 < /dev/null &
  local ok=0
  for i in $(seq 1 80); do curl -sf http://127.0.0.1:8090/v1/models >/dev/null && { ok=1; break; }; sleep 15; done
  [ $ok = 1 ] || { log "boot FAILED $tag"; exit 3; }
  log "start $tag"
  BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 \
    CLIENT_MAX_TIME=1800 bench "$O/$tag" "$tag" "$users"
  log "done $tag"
  local pid
  pid=$(cat $HOME/memra-run/serve-bs-gguf-dflash.pid)
  kill "$pid"
  for i in $(seq 1 90); do [ -e /proc/$pid ] || break; sleep 2; done
  cp $HOME/memra-run/serve-bs-gguf-dflash.log "$O/$tag/server.log"
  sleep 3
  cp $HOME/memra-run/guard-bs-gguf-dflash.log "$O/$tag/guard.log"
  echo "unit=spark-b23d client=local" > "$O/$tag/UNIT"
  log "$tag vetoes=$(grep -a -c 'cache-preferred' "$O/$tag/server.log") cold-drafter=$(grep -a -c 'cold-drafter restore' "$O/$tag/server.log") $(grep -a 'prefix restore doors default' "$O/$tag/server.log" | head -1)"
}
log "local chain start lease=$LEASE"
serve dflash-flip-default $HOME/memra-run/bin-flip-6561499516 "1 2 4 8 16 32" "$DEF"
serve dflash-1031r-off $HOME/memra-run/bin-1031r-ea5ef13e8b "1 2 4 8 16 32" "$OFF"
serve dflash-1031r-off-c1r $HOME/memra-run/bin-1031r-ea5ef13e8b "1" "$OFF"
serve dflash-flip-default-c1r $HOME/memra-run/bin-flip-6561499516 "1" "$DEF"
log LOCAL_CHAIN_DONE
