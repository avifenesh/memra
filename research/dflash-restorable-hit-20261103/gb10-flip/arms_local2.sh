#!/usr/bin/env bash
# Runs ON a Spark unit (detached), bench client on the unit: GB10 served arms for memra#1031 and the
# restore-default flip. Same serve script and bench steps as arms 1 and 2 (bench_memra.sh), with the
# MemAvailable sampler local instead of over ssh.
#   LEASE=<sparklock lease> UNIT=spark-xxxx bash arms_local2.sh OUTDIR TAG:BIN:USERS:DOORS ...
# BIN is a directory name under ~/memra-run; USERS is comma-separated; DOORS is DEF (the three restore
# doors unset, device default) or OFF (all four doors 0); both set MEMRA_PREFIX_STABLE_BOUNDARY=0.
set -u
LEASE=${LEASE:?lease}
UNIT=${UNIT:?unit}
O=$1; shift
mkdir -p "$O"
H=/home/avifenesh/.cache/model-scout/harness
SERVE=${SERVE:-$HOME/memra-run/serve-batchspec.sh}   # cb33: the b23d copy (no MEMRA_VISION_DIR), to match b23d
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) [$UNIT] $*" | tee -a "$O/chain.log"; }
OFF="MEMRA_DSPARK_PREFIX_RESTORE=0 MEMRA_DSPARK_PARTIAL_RESTORE=0 MEMRA_DSPARK_BOUNDARY_CAPTURE=0 MEMRA_PREFIX_STABLE_BOUNDARY=0"
DEF="MEMRA_DSPARK_PREFIX_RESTORE= MEMRA_DSPARK_PARTIAL_RESTORE= MEMRA_DSPARK_BOUNDARY_CAPTURE= MEMRA_PREFIX_STABLE_BOUNDARY=0"
bench() { # OUT TAG USERS
  local out=$1 tag=$2 users=$3
  mkdir -p "$out"
  blog(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" >> "$out/run.log"; }
  ( while :; do awk -v t=$(date +%s) '/MemAvailable/{printf "{\"unix\": %d, \"mem_available_gib\": %.3f}\n", t, $2/1048576}' /proc/meminfo; sleep 1; done ) > "$out/$UNIT-memory.jsonl" &
  local mw=$!
  blog "start $tag base=$BENCH_BASE model=$BENCH_MODEL node=$UNIT client=local"
  (cd $H && timeout 900 python3 turnbench.py "$out/warm" --plan 8192:1 32768:1 --new 256 --reps 1 >> "$out/turn.log" 2>&1) || blog "warm rc=$?"
  blog "warm $(tail -n 2 "$out/turn.log" | cut -c1-90 | tr '\n' ' ')"
  for u in $users; do
    (cd $H && timeout 3600 python3 ab_agent.py run /home/avifenesh/.cache/claude-spark-20261025/ab-workload.json "$out" "$tag" --users $u --think 1 --temperature 1.0 --seed 1234 >> "$out/cells.log" 2>&1) || blog "cell c$u rc=$?"
    blog "cell c$u $(tail -n 1 "$out/cells.log" | cut -c1-160)"
  done
  kill $mw 2>/dev/null
  python3 - "$out/$UNIT-memory.jsonl" >> "$out/run.log" <<'EOF'
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
  (ARM=gguf-dflash BIN=$bin LEASE=$LEASE EXTRA="$extra" nohup bash $SERVE) > $HOME/memra-run/serve-$tag.out 2>&1 < /dev/null &
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
  echo "unit=$UNIT client=local" > "$O/$tag/UNIT"
  log "$tag vetoes=$(grep -a -c 'cache-preferred' "$O/$tag/server.log") cold-drafter=$(grep -a -c 'cold-drafter restore' "$O/$tag/server.log") $(grep -a 'prefix restore doors default' "$O/$tag/server.log" | head -1)"
}
log "arms_local2 start lease=$LEASE serve=$SERVE: $*"
for spec in "$@"; do
  IFS=: read -r tag bin users doors <<< "$spec"
  case "$doors" in DEF) extra=$DEF ;; OFF) extra=$OFF ;; *) log "bad doors '$doors'"; exit 2 ;; esac
  serve "$tag" "$HOME/memra-run/$bin" "${users//,/ }" "$extra"
done
log ARMS_LOCAL2_DONE
