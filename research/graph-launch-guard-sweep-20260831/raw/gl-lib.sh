#!/bin/bash
# graph-launch-guard-sweep battery helpers (lane/graph-launch-guard-sweep-20260831).
# Modeled on the step37 admission lane's vg-lib.sh; adapted for the q38 DSPARK stack
# (Qwen3.8-27B-NVFP4-Q5K-mtp + DFlash2 drafter: the box12 serving shape).
# Source this; then boot/shutdown/receipts per cell. GPU 0 only; refuses when any
# memra-server is alive or the card is not empty.
set -u
G=/home/ubuntu/guard-lane
BINLANE=/home/ubuntu/guard-bins/memra-server-lane-d6701001e
BINBASE=/home/ubuntu/guard-bins/memra-server-base-b78b439bc
MODEL=/data/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DRAFT=/data/models/q38-dflash2
RANKS=/data/models/q38-gguf/q38-ranks-sxc32768.gguf.txt
PROMPTS=/home/ubuntu/vram-gates/assets/agentic8.json   # 8 real agentic prompts (never synthetic)
BALLAST=/home/ubuntu/vram-gates/ballast                # step37 lane tool, source banked in raw/ballast.cu
PORT=18912
KEY=cellkey
KEYHASH=$(printf %s "$KEY" | sha256sum | cut -d' ' -f1)
OUT=$G/battery.txt
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUT; }

SERVE_ENV_COMMON="MEMRA_MODELS=q38=$MODEL MEMRA_SERVE_SPEC=1 MEMRA_CTX=131072 MEMRA_API_KEYS=cell:$KEYHASH MEMRA_ADDR=127.0.0.1:$PORT"

# The q38 dspark serve shape (box12 launcher parity: MEMRA_DSPARK_SPEC=1 + DFlash2
# draft dir + FRSPEC trim ranks, MTP draft NEVER set). RUN 1-20 FINDING (confirmed in
# code, dflash.rs serve burst: `deferred` is set ONLY on the markov/plain-chain
# branch, never on the DFlash2 branch): a DFlash2 drafter NEVER passes the vgraphs
# ctx to the verify, so the verify-graph pool ENGAGES but takes ZERO captures on this
# shape and spec.rs run_full/run_segment are unreachable from it. Used for identity
# cells and route receipts, not for the vg squeeze.
SERVE_ENV_DSPARK="$SERVE_ENV_COMMON MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_FRSPEC_TRIM=$RANKS MEMRA_DFLASH_ADAPT=0 MEMRA_DFLASH_VERIFY_T=4"

# The MTP-route verify-graph shape (the ornith-class serve program): the q38 artifact
# CARRIES its MTP block, and MEMRA_SPEC_VERIFY_GRAPH=1 opts the GDN+DENSE family into
# the vg door, so decode_step_t_core_vg replays the SAME DsparkVerifyGraphs pool
# (spec.rs run_full/run_segment) through the SAME guarded qwen35_verify_tparallel the
# dflash markov-drafter serve arm uses. This is the on-box-reachable caller of the
# guarded sites.
SERVE_ENV_MTP="$SERVE_ENV_COMMON MEMRA_SPEC_VERIFY_GRAPH=1 MEMRA_SPEC_K=3"

# default for existing cells
SERVE_ENV="$SERVE_ENV_DSPARK"

gpu0_free_mb() { nvidia-smi -i 0 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }

# Anchored patterns everywhere (run-6 lesson: an unanchored `pkill -f` matched the
# ssh wrapper's own command line and killed the cleanup shell before it cleaned).
SRV_PAT='^/home/ubuntu/guard-bins/memra-server'
BAL_PAT='^/home/ubuntu/vram-gates/ballast'

gpu_empty() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  [ "$used" -lt 200 ] && ! pgrep -f "$SRV_PAT" >/dev/null
}

dmesg_mark() { date -u '+%Y-%m-%d %H:%M:%S' > $G/.dmesg-mark; }
dmesg_check() { # $1=tag ; prints fault lines since mark
  sudo journalctl -k --since "$(cat $G/.dmesg-mark)" 2>/dev/null | grep -aiE 'segfault|Xid|general protection|traps' | tee $G/dmesg-$1.txt
  local n
  n=$(grep -c . $G/dmesg-$1.txt 2>/dev/null | head -1); n=${n:-0}
  say "dmesg[$1]: $n fault line(s)"
}

boot() { # $1=bin $2=extra-env $3=logname -> sets SRV_PID. Route env comes from
  # $SERVE_ENV (default dspark); a cell overrides SERVE_ENV=$SERVE_ENV_MTP first.
  if pgrep -f "$SRV_PAT" >/dev/null; then say "PRE-BOOT: a battery server is alive, refusing"; return 8; fi
  say "boot bin=$1 md5=$(md5sum $1 | cut -d' ' -f1) extra='$2' log=$3"
  ulimit -c unlimited
  env CUDA_VISIBLE_DEVICES=0 $SERVE_ENV $2 nohup "$1" > $G/$3 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 360); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$CODE" = "200" ] && { say "HEALTH=200 pid=$SRV_PID"; return 0; }
    kill -0 $SRV_PID 2>/dev/null || { say "SERVER_DIED during boot"; tail -20 $G/$3 | tee -a $OUT; return 2; }
    sleep 5
  done
  say "BOOT_TIMEOUT"; kill -TERM $SRV_PID; return 3
}

shutdown_srv() {
  kill -TERM ${SRV_PID:-0} 2>/dev/null
  for i in $(seq 1 60); do kill -0 ${SRV_PID:-0} 2>/dev/null || break; sleep 2; done
  kill -0 ${SRV_PID:-0} 2>/dev/null && kill -KILL $SRV_PID
  pkill -9 -f "$BAL_PAT" 2>/dev/null
  sleep 3
  pgrep -f "$SRV_PAT" >/dev/null && { say "SHUTDOWN: stray battery server"; pkill -9 -f "$SRV_PAT"; sleep 3; }
}

sampler_start() { # $1=csv
  ( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',')"; sleep 0.2; done ) > "$1" 2>/dev/null &
  SAMPLER_PID=$!
}
sampler_stop() { kill ${SAMPLER_PID:-0} 2>/dev/null; wait ${SAMPLER_PID:-0} 2>/dev/null; }

# One completion request from the real prompt pool. $1=prompt-index(0..7) $2=max_tokens
# $3=temp ("0" greedy) $4=seed ("" none) $5=outfile
# Per-request tmp body (run-1 lesson: a shared /tmp body file raced under concurrency
# and four "different" requests all carried the same prompt).
req() {
  local body
  body=$(mktemp /tmp/gl-req.XXXXXX.json)
  python3 - "$1" "$2" "$3" "$4" <<'EOF' > "$body"
import json, sys
idx, mt, temp, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
prompts = json.load(open("/home/ubuntu/vram-gates/assets/agentic8.json"))
def pick(i):
    p = prompts[int(i) % len(prompts)]
    return p if isinstance(p, str) else json.dumps(p)
# "2+5" concatenates real prompts (unique prefix-cache keys without synthetic text);
# "u<N>" expands N to ALL its base-8 digits as a combo (length = number of digits):
# unique keys, and the digit count controls the prompt length ramp
if idx == "F":
    # the real ~30k-token curve prompt (extracted from /root/curve-30k-1.json): the
    # long-context internal squeezer whose chunked prime ratchets driver-free down
    # from an already-admitted session
    p = open("/home/ubuntu/guard-lane/prompt-30k.txt").read()
elif idx.startswith("T"):
    # T<chars>: the 30k curve prompt truncated at <chars> characters. Unique real
    # mid-document prefixes that a model CONTINUES (run-13 lesson: concatenated
    # whole prompts EOS after 1 token, so "live" stacked sessions exited in 0.3s)
    p = open("/home/ubuntu/guard-lane/prompt-30k.txt").read()[: int(idx[1:])]
elif idx.startswith("u"):
    n = int(idx[1:])
    digits = []
    while True:
        digits.append(n % 8)
        n //= 8
        if n == 0:
            break
    p = "\n\n".join(pick(i) for i in digits)
else:
    p = "\n\n".join(pick(i) for i in idx.split("+"))
body = {"model": "q38", "prompt": p, "max_tokens": mt}
if temp > 0:
    body["temperature"] = temp
    body["top_p"] = 0.9
elif temp < 0:
    # PENALIZED GREEDY: the one shape dspark refuses by design and the worker serves
    # on the PLAIN path regardless of concurrency (run-16 lesson: sampled solo
    # requests still ride the dspark arm, whose prime OOMs at the wall)
    body["temperature"] = 0
    body["presence_penalty"] = 0.1
else:
    body["temperature"] = 0
if seed:
    body["seed"] = int(seed)
if mt > 5000:
    # the non-streaming deadline gate refuses ~5400+ token budgets (run-13 receipt);
    # a big-budget request must stream. Used by the squeeze's stack rows, whose
    # CAPACITY-SIZED cache birth (ctx_cap KV, allocated layer by layer) is what walks
    # driver-free through the floor from inside the server (runs 15-17: no request
    # small enough to fit at the external wall allocates anything).
    body["stream"] = True
print(json.dumps(body))
EOF
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d @"$body" > "$5" 2>&1
  rm -f "$body"
}

# FALSE-POSITIVE CHECK (fleet-peer requirement): any suspended line at HEALTHY
# headroom is a bug; the caller kills the run on non-zero.
suspended_count() { grep -ac "graph replay suspended:" "$G/$1" || true; }

receipts() { # $1=logname $2=tag
  {
    echo "--- receipts $2 ($1)"
    grep -a '\[admit-cal\] boot calibration' $G/$1 | head -2
    echo "route_dspark_cal: $(grep -ac 'boot calibration done.*route=dspark' $G/$1)"
    echo "dspark_route_armed: $(grep -ac 'DSPARK SPEC route armed' $G/$1)"
    echo "vg_pool_engaged: $(grep -ac 'serve pool ENGAGED' $G/$1)"
    echo "vg_debt_lines: $(grep -ac 'verify-graph pool debt' $G/$1)"
    echo "suspended_total: $(grep -ac 'graph replay suspended:' $G/$1)"
    grep -a 'graph replay suspended:' $G/$1 | head -3
    echo "oom_lines: $(grep -ac 'CUDA_ERROR_OUT_OF_MEMORY' $G/$1)"
    echo "illegal=$(grep -aic 'ILLEGAL' $G/$1) sentinel87=$(grep -ac '#87' $G/$1) panic=$(grep -aic 'panic' $G/$1)"
  } | tee -a $OUT
}
