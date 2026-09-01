#!/bin/bash
# CELL: guard-fires squeeze on the GRAPH-SESSION route, co-resident squeezer shape
# (lane/graph-launch-guard-sweep-20260831).
#
# ROUTE STATEMENT: decode.rs GraphSession::step (site decode.rs:75). Two findings the
# earlier g-runs banked: (1) GraphSession serving is OPT-IN (MEMRA_SERVE_GS=1, default
# OFF: the load-stable default keeps every width on the batched body) and DEGRADES to
# batched-eager the moment a second session admits, so no single-server storm can ever
# squeeze a live graph session; (2) a foreign process's cudaMalloc is refused far above
# the floor, so external ballast cannot cross either. The shape that CAN fire in the
# fleet is the box10 two-stack shape: a CO-RESIDENT server's internal allocation storm
# exhausts the SHARED driver while this server's solo graph session steps. This cell
# boots server A (MEMRA_SERVE_GS=1, plain, solo greedy = graph session) and server B
# (the MTP-vg storm shape) on the same card; B's birth-race walks driver-free through
# the floor; A's next step() refuses RECOVERABLY:
# `[graph-session] graph replay suspended:` + session-scoped error; A lives.
# Usage: cell-squeeze-gsession2.sh <run-tag>
set -u
. /home/ubuntu/guard-lane/gl-lib.sh
TAG=${1:?run tag}
LOGA=serve-gsA-$TAG.log
LOGB=serve-gsB-$TAG.log
PORTB=18913
resp_ok() { python3 -c "import json,sys;d=json.load(open('$1'));t=d.get('text') or (d.get('choices') or [{}])[0].get('text');print('OK:'+str(len(t)) if t else 'FAIL:'+str(d)[:120])" 2>/dev/null || echo PARSE_FAIL; }
reqB() { # same as req but against server B
  local body out
  body=$(mktemp /tmp/gl-reqB.XXXXXX.json); out=$5
  python3 - "$1" "$2" "$3" "$4" <<'EOF' > "$body"
import json, sys
idx, mt, temp, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
if idx.startswith("T"):
    p = open("/home/ubuntu/guard-lane/prompt-30k.txt").read()[: int(idx[1:])]
else:
    prompts = json.load(open("/home/ubuntu/vram-gates/assets/agentic8.json"))
    n = int(idx[1:]) if idx.startswith("u") else int(idx)
    digits = []
    while True:
        digits.append(n % 8); n //= 8
        if n == 0: break
    p = "\n\n".join(str(prompts[i % len(prompts)]) for i in digits)
body = {"model": "q38", "prompt": p, "max_tokens": mt}
if temp < 0:
    body["temperature"] = 0; body["presence_penalty"] = 0.1
else:
    body["temperature"] = temp
if mt > 5000: body["stream"] = True
print(json.dumps(body))
EOF
  curl -s -m 900 "http://127.0.0.1:$PORTB/v1/completions" -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' -d @"$body" > "$out" 2>&1
  rm -f "$body"
}

say "=== GRAPH-SESSION CO-RESIDENT SQUEEZE RUN $TAG (bin=lane) ==="
gpu_empty || { say "REFUSING: GPU not empty / server alive"; exit 2; }
dmesg_mark
sampler_start $G/free-gs2-$TAG.csv

# server A: the graph-session stack (solo greedy interactive rides GraphSession)
SERVE_ENV="$SERVE_ENV_COMMON"
boot "$BINLANE" "MEMRA_SERVE_SPEC=0 MEMRA_SERVE_GS=1" "$LOGA" || { sampler_stop; exit 1; }
SRV_A=$SRV_PID

# server B: the MTP-vg storm stack, co-resident on the same card
SERVE_ENV="$SERVE_ENV_MTP MEMRA_ADDR=127.0.0.1:$PORTB"
say "boot B bin=$BINLANE (storm stack, port $PORTB)"
env CUDA_VISIBLE_DEVICES=0 $SERVE_ENV MEMRA_ADMIT_RESERVE_MB=16 nohup "$BINLANE" > $G/$LOGB 2>&1 &
SRV_B=$!
for i in $(seq 1 360); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORTB/health 2>/dev/null)
  [ "$CODE" = "200" ] && { say "B HEALTH=200 pid=$SRV_B"; break; }
  kill -0 $SRV_B 2>/dev/null || { say "B DIED during boot"; tail -5 $G/$LOGB | tee -a $OUT; break; }
  sleep 5
done

# ---- phase 1: false-positive check on BOTH; A runs a solo graph-session request ----
req 0 128 0 "" $G/resp-$TAG-p1a.json
SA=$(grep -ac "graph replay suspended:" $G/$LOGA || true)
SB=$(grep -ac "graph replay suspended:" $G/$LOGB || true)
[ "${SA:-0}${SB:-0}" != "00" ] && { say "FALSE-POSITIVE at healthy headroom (A=$SA B=$SB)"; kill -9 $SRV_B 2>/dev/null; shutdown_srv; sampler_stop; exit 9; }
say "phase1 clean: A/B suspended=0/0 free=$(gpu0_free_mb)MB"

# ---- phase 2: A's RESTORED-HIT solo graph-session generations + B's storm ----
# GraphSession promotion admits RESTORED-HIT / POOL-RESUME sessions ONLY, never a
# cold chunked-prefill one (worker.rs REACHABILITY note, h1 lesson) - so each loop
# iteration first SEEDS the prefix cache with a short cold turn on a long real
# prompt, then re-sends the SAME prompt long: the exact hit restores prefill-done,
# the solo session starts its tick with generated.is_empty(), and the worker
# promotes it to GraphSession (budget 4900 >= gs_min 384).
rm -f $G/.stop-$TAG
( n=0
  while [ ! -f $G/.stop-$TAG ]; do
    n=$((n+1))
    # h3 lesson: the truncated-30k doc EOSes after a token or two, so the promoted
    # session never actually STEPS; the agentic prompts generate long, so seed the
    # cache cold then re-send the SAME prompt long: exact hit -> restore -> promote
    # -> a genuinely long graph-stepping generation
    req "u$((n % 8))" 32 0 "" $G/resp-$TAG-seedA$n.json
    req "u$((n % 8))" 4900 0 "" $G/resp-$TAG-loopA$n.json
    echo $n > $G/.loops-$TAG
    sleep 1
  done ) &
LOOP_PID=$!
sleep 6

# pre-squeeze ballast sized from live free (leave ~13GB), then B's storm + chase
FREE=$(gpu0_free_mb)
HOLD=$((FREE - 13000)); [ $HOLD -lt 256 ] && HOLD=256
nohup $BALLAST 0 $HOLD > $G/ballast-pre-$TAG.log 2>&1 &
BPRE_PID=$!
sleep 6
# B's own observer loop: B's MTP spec rounds sample the floor every ~25ms and its
# [spec] line is the crossing's ground truth (h2 lesson: without a generation on B,
# nothing on B observes and the storm's crossing has no receipt)
rm -f $G/.stopB-$TAG
( n=0
  while [ ! -f $G/.stopB-$TAG ]; do
    n=$((n+1))
    reqB "u$n" 4000 0 "" $G/resp-$TAG-loopB$n.json
    sleep 1
  done ) &
LOOPB_PID=$!
say "pre-ballast holding: free=$(gpu0_free_mb)MB; B storm starts"
( for sidx in 1 2 3 4 5 6; do
    reqB "T$((5000 + sidx * 137))" 125000 -1 "" $G/resp-$TAG-stackB$sidx.json &
  done; wait ) &
PRESS_PID=$!
( sleep 35
  for sidx in 7 8 9 10 11 12; do
    reqB "T$((5000 + sidx * 137))" 125000 -1 "" $G/resp-$TAG-stackB$sidx.json &
  done; wait ) &
PRESS2_PID=$!
sleep 15
nohup $BALLAST 0 1024 1024 2 60000 > $G/ballast-$TAG.log 2>&1 &
BPID=$!

FIRED_AT=""
BFIRED=""
for i in $(seq 1 210); do
  S=$(grep -ac "\[graph-session\] graph replay suspended:" $G/$LOGA || true)
  SB2=$(grep -ac "graph replay suspended:" $G/$LOGB || true)
  [ -z "$BFIRED" ] && [ "${SB2:-0}" != "0" ] && { BFIRED="t=$((i*2))s"; say "B crossing ground-truth observed ($BFIRED)"; }
  if [ "${S:-0}" != "0" ]; then
    FIRED_AT="t=$((i*2))s"
    say "A graph-session suspension observed ($FIRED_AT)"
    break
  fi
  # once B has receipted the crossing, give A another 60s then stop waiting
  if [ -n "$BFIRED" ] && [ $i -gt 30 ]; then
    BSEC=${BFIRED#t=}; BSEC=${BSEC%s}
    [ $((i*2 - BSEC)) -gt 60 ] && break
  fi
  sleep 2
done

sleep 10
touch $G/.stop-$TAG $G/.stopB-$TAG
ALIVE_A=no; kill -0 $SRV_A 2>/dev/null && ALIVE_A=yes
ALIVE_B=no; kill -0 $SRV_B 2>/dev/null && ALIVE_B=yes
REFUSALS=$(grep -ac "graph-session replay refused" $G/$LOGA || true)
STEPFAIL=$(grep -ac "graph session step FAILED" $G/$LOGA || true)
say "A alive: $ALIVE_A ; B alive: $ALIVE_B ; A refusals: $REFUSALS ; A step-FAILED lines: $STEPFAIL"

# ---- phase 3: release, recovery on A ----
kill -TERM $BPID ${BPRE_PID:-0} 2>/dev/null
kill -TERM $SRV_B 2>/dev/null
wait $LOOP_PID 2>/dev/null; wait ${LOOPB_PID:-0} 2>/dev/null; wait $PRESS_PID 2>/dev/null; wait ${PRESS2_PID:-0} 2>/dev/null
sleep 8
req 7 96 0 "" $G/resp-$TAG-recovery.json
RECOV=$(resp_ok $G/resp-$TAG-recovery.json)

SUS=$(grep -ac "\[graph-session\] graph replay suspended:" $G/$LOGA || true)
SUSB=$(grep -ac "graph replay suspended:" $G/$LOGB || true)
kill -9 $SRV_B 2>/dev/null
shutdown_srv
sampler_stop
dmesg_check gs2-$TAG
FAULTS=$(grep -c . $G/dmesg-gs2-$TAG.txt 2>/dev/null | head -1); FAULTS=${FAULTS:-0}
say "VERDICT $TAG: A_gsession_suspended=$SUS A_refusals=$REFUSALS A_stepfail=$STEPFAIL B_suspended=$SUSB fired_at=${FIRED_AT:-NEVER} recovery=$RECOV aliveA=$ALIVE_A dmesg_faults=$FAULTS"
rm -f $G/.stop-$TAG $G/.stopB-$TAG $G/.loops-$TAG
if [ "${SUS:-0}" != "0" ] && [ "$FAULTS" = "0" ] && [ "$ALIVE_A" = "yes" ] && [ "${RECOV%%:*}" = "OK" ]; then
  say "RUN $TAG: PASS"; exit 0
fi
say "RUN $TAG: FAIL"; exit 1
