#!/usr/bin/env bash
# GB10 serve smoke of memra#1292 (per-row attribution of the deepseek-v4.1 route's call flag): memra-server on
# a 2-layer trunk (MEMRA_DSV41_TRUNK_LAYERS: a rehearsal of the serving loop, not the checkpoint's answers),
# greedy, ignore_eos, so every request runs exactly max_tokens steps and the step numbers below are exact.
#   boot 1, MEMRA_DSV41_CALL_FAULT=24:1,64: A (steps 0..31) and B (admitted at step 1 or later) share batched
#     calls; step 24 poisons running request 1 (B, row 1 of the batched call): B fails alone, A completes. Then A
#     alone (steps 32..63) is A's solo run: A's two answers must be the same bytes. Step 64 is C's prefill with
#     the whole-step injection (a fault no row map names): C fails, D is served after it.
#   boot 2, MEMRA_DSV41_CALL_FAULT=0:0,25:0: W's prefill (step 0, its own call) poisoned: W fails alone. A (step 1
#     on) and B share batched calls; step 25 poisons running request 0 (A, row 0): A fails alone, B completes. Then
#     B alone is B's solo run: B's two answers must be the same bytes.
# The poisoned decode steps are 24 and 25 (smoke.sh: 8 and 9), so B has some 20 steps, not 8, to reach the server
# before the door's step: with 8, a run where B was admitted at step 11 poisoned nobody (`step 8 has no running
# request 1`) and its checks failed on the harness, not the route. The summary also checks that each poisoned
# decode step was a two-row batched call.
# usage: smoke3.sh TAG   Done-file out/smoke3-TAG/s.txt, marker SMOKE-DONE.
set -u
TAG=$1
R=$HOME/.cache/dsv41-fu1113
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$R/bins/$TAG
O=$R/out/smoke3-$TAG; mkdir -p $O; : > $O/s.txt
ADDR=127.0.0.1:18393
KEY=dsv41-row-fault-smoke
sha256sum $B/memra-server >> $O/s.txt
# GB10 memory is the host's: the checkpoint's cached pages count as used, so drop them before each boot
# (fadvise, no root)
drop_cache() {
  python3 - "$C" <<'PY'
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/s.txt 2>&1

SRV=
boot() {  # boot N FAULTS
  local n=$1 faults=$2
  drop_cache
  free -g | head -2 | tail -1 >> $O/s.txt
  MEMRA_MODELS=v41=$C MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_CTX=8192 \
    MEMRA_DSV41_TRUNK_LAYERS=2 MEMRA_DSV41_CALL_FAULT=$faults $B/memra-server > $O/server-$n.log 2>&1 &
  SRV=$!
  local ready=0 t0=$SECONDS
  for _ in $(seq 1 450); do
    if ! kill -0 $SRV 2>/dev/null; then echo "boot $n: server exited before readiness" >> $O/s.txt; break; fi
    if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
    sleep 2
  done
  echo "boot $n ($faults): ready=$ready after $((SECONDS - t0))s" >> $O/s.txt
  [ $ready = 1 ]
}
stop() {  # stop N
  curl -s -H "Authorization: Bearer $KEY" http://$ADDR/health > $O/health-$1.json
  kill -TERM $SRV 2>/dev/null
  for _ in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
  kill -KILL $SRV 2>/dev/null
  wait $SRV 2>/dev/null
}
ask() {  # ask NAME PROMPT MAX: one greedy request, ignore_eos, its answer in resp-NAME.json
  local name=$1 prompt=$2 n=$3
  local body="{\"model\":\"v41\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$n,\"temperature\":0,\"ignore_eos\":true}"
  local code
  code=$(curl -s -o $O/resp-$name.json -w '%{http_code}' -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' -d "$body" --max-time 600 http://$ADDR/v1/chat/completions)
  echo "request $name: http $code $(head -c 300 $O/resp-$name.json | tr '\n' ' ')" >> $O/s.txt
}
steps_at_least() {  # steps_at_least LOG N: wait (at most 60 s) for N step lines
  for _ in $(seq 1 300); do
    [ "$(grep -ac '^\[dsv41-step\]' $1)" -ge $2 ] && return 0
    sleep 0.2
  done
  echo "timed out waiting for $2 steps in $1" >> $O/s.txt
  return 1
}
PA="What is the capital of France? Answer in one word."
PB="Name three primary colors."

if boot 1 "24:1,64"; then
  ask A1 "$PA" 32 & a=$!
  steps_at_least $O/server-1.log 1
  ask B1 "$PB" 32 & b=$!
  wait $a $b
  ask A1solo "$PA" 32
  ask C1 "Say hello." 8
  ask D1 "Say goodbye." 8
fi
stop 1
if boot 2 "0:0,25:0"; then
  ask W2 "Count to three." 8
  ask A2 "$PA" 32 & a=$!
  steps_at_least $O/server-2.log 2
  ask B2 "$PB" 32 & b=$!
  wait $a $b
  ask B2solo "$PB" 32
fi
stop 2

python3 - "$O" >> $O/s.txt <<'EOF'
import json, os, sys
o = sys.argv[1]
def resp(n):
    p = f"{o}/resp-{n}.json"
    try:
        return json.load(open(p))
    except Exception as e:
        return {"unreadable": str(e)}
def answer(r):
    try:
        c = r["choices"][0]
        m = c["message"]
        return (m.get("content"), m.get("reasoning"), c.get("finish_reason"), r["usage"]["completion_tokens"])
    except Exception:
        return None
def routes(x):
    if isinstance(x, dict):
        if "routes" in x and isinstance(x["routes"], list):
            return x["routes"]
        for v in x.values():
            r = routes(v)
            if r is not None:
                return r
    return None
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name, detail)
a1, a1s, b2, b2s = answer(resp("A1")), answer(resp("A1solo")), answer(resp("B2")), answer(resp("B2solo"))
check("boot1: A beside the poisoned B completes 32 tokens", a1 is not None and a1[3] == 32, repr(a1)[:200])
check("boot1: A's answer beside the poisoned B is its solo answer, byte for byte", a1 is not None and a1 == a1s)
check("boot1: the poisoned B fails", answer(resp("B1")) is None, json.dumps(resp("B1"))[:200])
check("boot1: C (whole-step injection) fails", answer(resp("C1")) is None)
check("boot1: D after it is served", answer(resp("D1")) is not None)
check("boot2: W (prefill poisoned) fails", answer(resp("W2")) is None)
check("boot2: the poisoned A fails", answer(resp("A2")) is None)
check("boot2: B beside the poisoned A completes 32 tokens", b2 is not None and b2[3] == 32, repr(b2)[:200])
check("boot2: B's answer beside the poisoned A is its solo answer, byte for byte", b2 is not None and b2 == b2s)
check("A and B answer differently (the comparison can fail)", a1 is not None and b2 is not None and a1[:2] != b2[:2])
for n, want in (("1", 2), ("2", 2)):
    try:
        h = json.load(open(f"{o}/health-{n}.json"))
    except Exception as e:
        h = {}
    for r in routes(h) or []:
        print("boot", n, "route", r.get("name"), "phase", r.get("phase"), "requests", r.get("requests"),
              "request_faults", r.get("request_faults"), "dead_reason", r.get("dead_reason"))
        check(f"boot{n}: the route is alive with {want} request faults",
              r.get("dead_reason") is None and r.get("request_faults") == want)
for n in ("1", "2"):
    log = open(f"{o}/server-{n}.log", errors="replace").read().splitlines()
    alone = [l for l in log if "failing it alone" in l]
    whole = [l for l in log if "failing" in l and "request(s)" in l]
    print(f"boot{n}: {len(alone)} request(s) failed alone, {len(whole)} whole-step failure(s)")
    for l in log:
        if "MEMRA_DSV41_CALL_FAULT" in l or "failing it alone" in l or ("failing" in l and "request(s)" in l) or "probe" in l:
            print("  ", l[:300])
check("boot1: one request failed alone, one whole step failed",
      sum("failing it alone" in l for l in open(f"{o}/server-1.log", errors="replace")) == 1
      and sum("failing 1 request(s)" in l for l in open(f"{o}/server-1.log", errors="replace")) == 1)
check("boot2: two requests failed alone, no whole step failed",
      sum("failing it alone" in l for l in open(f"{o}/server-2.log", errors="replace")) == 2
      and not any("request(s)" in l and "failing" in l for l in open(f"{o}/server-2.log", errors="replace")))
for n, step in (("1", 24), ("2", 25)):
    log = open(f"{o}/server-{n}.log", errors="replace").read().splitlines()
    at = [i for i, l in enumerate(log) if f"MEMRA_DSV41_CALL_FAULT: step {step} poisons" in l]
    nxt = next((l for l in log[at[0] + 1:] if l.startswith("[dsv41-step]")), "") if at else ""
    check(f"boot{n}: the poisoned step {step} is one batched call of two rows",
          " calls=1 " in nxt and " prefill=0 " in nxt and " decode_rows=2 " in nxt, nxt[:60])
print("SMOKE", "PASS" if ok else "FAIL")
EOF
echo "SMOKE-DONE $(date +%T)" >> $O/s.txt
