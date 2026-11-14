#!/usr/bin/env bash
# smoke_wrx2.sh (2026-11-14, after the first 40-layer run): the first script assumed the 2-layer rehearsal's timing and
# token counts. At 40 layers a 32-token request ends in under a second (and 'Paris' stops at 19 tokens even with
# ignore_eos), so the second request never shared a call with the first. Here both requests are long (N tokens of an
# essay), the second is sent once the first has run 20 steps, and the poisoned step (FS) is far enough in that both are
# decoding. The whole-step injection is left to the GB10 smoke.
# (Header corrected after the run, 2026-11-14: it still described the first script's steps and counts. Only
# these comment lines changed; the commands below are the ones that ran.)
# Serve smoke of memra#1292 (per-row attribution of the deepseek-v4.1 route's call flag) on one RTX PRO 6000:
# memra-server on the checkpoint's whole trunk (40 layers), on the residency the card picks by itself, the
# one-card residency (the routed experts in pinned host memory behind the device cache: the off-card MoE that
# reads its plan back, the path #1292 changes). Greedy; two essay prompts of N tokens (SMOKE_N, default 300), which
# end on the budget. The requests carry ignore_eos, which the chat surface drops (memra#1325): nothing here
# relies on it.
#   boot 1, MEMRA_DSV41_CALL_FAULT=FS:1 (SMOKE_FS, default 150): A starts, B is sent once A has run 20 steps, and
#     they share batched calls; step FS poisons running request 1 (B, row 1 of the batched call): B fails alone, A
#     completes its N tokens. Then A alone is A's solo run: A's two answers must be the same bytes.
#   boot 2, MEMRA_DSV41_CALL_FAULT=0:0,FS:0: W's prefill (step 0, its own call) poisoned: W fails alone. A and B
#     share batched calls as in boot 1; step FS poisons running request 0 (A, row 0): A fails alone, B completes.
#     Then B alone is B's solo run: B's two answers must be the same bytes.
# No whole-step injection here; the GB10 smoke has it.
# The summary also checks that each poisoned decode step was one batched call of two rows, and the residency:
# at least one host-experts line in the boot log and an expert cache on every step.
# Run it inside a one-card lease, from any directory; it starts and stops its own server (127.0.0.1:18393, or
# SMOKE_ADDR). Two boots of the whole checkpoint and about 1,200 steps: about two minutes with the checkpoint in
# the page cache (08:53 to 08:55 in the window).
# PASS: s.txt ends with "SMOKE PASS" and "SMOKE-DONE", with 16 PASS lines and no FAIL line.
# usage: smoke_wrx2.sh TAG [OUT_DIR]   binaries from /home/access/dsv41/perf-bins/TAG (memra-server);
#        receipts in OUT_DIR (default /home/access/dsv41/rowfault-1315/smoke-TAG): s.txt, server-{1,2}.log,
#        resp-*.json, health-{1,2}.json
set -u
TAG=$1
D=/home/access/dsv41
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$D/perf-bins/$TAG
O=${2:-$D/rowfault-1315/smoke-$TAG}; mkdir -p $O; : > $O/s.txt
ADDR=${SMOKE_ADDR:-127.0.0.1:18393}
[ -x $B/memra-server ] || { echo "no $B/memra-server" | tee -a $O/s.txt; echo "SMOKE-DONE $(date +%T)" >> $O/s.txt; exit 2; }
KEY=dsv41-row-fault-smoke
sha256sum $B/memra-server >> $O/s.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/s.txt 2>&1

SRV=
boot() {  # boot N FAULTS
  local n=$1 faults=$2
  free -g | head -2 | tail -1 >> $O/s.txt
  MEMRA_MODELS=v41=$C MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM \
    MEMRA_DSV41_CALL_FAULT=$faults $B/memra-server > $O/server-$n.log 2>&1 &
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
N=${SMOKE_N:-300}; FS=${SMOKE_FS:-150}
PA="Write a detailed 800-word essay about the history of the bicycle."
PB="Write a detailed 800-word essay about how bread is baked, step by step."

if boot 1 "$FS:1"; then
  ask A1 "$PA" $N & a=$!
  steps_at_least $O/server-1.log 20
  ask B1 "$PB" $N & b=$!
  wait $a $b
  ask A1solo "$PA" $N
fi
stop 1
if boot 2 "0:0,$FS:0"; then
  ask W2 "Count to three." 8
  ask A2 "$PA" $N & a=$!
  steps_at_least $O/server-2.log 21
  ask B2 "$PB" $N & b=$!
  wait $a $b
  ask B2solo "$PB" $N
fi
stop 2

python3 - "$O" one-card $N $FS >> $O/s.txt <<'EOF'
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
N, FS = int(sys.argv[3]), int(sys.argv[4])
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name, detail)
a1, a1s, b2, b2s = answer(resp("A1")), answer(resp("A1solo")), answer(resp("B2")), answer(resp("B2solo"))
check(f"boot1: A beside the poisoned B completes {N} tokens", a1 is not None and a1[3] == N, repr(a1)[-60:] if a1 else "")
check("boot1: A's answer beside the poisoned B is its solo answer, byte for byte", a1 is not None and a1 == a1s)
check("boot1: the poisoned B fails", answer(resp("B1")) is None, json.dumps(resp("B1"))[:200])
check("boot2: W (prefill poisoned) fails", answer(resp("W2")) is None)
check("boot2: the poisoned A fails", answer(resp("A2")) is None)
check(f"boot2: B beside the poisoned A completes {N} tokens", b2 is not None and b2[3] == N, repr(b2)[-60:] if b2 else "")
check("boot2: B's answer beside the poisoned A is its solo answer, byte for byte", b2 is not None and b2 == b2s)
check("A and B answer differently (the comparison can fail)", a1 is not None and b2 is not None and a1[:2] != b2[:2])
for n, want in (("1", 1), ("2", 2)):
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
check("boot1: one request failed alone, no whole step failed",
      sum("failing it alone" in l for l in open(f"{o}/server-1.log", errors="replace")) == 1
      and not any("request(s)" in l and "failing" in l for l in open(f"{o}/server-1.log", errors="replace")))
check("boot2: two requests failed alone, no whole step failed",
      sum("failing it alone" in l for l in open(f"{o}/server-2.log", errors="replace")) == 2
      and not any("request(s)" in l and "failing" in l for l in open(f"{o}/server-2.log", errors="replace")))
for n, step in (("1", FS), ("2", FS)):
    log = open(f"{o}/server-{n}.log", errors="replace").read().splitlines()
    at = [i for i, l in enumerate(log) if f"MEMRA_DSV41_CALL_FAULT: step {step} poisons" in l]
    nxt = next((l for l in log[at[0] + 1:] if l.startswith("[dsv41-step]")), "") if at else ""
    check(f"boot{n}: the poisoned step {step} is one batched call of two rows",
          " calls=1 " in nxt and " prefill=0 " in nxt and " decode_rows=2 " in nxt, nxt[:60])
arm = sys.argv[2]
for n in ("1", "2"):
    log = open(f"{o}/server-{n}.log", errors="replace").read().splitlines()
    steps = [l for l in log if l.startswith("[dsv41-step]")]
    empty = sum(" cache_slots=[] " in l for l in steps)
    pinned = [l for l in log if "in pinned host memory" in l and "routed experts" in l]
    print(f"boot{n}: residency {arm}: {len(steps)} steps, {empty} with cache_slots=[], {len(pinned)} host-experts line(s)")
    for l in pinned:
        print("  ", l[:300])
    if arm == "one-card":
        check(f"boot{n}: the one-card residency ran (experts in pinned host memory, a cache on every step)",
              steps and empty == 0 and len(pinned) >= 1)
    else:
        check(f"boot{n}: the resident trunk ran (no expert cache on any step)",
              steps and empty == len(steps) and not pinned)
print("SMOKE", "PASS" if ok else "FAIL")
EOF
echo "SMOKE-DONE $(date +%T)" >> $O/s.txt
