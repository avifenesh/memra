#!/usr/bin/env bash
# Serve smoke of memra#1332 (the chat surface honours ignore_eos, memra#1325) on one RTX PRO 6000: memra-server on the
# checkpoint's whole trunk (40 layers, the residency the card picks), greedy chat requests of a prompt whose answer ends
# early ("Paris" stops at 19 tokens on the 2026-11-14 smoke):
#   eos-true   ignore_eos: true,  max_tokens 32 -> 32 completion tokens, finish_reason "length"
#   eos-false  ignore_eos: false, max_tokens 32 -> fewer than 32 tokens, finish_reason "stop"
#   eos-absent (no field)                        -> the same answer as eos-false, byte for byte
#   eos-null   ignore_eos: null (null = absent)  -> the same answer as eos-false, byte for byte
#   prefix     eos-true's text starts with eos-false's text (one program: the same greedy tokens until the stop)
#   completions (the /v1/completions surface, which honoured the field before) with ignore_eos: true -> 32 tokens
# Run inside a one-card lease (CUDA_VISIBLE_DEVICES set). PASS: s.txt ends with "SMOKE PASS" and "SMOKE-DONE".
# usage: smoke_eos.sh TAG [OUT_DIR]   (binaries perf-bins/TAG/memra-server)
set -u
TAG=$1
D=/home/access/dsv41
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
B=$D/perf-bins/$TAG
O=${2:-$D/receipts-perf/eos-$TAG}; mkdir -p $O; : > $O/s.txt
ADDR=${SMOKE_ADDR:-127.0.0.1:18394}
KEY=dsv41-eos-smoke
[ -x $B/memra-server ] || { echo "no $B/memra-server" | tee -a $O/s.txt; echo "SMOKE-DONE $(date +%T)" >> $O/s.txt; exit 2; }
sha256sum $B/memra-server >> $O/s.txt
cat $B/SOURCE >> $O/s.txt
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> $O/s.txt 2>&1
MEMRA_MODELS=v41=$C MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM $B/memra-server > $O/server.log 2>&1 &
SRV=$!
ready=0; t0=$SECONDS
for _ in $(seq 1 450); do
  kill -0 $SRV 2>/dev/null || { echo "server exited before readiness" >> $O/s.txt; break; }
  if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
  sleep 2
done
echo "boot: ready=$ready after $((SECONDS - t0))s" >> $O/s.txt
P="What is the capital of France?"
chat() {  # chat NAME EXTRA_JSON
  local body="{\"model\":\"v41\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":32,\"temperature\":0$2}"
  local code
  code=$(curl -s -o $O/resp-$1.json -w '%{http_code}' -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$body" --max-time 600 http://$ADDR/v1/chat/completions)
  echo "request $1 [$2]: http $code $(head -c 300 $O/resp-$1.json | tr '\n' ' ')" >> $O/s.txt
}
if [ $ready = 1 ]; then
  chat eos-true ',"ignore_eos":true'
  chat eos-false ',"ignore_eos":false'
  chat eos-absent ''
  chat eos-null ',"ignore_eos":null'
  body="{\"model\":\"v41\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0,\"ignore_eos\":true}"
  code=$(curl -s -o $O/resp-completions.json -w '%{http_code}' -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$body" --max-time 600 http://$ADDR/v1/completions)
  echo "request completions: http $code $(head -c 300 $O/resp-completions.json | tr '\n' ' ')" >> $O/s.txt
fi
kill -TERM $SRV 2>/dev/null
for _ in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -KILL $SRV 2>/dev/null; wait $SRV 2>/dev/null
python3 - "$O" >> $O/s.txt <<'EOF'
import json, sys
o = sys.argv[1]
def r(n):
    try:
        return json.load(open(f"{o}/resp-{n}.json"))
    except Exception:
        return {}
def a(n):
    try:
        c = r(n)["choices"][0]
        m = c.get("message") or {}
        return {"text": (m.get("reasoning") or "") + "\x00" + (m.get("content") or c.get("text") or ""),
                "finish": c.get("finish_reason"), "n": r(n)["usage"]["completion_tokens"]}
    except Exception:
        return None
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print("PASS" if cond else "FAIL", name, detail)
t, f, ab, nu, cm = a("eos-true"), a("eos-false"), a("eos-absent"), a("eos-null"), a("completions")
check("eos-true runs to max_tokens: 32 tokens, finish length", t and t["n"] == 32 and t["finish"] == "length", repr(t and (t["n"], t["finish"])))
check("eos-false stops early: fewer than 32 tokens, finish stop", f and f["n"] < 32 and f["finish"] == "stop", repr(f and (f["n"], f["finish"])))
check("no field = eos-false, byte for byte", ab is not None and ab == f)
check("null = no field, byte for byte", nu is not None and nu == f)
check("eos-true's text begins with eos-false's (the same greedy program until the stop)",
      t and f and t["text"].replace("\x00", "").startswith(f["text"].replace("\x00", "")))
check("completions with ignore_eos runs to 32 tokens", cm and cm["n"] == 32, repr(cm and (cm["n"], cm["finish"])))
print("SMOKE", "PASS" if ok else "FAIL")
EOF
echo "SMOKE-DONE $(date +%T)" >> $O/s.txt
