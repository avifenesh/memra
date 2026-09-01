#!/usr/bin/env bash
# gemma4 TOOLS real-surface acceptance gate (lane/gemma4-tools, 2026-08-18).
#
# The engine law: the actual client behaviour against the actual server, never "renders =
# works". The renderer fixtures prove byte-parity with the official Google tooluse jinja; THIS
# gate proves the served loop end to end on the local 5090:
#
#   boot     — 31B tooluse-template trunk; caps.tools=true must be advertised, boot output
#              non-degenerate (house convention).
#   phase 1  — chat/completions + a weather tool: finish_reason=tool_calls and a well-formed
#              arguments JSON object carrying the location.
#   phase 2  — feed the tool result back: a coherent final answer (finish_reason=stop, the
#              tool span never leaks into content).
#   phase 3  — streamed variant: tool_calls deltas (id/name header + arguments) then [DONE].
#   phase 4  — REAL codex exec 0.147.0 over /v1/responses (wire_api=responses): a prompt that
#              forces a shell tool call; the marker string must round-trip through the
#              function_call -> tool run -> function_call_output cycle.
#
# The trunk MUST carry the tooluse template (<|turn> + <|tool>); the q4_0 QAT trunk does
# (verified: research/gemma4-tools-20260817 template diff). If a given artifact lacks <|tool>
# the gemma4 tools arm cannot engage and the gate SKIPS with that diagnosis rather than
# green-lighting an unexercised path.
#
# Usage: tools/serve-gemma4-tools-gate.sh [model.gguf] [max_tokens]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf}"
MAXTOK="${2:-256}"
# Codex sends a ~8k-token agentic prompt; the 31B's MONOLITHIC gemma4 prefill OOMs that on a
# 24GB card (weights 16.4G leave too little for the prefill spike — the curl phases below use
# short prompts and fit fine). This smaller tooluse trunk drives the IDENTICAL gemma4 tools arm
# and fits codex's context; phase 4 falls back to it only if the 31B OOMs. Override with
# GATE_CODEX_MODEL= to force a specific trunk (empty = no fallback, 31B-only).
CODEX_FALLBACK="${GATE_CODEX_MODEL-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf}"
[ -f "$MODEL" ] || { echo "gemma4-tools-gate: SKIP (no model at $MODEL)"; exit 0; }

# Port occupancy guard (GATE-INTEGRITY-20260819 A-16; deferred to this file's merge because
# lane/vendor-default-sampling had it open). No collision on 8191, so the risk is a foreign
# responder: this gate's first assertion is a boot-log grep, and a squatter answering /health
# would make it read OUR log and pass while every later measurement came from a stranger.
PORT="${MEMRA_G4TOOLS_PORT:-8191}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
KEY="gemma-tools-gate-key"
. tools/port-guard.sh
memra_port_guard gemma4-tools-gate "$PORT" MEMRA_G4TOOLS_PORT || exit 1
OUT="${GATE_OUT:-/tmp/gemma4-tools-gate}"
rm -rf "$OUT"; mkdir -p "$OUT"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

# Serialize every 5090 boot behind the shared lock (owner law). Held for the whole gate.
exec 9>/tmp/memra-5090.lock
flock -w "${GATE_LOCK_WAIT:-7200}" 9 || { echo "gemma4-tools-gate: could not take 5090 lock"; exit 1; }

# Confirm the trunk carries the tooluse template BEFORE a costly boot.
if ! python3 tools/gguf-has-tooluse.py "$MODEL" 2>/dev/null; then
  echo "gemma4-tools-gate: SKIP (trunk template lacks <|tool> — gemma4 tools arm cannot engage)"
  exit 0
fi

cargo build --release -p memra-server || { echo "gemma4-tools-gate: build FAILED"; exit 1; }

SPID=""
stop_server() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true; SPID=""; }
trap stop_server EXIT
start_server() {  # $1 = model path, $2 = ctx, $3 = log
  MEMRA_COMPAT=openai MEMRA_MODELS="g4=$1" MEMRA_ADDR=$ADDR MEMRA_API_KEY="$KEY" \
    MEMRA_CTX="$2" target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  # $SPID is the server itself (direct boot, no flock wrapper), so the post-boot ownership
  # check is sound here: the healthy responder must BE our child, closing the check-to-bind race.
  for _ in $(seq 240); do
    curl -sf $BASE/health >/dev/null 2>&1 && { memra_port_owned gemma4-tools-gate "$PORT" "$SPID" || return 1; return 0; }
    sleep 2
  done
  echo "server did not come up; log tail:"; tail -12 "$3"; return 1
}

LOG="$OUT/server.log"
start_server "$MODEL" 8192 "$LOG" || exit 1

# caps: /v1/models must advertise tools:true for the gemma trunk (deliverable D).
caps=$(curl -s $BASE/v1/models -H "Authorization: Bearer $KEY")
echo "$caps" | python3 -c '
import json,sys
m=json.load(sys.stdin)["data"][0]
sys.exit(0 if m.get("capabilities",{}).get("tools") is True else 1)' \
  && PASS "/v1/models advertises capabilities.tools=true" \
  || FAIL "/v1/models did NOT advertise tools=true"

# ---- boot output-sample non-degenerate check (house convention) ----
boot_txt=$(curl -s -m 300 $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"g4","messages":[{"role":"user","content":"Explain binary search in two sentences."}],"max_tokens":64,"temperature":0,"stream":false}' \
  | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; sys.stdout.write((m.get("content") or ""))')
echo "BOOT-SAMPLE: $(printf '%s' "$boot_txt" | head -c 100)" >> "$OUT/boot-sample.txt"
words=$(printf '%s' "$boot_txt" | grep -oE '[A-Za-z]{3,}' | wc -l)
uniq=$(printf '%s' "$boot_txt" | grep -oE '[A-Za-z]{3,}' | sort -u | wc -l)
top=$(printf '%s' "$boot_txt" | grep -oE '[A-Za-z]{3,}' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
if [ "${uniq:-0}" -ge 5 ] && [ "${words:-0}" -gt 0 ] && [ $(( ${top:-0} * 2 )) -le "${words:-0}" ]; then
  PASS "boot output-sample non-degenerate ($uniq distinct words)"
else
  FAIL "boot output-sample DEGENERATE (words=$words uniq=$uniq top=$top): $(printf '%s' "$boot_txt" | head -c 60)"
fi

WEATHER_TOOL='{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}'

# ---- phase 1: model must CALL the tool ----
echo "== phase 1: chat/completions tool call =="
req1=$(printf '{"model":"g4","messages":[{"role":"user","content":"What is the weather in Paris right now? Use the get_weather tool."}],"tools":[%s],"max_tokens":%s,"temperature":0,"stream":false}' "$WEATHER_TOOL" "$MAXTOK")
resp1=$(curl -s -m 600 $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d "$req1")
echo "$resp1" > "$OUT/phase1.json"
CALL_ID=$(echo "$resp1" | python3 -c '
import json, sys
try:
    r = json.load(sys.stdin)
except Exception as e:
    print("PARSEFAIL: " + str(e), file=sys.stderr); sys.exit(2)
ch = r["choices"][0]
fr = ch.get("finish_reason")
tcs = ch["message"].get("tool_calls") or []
if fr != "tool_calls" or not tcs:
    print("NOCALL finish_reason=%s tool_calls=%d" % (fr, len(tcs)), file=sys.stderr); sys.exit(1)
fn = tcs[0]["function"]
try:
    a = json.loads(fn["arguments"])
except Exception as e:
    print("BADARGS %r: %s" % (fn.get("arguments"), e), file=sys.stderr); sys.exit(1)
if fn["name"] != "get_weather" or "location" not in a:
    print("BADCALL name=%s args=%s" % (fn["name"], a), file=sys.stderr); sys.exit(1)
c = ch["message"].get("content") or ""
if "<|tool_call>" in c or "<tool_call|>" in c:
    print("SPANLEAK", file=sys.stderr); sys.exit(1)
sys.stdout.write(tcs[0].get("id") or "call_x")
' 2>"$OUT/phase1.err")
if [ -n "$CALL_ID" ]; then
  PASS "phase 1: finish_reason=tool_calls, well-formed get_weather args (id=$CALL_ID)"
else
  FAIL "phase 1: no valid tool call ($(cat "$OUT/phase1.err"))"
fi

# ---- phase 2: feed the result back, expect a coherent final answer ----
echo "== phase 2: tool result -> final answer =="
req2=$(printf '{"model":"g4","messages":[{"role":"user","content":"What is the weather in Paris right now? Use the get_weather tool."},{"role":"assistant","content":null,"tool_calls":[{"id":"%s","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"Paris\\"}"}}]},{"role":"tool","tool_call_id":"%s","content":"{\\"temp_c\\":21,\\"sky\\":\\"clear\\"}"}],"tools":[%s],"max_tokens":%s,"temperature":0,"stream":false}' "$CALL_ID" "$CALL_ID" "$WEATHER_TOOL" "$MAXTOK")
resp2=$(curl -s -m 600 $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d "$req2")
echo "$resp2" > "$OUT/phase2.json"
echo "$resp2" | python3 -c '
import json,sys,re
r=json.load(sys.stdin)
ch=r["choices"][0]
c=(ch["message"].get("content") or "")
if "<|tool_call>" in c or "<tool_call|>" in c or "<|tool_response>" in c:
  print("SPANLEAK", file=sys.stderr); sys.exit(1)
# coherent = mentions the temperature or the sky condition it was told
if re.search(r"21|clear", c, re.I):
  sys.exit(0)
print(f"INCOHERENT: {c[:160]!r}", file=sys.stderr); sys.exit(1)
' 2>"$OUT/phase2.err" \
  && PASS "phase 2: coherent final answer citing the tool result" \
  || FAIL "phase 2: final answer did not reflect the tool result ($(cat "$OUT/phase2.err"))"

# ---- phase 3: streamed tool call ----
echo "== phase 3: streamed tool_calls deltas =="
curl -s -N -m 600 $BASE/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d "$(printf '{"model":"g4","messages":[{"role":"user","content":"Weather in Oslo? Call get_weather."}],"tools":[%s],"max_tokens":%s,"temperature":0,"stream":true}' "$WEATHER_TOOL" "$MAXTOK")" \
  > "$OUT/phase3.sse"
if grep -q '"tool_calls"' "$OUT/phase3.sse" && grep -q '\[DONE\]' "$OUT/phase3.sse" \
   && grep -q '"finish_reason":"tool_calls"' "$OUT/phase3.sse"; then
  PASS "phase 3: stream carried tool_calls deltas + finish_reason + [DONE]"
else
  FAIL "phase 3: stream missing tool_calls/finish/[DONE] (see $OUT/phase3.sse)"
fi

# ---- phase 4: REAL codex exec over /v1/responses ----
# Genuine round-trip only: the marker must appear on a line that is NOT codex's echoed prompt,
# and the run must carry no OOM/disconnect error (an OOM leaves the marker ONLY in the prompt
# echo — a false pass the earlier assertion fell for).
#
# SAMPLING NOTE (lane/vendor-default-sampling, 2026-08-19). This is the ONE gate we own whose
# request body omits `temperature`, and deliberately so: the body is built by the real
# `codex exec` client, which sends no sampling fields. Pinning temperature here would mean
# hand-writing the body, which would stop testing the real client — the whole point of the phase.
#
# It is NOT a greedy-determinism cell and never was. Before the vendor-default lane an omitted
# temperature resolved to 1.0 with top_p 1.0 / top_k 0 (main.rs `default_temperature`), i.e. FULL
# untruncated temperature-1.0 sampling — this phase has always been probabilistic. Under the
# per-model vendor defaults it becomes temperature 1.0 with top_p 0.95 + top_k 64 (google's
# gemma-4 card), which truncates the tail and makes this assertion strictly MORE stable, not less.
# The assertion is behavioral (did the model drive a shell tool call and echo the marker), not
# byte-identity, so sampling is the honest regime for it.
#
# If this phase ever needs determinism, the fix is a hand-built body in a SEPARATE phase with an
# explicit "temperature": 0 — phases 1-3 above are exactly that. Do not "fix" flake here by
# reaching for a server-side knob: an env var that forces greedy for tests would be a product
# default that only exists to make a gate green.
echo "== phase 4: codex exec (real client, /v1/responses) =="
CODEX_HOME="$OUT/codex-home"; mkdir -p "$CODEX_HOME"
PORT="${ADDR##*:}"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "g4"
model_provider = "memra"
model_reasoning_effort = "none"
approval_policy = "never"
sandbox_mode = "workspace-write"
[model_providers.memra]
name = "memra"
base_url = "http://127.0.0.1:$PORT/v1"
wire_api = "responses"
env_key = "MEMRA_GATE_KEY"
EOF
CODEX_WORK="$OUT/codex-work"; mkdir -p "$CODEX_WORK"
CODEX_PROMPT="run this exact shell command: echo GEMMA-TOOLS-GATE — then tell me what it printed"

run_codex() {  # $1 = log path; echoes "ok" | "oom" | "no"
  ( cd "$CODEX_WORK" && MEMRA_GATE_KEY="$KEY" CODEX_HOME="$CODEX_HOME" \
      timeout 600 codex exec --skip-git-repo-check "$CODEX_PROMPT" ) > "$1" 2>&1 || true
  # marker on a line without the verb "run this exact" = it round-tripped through the tool,
  # not the prompt echo.
  if grep -q "GEMMA-TOOLS-GATE" "$1" && grep "GEMMA-TOOLS-GATE" "$1" | grep -qv "run this exact"; then
    echo ok
  elif grep -qiE "out of memory|stream disconnected|Reconnecting" "$1"; then
    echo oom
  else
    echo no
  fi
}

if ! command -v codex >/dev/null 2>&1; then
  FAIL "phase 4: codex CLI not on PATH"
else
  verdict=$(run_codex "$OUT/codex.log")
  if [ "$verdict" = "ok" ]; then
    PASS "phase 4: codex drove a shell tool call and the marker round-tripped (31B)"
  elif [ "$verdict" = "oom" ] && [ -n "$CODEX_FALLBACK" ] && [ -f "$CODEX_FALLBACK" ]; then
    echo "  note: 31B OOM'd on codex's ~8k prefill (24GB card) — retrying codex on the smaller"
    echo "        tooluse trunk (identical gemma4 tools arm): $CODEX_FALLBACK"
    stop_server
    if python3 tools/gguf-has-tooluse.py "$CODEX_FALLBACK" 2>/dev/null \
       && start_server "$CODEX_FALLBACK" 16384 "$OUT/server-codex.log"; then
      verdict=$(run_codex "$OUT/codex-fallback.log")
      if [ "$verdict" = "ok" ]; then
        PASS "phase 4: codex round-trip on the tooluse fallback trunk (31B OOMs codex on 24GB)"
      else
        FAIL "phase 4: codex fallback did not round-trip ($verdict; see $OUT/codex-fallback.log)"
      fi
    else
      FAIL "phase 4: fallback trunk lacks tooluse markers or did not boot"
    fi
  else
    FAIL "phase 4: marker did not round-trip through codex ($verdict; see $OUT/codex.log)"
  fi
fi

echo
if [ "$FAILS" -eq 0 ]; then
  echo "gemma4-tools-gate: ALL GREEN"
else
  echo "gemma4-tools-gate: $FAILS FAILURES"; exit 1
fi
