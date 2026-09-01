#!/usr/bin/env bash
# gemma4 batched-decode SERVE-STREAM IDENTITY GATE (lane/gemma-batched, 2026-08-16).
#
# Q8RP note: the 2026-08-17 mirror regression (NVFP4 prefill NaN on 96GB boots) is
# FIXED at lane/gemma-pnfold abf155e8 (build_q4_rp_swap qtype guard) — no pin needed
# at or after that commit; the boot output-sample gate is the standing guard.
#
# The engine battery (decode-batch-gate) proved the arm exact at the engine call; this
# gate proves it at the SERVED level: the same greedy prompts through the real HTTP
# surface must produce BYTE-IDENTICAL completions
#   (a) kill switch (MEMRA_GEMMA4_BATCH=0), sequential — the eager reference path,
#   (b) DEFAULT env (var unset), sequential — B=1 chunks through gemma4_decode_batch
#       (the arm is default-on since the 2026-08-16 owner flip),
#   (c) DEFAULT env, concurrent — B=N chunks (the batchmate-isolation contract, served).
#
# REFUSE-ON-AMBIGUITY: (c) must show the "[gemma4-batch] first B>1" engine marker and
# the seam-on boots must print the "BATCHED DECODE" route notice — otherwise the run
# compared eager to eager and proves nothing; the gate FAILS rather than green-lighting
# an unexercised path. Exits nonzero on any mismatch.
#
# Usage: tools/serve-gemma4-batch-gate.sh [model.gguf] [concurrency] [max_tokens]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf}"
CONC="${2:-4}"
MAXTOK="${3:-48}"
[ -f "$MODEL" ] || { echo "gemma4-batch-gate: SKIP (no model at $MODEL)"; exit 0; }
# WAS 8179 — the SAME port as tools/serve-stress-gate.sh, a SECOND fixed-port collision beyond
# the apikeys/serve-st pair the round-1 audit enumerated (GATE-INTEGRITY-20260819 A-16). Moved
# to 8186 (unused across tools/) AND guarded, because a distinct default only removes the
# collision we happen to know about.
PORT="${MEMRA_G4BATCH_PORT:-8186}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
. tools/port-guard.sh
memra_port_guard gemma4-batch-gate "$PORT" MEMRA_G4BATCH_PORT || exit 1
OUT="${GATE_OUT:-/tmp/gemma4-batch-gate}"
rm -rf "$OUT"; mkdir -p "$OUT"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

cargo build --release -p memra-server || { echo "gemma4-batch-gate: build FAILED"; exit 1; }

start_server() {  # $1 = MEMRA_GEMMA4_BATCH value or "unset" (the shipping default), $2 = log
  local envargs=()
  if [ "$1" != "unset" ]; then envargs=(MEMRA_GEMMA4_BATCH="$1"); fi
  env "${envargs[@]}" MEMRA_COMPAT=openai MEMRA_MODELS="g4=$MODEL" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  # Belt and braces on the pre-flight guard: the healthy responder must BE our child.
  for _ in $(seq 180); do
    curl -sf $BASE/health >/dev/null 2>&1 \
      && { memra_port_owned gemma4-batch-gate "$PORT" "$SPID" || return 1; return 0; }
    sleep 2
  done
  echo "server did not come up; log tail:"; tail -8 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT

PROMPTS=(
  "Explain what a sliding-window attention layer does in two sentences."
  "Write a Python function that reverses a linked list."
  "List three prime numbers greater than 100 and say why they are prime."
  "Summarize the plot of Hamlet in one paragraph."
  "What is the derivative of x^3 * ln(x)? Show the steps."
  "Translate 'the quick brown fox jumps over the lazy dog' into French."
  "Give a SQL query returning the top 5 customers by total order value."
  "Describe the difference between TCP and UDP for a beginner."
)

ask() {  # prompt -> emitted text (reasoning+content concat, serve-smoke law) on stdout
  local prompt=$1
  curl -sf -m 600 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"g4\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],
         \"max_tokens\":$MAXTOK,\"temperature\":0,\"stream\":false}" |
  python3 -c '
import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
sys.stdout.write((m.get("reasoning") or "") + (m.get("content") or ""))'
}


# FRESH-BOOT OUTPUT-SAMPLE GATE (gap-lane recert convention, e85125e9d): real prompt,
# non-degenerate text asserted per boot — throughput/identity are blind to garbage.
boot_sample() {  # $1 = tag
  local txt
  txt=$(ask "Explain binary search in two sentences." 2>/dev/null)
  echo "$1 BOOT-SAMPLE: $(printf '%s' "$txt" | head -c 90)" >> "$OUT/boot-samples.txt"
  local words uniq top
  words=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | wc -l)
  uniq=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort -u | wc -l)
  top=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [ "$uniq" -ge 5 ] && [ "$words" -gt 0 ] && [ $((top * 2)) -le "$words" ]; then
    PASS "$1 boot output-sample non-degenerate ($uniq distinct words)"
  else
    FAIL "$1 boot output-sample DEGENERATE (words=$words uniq=$uniq top=$top)"
  fi
}

collect_sequential() {  # $1 = tag
  local tag=$1 i=0
  for p in "${PROMPTS[@]}"; do
    ask "$p" > "$OUT/$tag-p$i.txt" || FAIL "$tag p$i request errored"
    i=$((i+1))
  done
}

collect_concurrent() {  # $1 = tag — fire CONC at a time so decode chunks form at B>1
  local tag=$1 i=0
  while [ $i -lt ${#PROMPTS[@]} ]; do
    local pids=() j=0
    while [ $j -lt "$CONC" ] && [ $((i+j)) -lt ${#PROMPTS[@]} ]; do
      local k=$((i+j))
      ( ask "${PROMPTS[$k]}" > "$OUT/$tag-p$k.txt" ) & pids+=($!)
      j=$((j+1))
    done
    for pid in "${pids[@]}"; do wait "$pid" || FAIL "$tag concurrent request errored"; done
    i=$((i+CONC))
  done
}

echo "== boot A: kill switch ON (eager reference) =="
start_server 0 "$OUT/server-off.log" || exit 1
grep -q "EAGER-ONLY serving" "$OUT/server-off.log" || FAIL "seam-off boot lacks EAGER-ONLY notice"
boot_sample eager-boot
collect_sequential eager-c1
stop_server

echo "== boot B: DEFAULT env (batched arm, default-on) =="
start_server unset "$OUT/server-on.log" || exit 1
if grep -q "BATCHED DECODE (gemma4 dense arm, default-on" "$OUT/server-on.log"; then
  PASS "default-env boot routes gemma4 to batched decode"
else
  FAIL "default-env boot lacks the BATCHED DECODE route notice (ambiguous run)"
fi
boot_sample batched-boot
collect_sequential batch-c1
collect_concurrent batch-c$CONC
if grep -q "\[gemma4-batch\] first B>1" "$OUT/server-on.log"; then
  PASS "B>1 batched gemma4 walk engaged (engine marker present)"
else
  FAIL "no B>1 engine marker — concurrent phase never formed a batched chunk (ambiguous run)"
fi
stop_server

echo "== identity: eager-c1 vs batch-c1 vs batch-c$CONC, per prompt, byte compare =="
i=0
for _ in "${PROMPTS[@]}"; do
  if [ ! -s "$OUT/eager-c1-p$i.txt" ]; then FAIL "p$i eager output empty"; i=$((i+1)); continue; fi
  if cmp -s "$OUT/eager-c1-p$i.txt" "$OUT/batch-c1-p$i.txt"; then
    PASS "p$i eager-c1 == batch-c1"
  else
    FAIL "p$i eager-c1 != batch-c1 (B=1 served identity broken)"
  fi
  if cmp -s "$OUT/eager-c1-p$i.txt" "$OUT/batch-c$CONC-p$i.txt"; then
    PASS "p$i eager-c1 == batch-c$CONC"
  else
    FAIL "p$i eager-c1 != batch-c$CONC (batched serve changed a stream)"
  fi
  i=$((i+1))
done

if [ "$FAILS" -eq 0 ]; then
  echo "gemma4-batch-gate: ALL GREEN (${#PROMPTS[@]} prompts, c1 + c$CONC, byte-identical)"
else
  echo "gemma4-batch-gate: $FAILS FAILURES"; exit 1
fi
