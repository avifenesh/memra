#!/usr/bin/env bash
# gemma4 FINAL SHIPPING BATTERY (lane/gemma-ship, 2026-08-17): the ONE re-bank that
# measures the shipping config after the Q8RP fix (abf155e8) — Q6_K-embd NVFP4mix
# trunk + official-Q8 assistant head + 447k ranks trim, DEFAULT ENV ONLY (mirror
# ENGAGED — no MEMRA_Q8RP pin anywhere; the fix is the guard, the boot output-sample
# gate is the sentinel).
#
# Interleaved x REPS, alternating boot order per rep (the A/B law):
#   default boot     — spec route armed from MEMRA_DRAFT alone: boot-sample gate,
#                      then spec-c1 prose/code + batch c8/c16 + mixed (driver rep).
#   kill-switch boot — MEMRA_GEMMA4_SPEC=0: boot-sample gate, then plain-c1 cells.
# Ambiguity: default boots must print the GEMMA SPEC notice + [gspec-acc] lines;
# kill-switch boots must print EAGER routing for spec and add zero [gspec-acc].
#
# Usage: CUDA_VISIBLE_DEVICES=N tools/gemma4-ship-battery.sh <model> <draft> <ranks> <outdir> [reps]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?model.gguf}"
DRAFT="${2:?draft.gguf}"
RANKS="${3:?ranks.txt}"
EV="${4:?outdir}"
REPS="${5:-5}"
ADDR=127.0.0.1:8183
BASE=http://$ADDR
mkdir -p "$EV"
OUT="$EV/ship-cells.jsonl"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

start_server() {  # $1 = "default" | "plain", $2 = log
  local envs=(MEMRA_COMPAT=openai MEMRA_MODELS="g4=$MODEL" MEMRA_ADDR=$ADDR
              MEMRA_DRAFT="$DRAFT" MEMRA_GEMMA_DRAFT_RANKS="$RANKS" MEMRA_GEMMA_TRIM_ADAPT=512)
  [ "$1" = "plain" ] && envs+=(MEMRA_GEMMA4_SPEC=0)
  env "${envs[@]}" target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 240); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; tail:"; tail -8 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
trap stop_server EXIT

boot_sample() {  # $1 = tag — fresh-boot output-sample gate (gap-lane convention)
  local txt
  txt=$(curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"g4","messages":[{"role":"user","content":"Explain binary search in two sentences."}],
         "max_tokens":64,"temperature":0,"stream":false}' |
    python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning") or "")+(m.get("content") or ""))' 2>/dev/null)
  echo "$1 BOOT-SAMPLE: $(printf '%s' "$txt" | head -c 90)" >> "$EV/boot-samples.txt"
  local words uniq top
  words=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | wc -l)
  uniq=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort -u | wc -l)
  top=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [ "$uniq" -ge 5 ] && [ "$words" -gt 0 ] && [ $((top * 2)) -le "$words" ]; then
    PASS "$1 boot-sample non-degenerate ($uniq distinct words)"
  else
    FAIL "$1 boot-sample DEGENERATE (words=$words uniq=$uniq top=$top)"
  fi
}

run_default_rep() {  # $1 = rep
  local log="$EV/server-default-r$1.log"
  start_server default "$log" || return 1
  grep -q "GEMMA SPEC route armed" "$log" || FAIL "rep $1 default boot lacks GEMMA SPEC notice"
  boot_sample "default-r$1"
  python3 tools/gemma4-spec-cells.py --base $BASE --reps 1 --out "$OUT" || FAIL "rep $1 default cells errored"
  grep -q "\[gspec-acc\]" "$log" || FAIL "rep $1 default boot never specced (ambiguous)"
  stop_server
}
run_plain_rep() {  # $1 = rep
  local log="$EV/server-plain-r$1.log"
  start_server plain "$log" || return 1
  boot_sample "plain-r$1"
  python3 tools/gemma4-spec-cells.py --base $BASE --reps 1 --plain-only --out "$OUT" || FAIL "rep $1 plain cells errored"
  if grep -q "\[gspec-acc\]" "$log"; then FAIL "rep $1 kill-switch boot SPECCED (ambiguous)"; fi
  stop_server
}

for rep in $(seq 1 "$REPS"); do
  echo "== rep $rep =="
  if [ $((rep % 2)) -eq 1 ]; then
    run_default_rep "$rep" || exit 1
    run_plain_rep "$rep" || exit 1
  else
    run_plain_rep "$rep" || exit 1
    run_default_rep "$rep" || exit 1
  fi
done
# NOTE: the driver labels its rep field 1 within each boot; the jsonl order + server
# logs carry the true rep. Post-process by arrival order.
if [ "$FAILS" -eq 0 ]; then
  echo "gemma4-ship-battery: ALL GREEN ($REPS reps, default env, mirror engaged)"
else
  echo "gemma4-ship-battery: $FAILS FAILURES"; exit 1
fi
