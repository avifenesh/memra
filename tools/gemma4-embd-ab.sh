#!/usr/bin/env bash
# gemma4 embd DECOMPOSITION A/B (lane/gemma-ship, 2026-08-17): one-variable answer for
# the ship-bank prose delta (135.5 pinned -> 121.3 ship). Candidates: Q6K embd vs the
# pnfold×assistant-drafter interaction. This cell isolates the EMBD variable: same ship
# HEAD, same head/ranks, default env, mirror engaged — trunk A (Q8_0 embd) vs trunk B
# (Q6K embd), spec-c1 prose+code only, interleaved xREPS with alternating boot order
# (the A/B law). Acceptance per class read from each boot's [gspec-acc] cum lines.
#
# Usage: CUDA_VISIBLE_DEVICES=N tools/gemma4-embd-ab.sh <trunkA> <trunkB> <draft> <ranks> <outdir> [reps]
set -uo pipefail
cd "$(dirname "$0")/.."

A="${1:?trunkA (q8embd)}"
B="${2:?trunkB (q6kembd)}"
DRAFT="${3:?draft.gguf}"
RANKS="${4:?ranks.txt}"
EV="${5:?outdir}"
REPS="${6:-5}"
ADDR=127.0.0.1:8184
BASE=http://$ADDR
mkdir -p "$EV"
OUT="$EV/embd-ab.jsonl"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

start_server() {  # $1 = model path, $2 = log
  env MEMRA_COMPAT=openai MEMRA_MODELS="g4=$1" MEMRA_ADDR=$ADDR \
      MEMRA_DRAFT="$DRAFT" MEMRA_GEMMA_DRAFT_RANKS="$RANKS" MEMRA_GEMMA_TRIM_ADAPT=512 \
      target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 240); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; tail:"; tail -6 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
trap stop_server EXIT

boot_sample() {  # $1 = tag
  local txt
  txt=$(curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"g4","messages":[{"role":"user","content":"Explain binary search in two sentences."}],
         "max_tokens":64,"temperature":0,"stream":false}' |
    python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning") or "")+(m.get("content") or ""))' 2>/dev/null)
  echo "$1 BOOT-SAMPLE: $(printf '%s' "$txt" | head -c 90)" >> "$EV/boot-samples.txt"
  local uniq
  uniq=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort -u | wc -l)
  [ "$uniq" -ge 5 ] && PASS "$1 boot-sample non-degenerate" || FAIL "$1 boot-sample DEGENERATE"
}

run_side() {  # $1 = model, $2 = tag, $3 = rep
  local log="$EV/server-$2-r$3.log"
  start_server "$1" "$log" || return 1
  grep -q "GEMMA SPEC route armed" "$log" || FAIL "r$3 $2 boot lacks GEMMA SPEC notice"
  boot_sample "$2-r$3"
  python3 tools/gemma4-spec-cells.py --base $BASE --reps 1 --spec-only --tag "@$2" --out "$OUT" \
    || FAIL "r$3 $2 cells errored"
  grep -q "\[gspec-acc\]" "$log" || FAIL "r$3 $2 never specced (ambiguous)"
  # acceptance per class: the boot serves sample, prose, code in order — take each
  # request group's LAST cum line (per-request counters reset at admit).
  grep "\[gspec-acc\]" "$log" | awk '{print $4}' > "$EV/acc-$2-r$3.txt"
  stop_server
}

for rep in $(seq 1 "$REPS"); do
  echo "== rep $rep =="
  if [ $((rep % 2)) -eq 1 ]; then
    run_side "$A" q8embd "$rep" || exit 1
    run_side "$B" q6kembd "$rep" || exit 1
  else
    run_side "$B" q6kembd "$rep" || exit 1
    run_side "$A" q8embd "$rep" || exit 1
  fi
done
[ "$FAILS" -eq 0 ] && echo "gemma4-embd-ab: ALL GREEN ($REPS reps interleaved)" \
  || { echo "gemma4-embd-ab: $FAILS FAILURES"; exit 1; }
