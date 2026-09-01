#!/usr/bin/env bash
# gemma4 SPEC serve-stream identity gate (lane/gemma-batched stage 2, 2026-08-17, v2).
#
# Q8RP note: the 2026-08-17 mirror regression (NVFP4 prefill NaN on 96GB boots) is
# FIXED at lane/gemma-pnfold abf155e8 (build_q4_rp_swap qtype guard) — no pin needed
# at or after that commit; the boot output-sample gate is the standing guard.
#
# THE LOAD-KEY FINDING (v1 of this gate, receipts /tmp/gemma4-spec-gate-v1): comparing a
# no-drafter boot against a drafter boot measures the DOCUMENTED stream-k load key, not
# the spec route — MEMRA_DRAFT at load forces MMQ_SK tiling for n_embd >= 3500
# (hybrid.rs "SPEC-SERVING stream-k key ... governs the PRIME too"), so the two boots'
# primes are different FP compositions and near-tie completions legitimately diverge
# across boots. That is a load-config class difference (disclosed in SERVED-SPEC.md),
# not spec wrongness.
#
# THE SERVING LAW this gate enforces is WITHIN ONE BOOT: on a spec-armed server, a
# greedy request served through the spec route must be BYTE-IDENTICAL to the same
# request served through the plain (batched) fallback on the SAME loaded trunk.
#
#   phase 1 — solo sequential requests: admitted to spec (n_active == 0).
#   phase 2 — same prompts, each fired while a NON-GREEDY decoy holds a slot: the probe
#             admits at n_active >= 1 -> plain batched path, same trunk. The decoy is
#             temperature>0 so it can never take the spec route itself.
#   ambiguity — phase 1 must add [gspec-acc] lines; phase 2 must add ZERO.
#
# Usage: tools/serve-gemma4-spec-gate.sh [model.gguf] [draft.gguf] [max_tokens]
#   env: GATE_RANKS=<ranks file> (optional trim), GATE_K (default 5)
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/data/ai-ml/hf-models/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q8_0-MTP.gguf}"
MAXTOK="${3:-96}"
K="${GATE_K:-5}"
[ -f "$MODEL" ] || { echo "gemma4-spec-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -f "$DRAFT" ] || { echo "gemma4-spec-gate: SKIP (no draft at $DRAFT)"; exit 0; }
PORT="${MEMRA_G4SPEC_PORT:-8187}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
# PRE-FLIGHT PORT GUARD (GATE-INTEGRITY-20260819 A-16): this gate's first assertion is a boot
# log grep ("GEMMA SPEC route armed"). A foreign responder on the port answers /health, that grep
# reads OUR log and passes, and every later measurement is taken from a stranger's server.
. tools/port-guard.sh
memra_port_guard gemma4-spec-gate "$PORT" MEMRA_G4SPEC_PORT || exit 1
OUT="${GATE_OUT:-/tmp/gemma4-spec-gate}"
rm -rf "$OUT"; mkdir -p "$OUT"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

cargo build --release -p memra-server || { echo "gemma4-spec-gate: build FAILED"; exit 1; }

LOG="$OUT/server-spec.log"
# DEFAULT-ON proof (stage-3 flip): MEMRA_GEMMA4_SPEC is NOT set — the route must arm
# from MEMRA_DRAFT alone at the shipping K. GATE_K pins an explicit depth instead.
envs=(MEMRA_COMPAT=openai MEMRA_MODELS="g4=$MODEL" MEMRA_ADDR=$ADDR MEMRA_DRAFT="$DRAFT")
[ -n "${GATE_K_EXPLICIT:-}" ] && envs+=(MEMRA_GEMMA4_SPEC=$K)
[ -n "${GATE_RANKS:-}" ] && envs+=(MEMRA_GEMMA_DRAFT_RANKS="$GATE_RANKS" MEMRA_GEMMA_TRIM_ADAPT=512)
env "${envs[@]}" target/release/memra-server > "$LOG" 2>&1 &
SPID=$!
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT
for _ in $(seq 180); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { echo "server did not come up"; tail -8 "$LOG"; exit 1; }
# Belt and braces: the responder must BE our child before its log is treated as evidence.
memra_port_owned gemma4-spec-gate "$PORT" "$SPID" || exit 1
grep -q "GEMMA SPEC route armed" "$LOG" && PASS "boot arms the gemma spec route" \
  || FAIL "boot lacks the GEMMA SPEC notice (ambiguous run)"
boot_sample spec-boot


# FRESH-BOOT OUTPUT-SAMPLE GATE (gap-lane recert convention, e85125e9d): one real
# prompt per boot, non-degenerate text asserted — throughput and even byte-identity
# are blind to garbage (two degenerate sides can byte-match). >= 5 distinct words of
# >= 3 alpha chars, and no single word dominating > 50% of the text.
boot_sample() {  # $1 = tag
  local txt
  txt=$(ask "Explain binary search in two sentences." 0 64 2>/dev/null)
  echo "$1 BOOT-SAMPLE: $(printf '%s' "$txt" | head -c 90)" >> "$OUT/boot-samples.txt"
  local words uniq top
  words=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | wc -l)
  uniq=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort -u | wc -l)
  top=$(printf '%s' "$txt" | grep -oE '[A-Za-z]{3,}' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
  if [ "$uniq" -ge 5 ] && [ "$words" -gt 0 ] && [ $((top * 2)) -le "$words" ]; then
    PASS "$1 boot output-sample non-degenerate ($uniq distinct words)"
  else
    FAIL "$1 boot output-sample DEGENERATE (words=$words uniq=$uniq top=$top): $(printf '%s' "$txt" | head -c 60)"
  fi
}

PROMPTS=(
  "Explain the difference between a mutex and a semaphore."
  "Write a Rust function that computes the median of a slice of f64."
  "Summarize how HTTP/2 multiplexing works in one paragraph."
  "What are the first five Fibonacci numbers? Explain the recurrence."
  "Write a bash one-liner that finds the largest file in a directory tree."
  "Describe photosynthesis for a high-school student in three sentences."
)

ask() {  # $1 prompt, $2 temperature, $3 max_tokens
  local raw
  raw=$(curl -s -m 900 -w '\n%{http_code}' $BASE/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"g4\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],
         \"max_tokens\":$3,\"temperature\":$2,\"stream\":false}")
  local code="${raw##*$'\n'}" body="${raw%$'\n'*}"
  if [ "$code" != "200" ]; then
    echo "HTTP $code: $(echo "$body" | head -c 200)" >> "$OUT/errors.log"
    return 1
  fi
  printf '%s' "$body" | python3 -c '
import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
sys.stdout.write((m.get("reasoning") or "") + (m.get("content") or ""))'
}

echo "== phase 1: solo greedy (spec route) =="
i=0
for p in "${PROMPTS[@]}"; do
  ask "$p" 0 "$MAXTOK" > "$OUT/spec-p$i.txt" || FAIL "spec p$i errored"
  i=$((i+1))
done
GSPEC_AFTER_P1=$(grep -c "\[gspec-acc\]" "$LOG" || true)
[ "$GSPEC_AFTER_P1" -gt 0 ] && PASS "phase 1 served through spec ($GSPEC_AFTER_P1 burst lines)" \
  || FAIL "phase 1 produced no [gspec-acc] lines (ambiguous run)"

echo "== phase 2: same prompts, plain fallback (non-greedy decoy holds a slot) =="
i=0
for p in "${PROMPTS[@]}"; do
  ( ask "Write a slow meandering story about a lighthouse keeper." 0.7 300 > /dev/null ) &
  DECOY=$!
  sleep 1.5  # decoy admitted first; the probe below lands at n_active >= 1 -> plain
  ask "$p" 0 "$MAXTOK" > "$OUT/plain-p$i.txt" || FAIL "plain p$i errored"
  wait $DECOY || true
  i=$((i+1))
done
GSPEC_AFTER_P2=$(grep -c "\[gspec-acc\]" "$LOG" || true)
[ "$GSPEC_AFTER_P2" -eq "$GSPEC_AFTER_P1" ] \
  && PASS "phase 2 stayed plain (no new [gspec-acc] lines)" \
  || FAIL "phase 2 added [gspec-acc] lines ($GSPEC_AFTER_P1 -> $GSPEC_AFTER_P2) — probes were specced (ambiguous run)"

echo "== identity: spec vs plain, same boot, per prompt, byte compare =="
i=0
for _ in "${PROMPTS[@]}"; do
  if [ ! -s "$OUT/spec-p$i.txt" ]; then FAIL "p$i spec output empty"; i=$((i+1)); continue; fi
  if cmp -s "$OUT/spec-p$i.txt" "$OUT/plain-p$i.txt"; then
    PASS "p$i spec == plain (same trunk)"
  else
    FAIL "p$i spec != plain (served spec identity broken)"
    diff <(head -c 240 "$OUT/spec-p$i.txt") <(head -c 240 "$OUT/plain-p$i.txt") | head -6
  fi
  i=$((i+1))
done

if [ "$FAILS" -eq 0 ]; then
  echo "gemma4-spec-gate: ALL GREEN (${#PROMPTS[@]} prompts, within-boot spec == plain)"
else
  echo "gemma4-spec-gate: $FAILS FAILURES"; exit 1
fi
