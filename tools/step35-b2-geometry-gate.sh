#!/usr/bin/env bash
# step35-b2-geometry-gate — the standing form of research/step-sku-20260807/b2-geometry-ab.sh.
#
# WHAT IT PINS (three assertions, ALL required):
#   1. TEXT IDENTITY: step35 served at c=2 and c=4 (concurrent identical greedy requests, the
#      DEFAULT batched scheduler) must return responses byte-identical to the c=1 reference.
#      This is the assertion whose pre-fix failure was HTTP-200 GARBAGE ('::::…') — over PP-2
#      (this SKU's only placement) a B>1 tick walked the generic Full arm: global n_head=96
#      over-reading wq on the 12 full-attn layers, 128-dim rope on all 45 layers, no SWA
#      window, no head-wise gate (research/step-sku-20260807/raw/b2ab-pre-20260807T091553Z.log).
#   2. BATCHED EVIDENCE: the server must actually have RUN a B>1 batched step35 walk —
#      (a) the spawn log's `decode chunk cap` for the step35 model must be >= 2, and
#      (b) the engine's one-shot `[step35-batch] first B>1` line must appear.
#      Without (2) the gate is vacuously green under the fail-closed B=1 pin (chunk_cap_for
#      returns 1, every "batched" tick is a B=1 chunk, and identity holds trivially).
#   3. LIVE TRANSITION: one streaming request must emit while alone (`ready=1`), then two late
#      requests join it and produce a later `ready>=2` tick. All three completions must remain
#      byte-identical to the c=1 reference. Static c=1/c=N comparisons cannot catch a scheduler
#      crossing between two individually-stable numeric classes inside one session.
#
# REGISTERED RED (lane/step35-batched-decode, 2026-08-08): under the B=1 pin, assertion 2
# fails by construction — `decode chunk cap 1` and no batched-walk line. The batched arm's
# commits turn it green. Same pattern as tickinv35's red registration.
#
# TEETH (--canary): sets MEMRA_STEP35_BATCH=0. Under PP-N, the Step35 B=1 correctness default
# now refuses the eager numeric class, so disabling the batched trunk makes requests fail closed;
# the canary PASSES only if the naked assertions FAIL. The canary changes the WORLD, not the
# label (the chunkinv lesson, written wrong twice there).
#
# Requires: 2 GPUs (the artifact fits only across a PRO 6000 pair), the step35 artifact.
# SKIPs cleanly when either is absent — a missing artifact must not read as a pass
# (fast-gate reads this script's own SKIP word).
#
#   tools/step35-b2-geometry-gate.sh [--canary] [--port N]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CANARY=0
PORT="${MEMRA_B2GEO_PORT:-8094}"
while [ $# -gt 0 ]; do
  case "$1" in
    --canary) CANARY=1; shift ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "step35-b2-geometry-gate: unknown arg $1"; exit 2 ;;
  esac
done

# ---- artifact resolution (MEMRA_STEP37_GGUF override; box1 + box2 staged locations) ----
MODEL="${MEMRA_STEP37_GGUF:-}"
if [ -z "$MODEL" ]; then
  for cand in \
    "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
    "/data/models/step37/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
    "/data/models/step37/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"; do
    [ -f "$cand" ] && { MODEL="$cand"; break; }
  done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
  echo "step35-b2-geometry-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"; exit 0; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" -ge 2 ] || {
  echo "step35-b2-geometry-gate: SKIP (needs 2 GPUs for the 105GB PP-2 placement, have $NGPU)"; exit 0; }
BIN=./target/release/memra-server
[ -x "$BIN" ] || { echo "step35-b2-geometry-gate: FAIL (no $BIN — build release first)"; exit 1; }
# PRE-FLIGHT PORT GUARD (GATE-INTEGRITY-20260819 A-16). The headline assertion here is BYTE
# IDENTITY against a c=1 reference; if a foreign responder holds 8094 the gate compares one
# stranger's output to another's and calls it a geometry proof. See tools/port-guard.sh.
. tools/port-guard.sh
memra_port_guard step35-b2-geometry-gate "$PORT" MEMRA_B2GEO_PORT || exit 1

# drafter (optional — trunk-only serve WARNs but works; attach when staged next to the trunk)
DRAFT="$(dirname "$(dirname "$MODEL")")/Step3.7-flash-mtp-Q8_0.gguf"
[ -f "$DRAFT" ] || DRAFT="$(dirname "$MODEL")/Step3.7-flash-mtp-Q8_0.gguf"
MODELS_SPEC="step35=${MODEL}"
[ -f "$DRAFT" ] && MODELS_SPEC="step35=${MODEL}+${DRAFT}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
D=research/step35-batch-20260808/raw
mkdir -p "$D"
TAG=$([ "$CANARY" = 1 ] && echo canary || echo naked)
SLOG=$D/b2geo35-server-$TAG-$TS.log
GLOG=$D/b2geo35-$TAG-$TS.log
# The verdict block's own file. $GLOG is written by the `tee` in a process substitution below,
# whose buffer this shell cannot flush, so the canary must not parse it. This file is written
# and closed inside the subshell that produces the verdicts.
VLOG=$D/b2geo35-verdicts-$TAG-$TS.log
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-gpu.lock}
BASE=http://127.0.0.1:$PORT

exec > >(tee "$GLOG") 2>&1
echo "=== step35-b2-geometry-gate tag=$TAG ts=$TS model=$MODEL draft=${DRAFT:-none} ==="

RC=1
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  # Serve config: LIVE defaults, PP-2, spec OFF per #87 (spec over PP-2 is quarantined).
  # In particular, do NOT pin MEMRA_SERVE_B1FAST=0: Step35's code-level exclusion remains
  # defense in depth and must keep B=1 batched even if the global policy regresses.
  CANARY_ENV=()
  [ "$CANARY" = 1 ] && CANARY_ENV=(MEMRA_STEP35_BATCH=0)
  env -u MEMRA_SERVE_B1FAST -u MEMRA_STEP35_BATCH "${CANARY_ENV[@]}" \
    MEMRA_MODELS="$MODELS_SPEC" MEMRA_SERVE_SPEC=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_CTX=262144 MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_TICK_TRACE=1 \
    "$BIN" > "$SLOG" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server died during boot"; sed -n '1,50p' "$SLOG"; exit 1; }
  done
  curl -sf "$BASE/readyz" >/dev/null 2>&1 || { echo "FAIL: server never became ready"; exit 1; }
  # Belt and braces: the ready responder must BE our child (see tools/port-guard.sh).
  memra_port_owned step35-b2-geometry-gate "$PORT" "$SRV" || exit 1

  BODY='{"model":"step35","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated, then explain in two sentences why 1 is not prime."}],"max_tokens":64,"temperature":0.0,"seed":3407}'
  STREAM_BODY='{"model":"step35","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated, then explain in two sentences why 1 is not prime."}],"max_tokens":64,"temperature":0.0,"seed":3407,"stream":true}'
  ask() { curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -c 'import json,sys
r=json.load(sys.stdin); c=r.get("choices")
if c:
    m=c[0]["message"]; print(json.dumps({"reasoning": m.get("reasoning"), "content": m.get("content")}))
else:
    print("ERROR", json.dumps(r.get("error")))'; }

  echo "--- c=1 reference (greedy) ---"
  ask > /tmp/b2geo35-ref.txt
  cat /tmp/b2geo35-ref.txt

  for C in 2 4; do
    echo "--- c=$C concurrent identical requests ---"
    # wait ONLY for the curl PIDs — a bare `wait` also waits on the server background
    # job, which never exits (found live: the gate hung after a fully-correct c=2 round).
    CURL_PIDS=()
    for i in $(seq 1 "$C"); do ask > "/tmp/b2geo35-c$C-$i.txt" & CURL_PIDS+=($!); done
    wait "${CURL_PIDS[@]}"
    cat /tmp/b2geo35-c$C-*.txt
  done

  # Stateful cell: release one request first, wait until it has emitted a real token, then
  # admit two late requests. The tick trace is sliced at the cell boundary and must prove
  # ready=1 followed by ready>=2; output equality alone would be vacuous if the rows never met.
  echo "--- explicit B=1 -> B>1 transition (one early + two late) ---"
  TRANS_TICK_START=$(( $(wc -l < "$SLOG") + 1 ))
  TRANS_EARLY_SSE=/tmp/b2geo35-transition-early.sse
  TRANS_EARLY_TEXT=/tmp/b2geo35-transition-early.txt
  TRANS_TICKS=$D/b2geo35-transition-ticks-$TAG-$TS.log
  : > "$TRANS_EARLY_SSE"
  : > /tmp/b2geo35-transition-late-1.txt
  : > /tmp/b2geo35-transition-late-2.txt
  curl -sN --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$STREAM_BODY" > "$TRANS_EARLY_SSE" &
  EARLY_PID=$!
  TRANS_FIRST_TOKEN=0
  for _ in $(seq 1 3000); do
    if grep -Eq '"(content|reasoning)"[[:space:]]*:[[:space:]]*"[^"]' "$TRANS_EARLY_SSE"; then
      TRANS_FIRST_TOKEN=1
      break
    fi
    kill -0 "$EARLY_PID" 2>/dev/null || break
    sleep 0.01
  done

  TRANS_TRANSPORT_OK=1
  if [ "$TRANS_FIRST_TOKEN" -eq 1 ]; then
    CURL_PIDS=()
    for i in 1 2; do
      ask > /tmp/b2geo35-transition-late-$i.txt & CURL_PIDS+=($!)
    done
    for pid in "${CURL_PIDS[@]}"; do wait "$pid" || TRANS_TRANSPORT_OK=0; done
  else
    echo "transition early row never emitted a content-bearing SSE frame"
    TRANS_TRANSPORT_OK=0
  fi
  wait "$EARLY_PID" || TRANS_TRANSPORT_OK=0

  python3 - "$TRANS_EARLY_SSE" > "$TRANS_EARLY_TEXT" <<'PY' || TRANS_TRANSPORT_OK=0
import json
import sys

reasoning = []
content = []
for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if not raw.startswith("data:"):
        continue
    payload = raw[5:].strip()
    if payload == "[DONE]":
        break
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        continue
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("reasoning"):
            reasoning.append(delta["reasoning"])
        if delta.get("content"):
            content.append(delta["content"])
sys.stdout.write("".join(reasoning) + "".join(content))
PY
  sed -n "${TRANS_TICK_START},\$p" "$SLOG" > "$TRANS_TICKS"
  if awk '
    /\[tick\]/ && /ready=1([ ]|$)/ { solo=1 }
    solo && /\[tick\]/ && /ready=([2-9]|[1-9][0-9]+)([ ]|$)/ { crossed=1; exit }
    END { exit !crossed }
  ' "$TRANS_TICKS"; then
    TRANS_WIDTH_OK=1
  else
    TRANS_WIDTH_OK=0
  fi
  cat /tmp/b2geo35-transition-late-*.txt 2>/dev/null || true
  echo "transition tick trace: $TRANS_TICKS"

  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"

  # ---- verdicts ----
  # `v` echoes AND banks. The canary arm at the bottom of this file has to parse WHICH
  # assertion failed, and it cannot parse $GLOG (a process-substitution tee this shell cannot
  # flush). It must not recount inside a pipeline either — that is how round 1's fixture
  # printed "1 passed / 0 failed" with an arm visibly failing (GATE-INTEGRITY-20260819 §3).
  : > "$VLOG"
  v() { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$VLOG"; }
  FAILS=0
  REF_OK=1
  REF=$(cat /tmp/b2geo35-ref.txt)
  if [ -z "$REF" ] || grep -q '^ERROR' /tmp/b2geo35-ref.txt; then
    v "FAIL: empty/error c=1 reference"
    FAILS=$((FAILS+1))
    REF_OK=0
  fi
  for C in 2 4; do
    for i in $(seq 1 "$C"); do
      ROW=$(cat "/tmp/b2geo35-c$C-$i.txt")
      if [ "$ROW" = "$REF" ]; then v "c${C}[$i] == ref"
      else v "FAIL: c${C}[$i] != ref"; echo "  ref: $REF"; echo "  got: $ROW"; FAILS=$((FAILS+1)); fi
    done
  done

  if [ "$TRANS_FIRST_TOKEN" -eq 1 ]; then v "transition early row emitted before late admission OK"
  else v "FAIL: transition early row did not emit before late admission"; FAILS=$((FAILS+1)); fi
  if [ "$TRANS_TRANSPORT_OK" -eq 1 ]; then v "transition transport OK"
  else v "FAIL: transition request transport failed"; FAILS=$((FAILS+1)); fi
  if [ "$TRANS_WIDTH_OK" -eq 1 ]; then v "transition width evidence OK: ready=1 -> ready>=2"
  else v "FAIL: no ready=1 -> ready>=2 sequence in transition tick trace"; FAILS=$((FAILS+1)); fi
  if [ "$REF_OK" -eq 1 ]; then
    python3 - /tmp/b2geo35-ref.txt > /tmp/b2geo35-ref-emitted.txt <<'PY'
import json
import sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
sys.stdout.write((m.get("reasoning") or "") + (m.get("content") or ""))
PY
    REF_EMIT_HASH=$(sha256sum /tmp/b2geo35-ref-emitted.txt | awk '{print $1}')
    EARLY_HASH=$(sha256sum "$TRANS_EARLY_TEXT" | awk '{print $1}')
    if [ "$EARLY_HASH" = "$REF_EMIT_HASH" ]; then v "transition early == ref ($EARLY_HASH)"
    else v "FAIL: transition early != ref ($EARLY_HASH vs $REF_EMIT_HASH)"; FAILS=$((FAILS+1)); fi
  else
    v "transition early/ref comparison skipped: c=1 reference failed"
  fi
  for i in 1 2; do
    if [ -f /tmp/b2geo35-transition-late-$i.txt ] && [ "$(cat /tmp/b2geo35-transition-late-$i.txt)" = "$REF" ]; then
      v "transition late[$i] == ref"
    else
      v "FAIL: transition late[$i] != ref"
      FAILS=$((FAILS+1))
    fi
  done

  # assertion 2a: the spawn-time chunk cap for step35 must admit B>1
  CAP=$(grep -oE 'step35: decode chunk cap [0-9]+' "$SLOG" | grep -oE '[0-9]+$' | head -1)
  if [ -n "$CAP" ] && [ "$CAP" -ge 2 ]; then v "chunk cap $CAP >= 2 OK"
  else v "FAIL: step35 decode chunk cap is '${CAP:-unset}' (< 2) — no B>1 walk was tested"; FAILS=$((FAILS+1)); fi
  # assertion 2b: a B>1 batched step35 walk actually executed
  if grep -q '\[step35-batch\] first B>1' "$SLOG"; then
    v "batched-walk evidence OK: $(grep -m1 -oE '\[step35-batch\] first B>1[^"]*' "$SLOG" | head -c 120)"
  else v "FAIL: no '[step35-batch] first B>1' line in the server log — no B>1 step35 tick ran"; FAILS=$((FAILS+1)); fi

  echo "server log: $SLOG"
  if [ "$FAILS" -eq 0 ]; then v "VERDICT: PASS (static widths + live B=1->B>1 transition byte-identical, batched ticks proven)"; exit 0
  else v "VERDICT: FAIL ($FAILS failed assertions)"; exit 1; fi
) 9>"$GPU_LOCK"
RC=$?

if [ "$CANARY" = 1 ]; then
  # CANARY EVIDENCE CONTRACT (GATE-INTEGRITY-20260819 A-10, fixed 2026-08-19).
  # This was `[ "$RC" -ne 0 ]` and nothing else. RC is 75 when `flock -w 3600` times out — the
  # lock was held by another lane and NOT ONE ASSERTION RAN — and 1 when the server died during
  # boot or never became ready. All three printed "CANARY OK (disabling the Step35 batched trunk
  # broke the live assertions as required)". A canary whose evidence is "something went wrong"
  # cannot certify the naked arm's teeth, which is the only reason it exists.
  #
  # MEMRA_STEP35_BATCH=0 has ONE guaranteed consequence in the verdict block: no
  # `[step35-batch] first B>1` line can be emitted, so assertion 2b must be the red one.
  # Assert that specific verdict, from the verdict block's own banked file.
  if [ "$RC" -eq 0 ]; then
    echo "step35-b2-geometry-gate: CANARY FAILED — gate PASSED under MEMRA_STEP35_BATCH=0 (no teeth)"
    exit 1
  fi
  if [ "$RC" -eq 75 ]; then
    echo "step35-b2-geometry-gate: CANARY INCONCLUSIVE — rc=75 is the flock -w 3600 timeout on"
    echo "  $GPU_LOCK. The gate never booted a server and asserted nothing; 'the lock was busy'"
    echo "  is not 'the batched trunk is load-bearing'. Re-run when the lock is free."
    exit 1
  fi
  if [ ! -s "$VLOG" ]; then
    echo "step35-b2-geometry-gate: CANARY INCONCLUSIVE — rc=$RC and the verdict block never ran"
    echo "  (no $VLOG). The run died before asserting: server died during boot, never became"
    echo "  ready, or the artifact/placement check failed. See $SLOG."
    exit 1
  fi
  if ! grep -q '^VERDICT: FAIL' "$VLOG"; then
    echo "step35-b2-geometry-gate: CANARY INCONCLUSIVE — rc=$RC but the verdict block did not"
    echo "  record a FAIL verdict. Banked verdicts: $VLOG"
    exit 1
  fi
  if ! grep -q "FAIL: no '\[step35-batch\] first B>1' line" "$VLOG"; then
    echo "step35-b2-geometry-gate: CANARY FAILED — the run went red, but NOT on the batched-walk"
    echo "  evidence assertion, which is the one MEMRA_STEP35_BATCH=0 must break. Something else"
    echo "  failed, so this proves nothing about assertion 2b. Banked verdicts: $VLOG"
    grep '^FAIL' "$VLOG" | sed 's/^/    /'
    exit 1
  fi
  echo "step35-b2-geometry-gate: CANARY OK — MEMRA_STEP35_BATCH=0 turned the batched-walk"
  echo "  evidence assertion red as required ($(grep -c '^FAIL' "$VLOG") failed assertion(s); $VLOG)"
  exit 0
fi
exit $RC
