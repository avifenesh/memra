#!/usr/bin/env bash
# Normal-usage serving battery: boots memra-server like a user would and exercises the
# real API surface — chat (stream + non-stream), plain completions, concurrency, greedy
# determinism, and spec-vs-plain greedy identity (the exactness contract at
# the SERVING level, not just the kernel gates).
#
# Usage: tools/serve-smoke.sh [model.gguf [draft.gguf]]
# Defaults to the E4B QAT pair (fast load). Exits nonzero on any failed check.
set -uo pipefail
cd "$(dirname "$0")/.."

# No producer pipeline: grep -q may close after its first match, making echo
# fail with SIGPIPE under pipefail even when the complete SSE body is valid.
stream_has_frames() {
  grep -q '^data: ' <<<"$1" && grep -q 'data: \[DONE\]' <<<"$1"
}
if [ "${1:-}" = "--test-stream" ]; then
  sample=$(printf 'data: {"choices":[{"delta":{"content":"hello"}}]}\n%.0s' {1..500}; printf 'data: [DONE]\n')
  for _ in {1..10}; do
    stream_has_frames "$sample" || { echo 'FAIL valid large SSE body'; exit 1; }
  done
  for sample in '' 'data: {"choices":[]}' 'not an SSE response'; do
    if stream_has_frames "$sample"; then
      echo 'FAIL missing SSE sentinel accepted'; exit 1
    fi
  done
  echo 'PASS streaming assertion: 10 large valid bodies; 3 malformed bodies rejected'
  exit 0
fi

# Default = the 9B NVFP4 + its regime draft (full serving support; E4B's serve path is
# first-light only — dc/graph/spec unwired — and gemma assistant drafts use MEMRA_DRAFT,
# not the '+draft' NextN attach).
MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}"
[ -f "$MODEL" ] || { echo "serve-smoke: SKIP (no model at $MODEL)"; exit 0; }
PORT="${MEMRA_SMOKE_PORT:-8177}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
# PRE-FLIGHT PORT GUARD (GATE-INTEGRITY-20260819 A-16). Without it a foreign responder holding
# 8177 answers /health instantly, the boot wait never happens, and this battery measures
# someone else's model — see tools/port-guard.sh for the receipt.
. tools/port-guard.sh
memra_port_guard serve-smoke "$PORT" MEMRA_SMOKE_PORT || exit 1
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

# Build UNCONDITIONALLY (cargo is incremental — a no-op when fresh, cheap). The old
# `[ -x BIN ] || cargo build` idiom silently ran a STALE binary whenever one merely
# existed: on 2026-08-11 it booted a week-old (Aug-4) memra-server carrying the
# already-fixed gemma4 decode_step_batch panic, reporting a phantom regression against
# HEAD. A serving gate that can run a stale binary is a rotted gate (H100 lane law 3).
cargo build --release -p memra-server || { echo "serve-smoke: build FAILED"; exit 1; }

start_server() {  # extra env via prefix, e.g. start_server "smoke=/path.gguf"
  # MEMRA_COMPAT=openai: this battery tests the OpenAI-compatible surface the README
  # sells (the default native /v1/completions shape is a different contract).
  MEMRA_COMPAT=openai MEMRA_MODELS="$1" MEMRA_ADDR=$ADDR target/release/memra-server > /tmp/serve-smoke.log 2>&1 &
  SPID=$!
  # Belt and braces on the pre-flight guard: a process that grabbed the port in the window
  # between the check above and our bind would otherwise be measured as the model under test.
  for _ in $(seq 120); do
    curl -sf $BASE/health >/dev/null 2>&1 && { memra_port_owned serve-smoke "$PORT" "$SPID" || return 1; return 0; }
    sleep 2
  done
  echo "server did not come up; log tail:"; tail -5 /tmp/serve-smoke.log; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT

chat() {  # prompt max_tokens [stream] -> body to stdout
  local prompt=$1 maxtok=$2 stream=${3:-false}
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],
         \"max_tokens\":$maxtok,\"temperature\":0,\"stream\":$stream}"
}
# THE EMITTED STREAM = reasoning + content (gate-rot fix, 2026-08-04). The default smoke model
# (q9 Qwen3.5) is a REASONING model: at the small budgets this battery uses, every emitted token
# is still inside the thinking block, so `message.content` is legitimately "" with
# finish_reason=length. Asserting on `content` alone made checks 2/5/6/8 structurally
# unpassable — measured identically red on the pre-lane binary at 0a7349f6 (v0.68.0), i.e. this
# gate had been rotted in main, not broken by a lane. What the checks actually mean is "the
# server emitted deterministic non-empty output" and "spec emits the same text as plain", and
# that is the reasoning+content concatenation. Kept budget-independent on purpose: raising
# max_tokens until the model happens to close its thinking block would make the gate a
# model-verbosity coin flip.
say() { python3 -c '
import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
print((m.get("reasoning") or "") + (m.get("content") or ""))'; }

echo "== serve-smoke: plain serving =="
start_server "smoke=$MODEL" || exit 1

# 1. health + models
curl -sf $BASE/models | grep -q smoke && PASS "/models lists the model" || FAIL "/models"

# 2. non-stream chat: 200, non-empty content, usage populated
R=$(chat "Name three primary colors, comma-separated." 48)
echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
# reasoning OR content — a thinking model at a small budget emits only the former (see say()).
assert ((m.get("reasoning") or "") + (m.get("content") or "")).strip(), "empty emitted text"
assert r["usage"]["completion_tokens"] > 0, "no completion tokens"
assert r["choices"][0]["finish_reason"] in ("stop","length"), "bad finish_reason"
' && PASS "chat non-stream (text + usage + finish_reason)" || FAIL "chat non-stream"

# 3. streaming: data: chunks then [DONE]
S=$(chat "Count from one to five in words." 48 true)
stream_has_frames "$S" \
  && PASS "chat stream (SSE chunks + [DONE])" || FAIL "chat stream"

# 4. plain /v1/completions
curl -sf -m 300 $BASE/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"smoke","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["choices"][0]["text"].strip()' \
  && PASS "/v1/completions" || FAIL "/v1/completions"

# 5. greedy determinism: same prompt twice -> identical text
A=$(chat "Explain what a mutex is in one sentence." 64 | say)
B=$(chat "Explain what a mutex is in one sentence." 64 | say)
[ -n "$A" ] && [ "$A" = "$B" ] && PASS "greedy determinism (2 runs identical)" || FAIL "greedy determinism"

# 6. concurrency: 3 parallel chats all complete non-empty
pids=(); outs=()
for i in 1 2 3; do
  o=/tmp/serve-smoke-conc-$i.json
  outs+=("$o")
  ( chat "Write one fact about the number $i." 32 > "$o" ) & pids+=($!)
done
okc=0
for i in 0 1 2; do
  wait "${pids[$i]}" 2>/dev/null
  python3 -c "
import json
m = json.load(open('${outs[$i]}'))['choices'][0]['message']
assert ((m.get('reasoning') or '') + (m.get('content') or '')).strip()" 2>/dev/null && okc=$((okc+1))
done
[ $okc -eq 3 ] && PASS "3 concurrent chats" || FAIL "concurrency ($okc/3)"

# 7. long generation (covers the long-budget default; the graph door is explicit opt-in)
R=$(chat "Tell a short story about a lighthouse keeper." 300)
echo "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["usage"]["completion_tokens"] >= 100' \
  && PASS "long generation (>=100 tok)" || FAIL "long generation"

PLAIN_MUTEX=$A
stop_server

# 7b. CACHE-METERING exactness (lane/cache-metering): synthetic shared-prefix workload,
# per-request usage.prompt_tokens_details.cached_tokens exact against the learning
# sequence (seed/split/hit + cross-salt cold), /metrics totals closed-form, LCP
# histogram bucket-exact, per-tenant split exact, economics row crosschecked.
# MEMRA_SERVE_SPEC=0: the smoke model embeds an MTP head and spec sessions bypass the
# prefix cache by policy — the gate must run the batched bulk tier the cache serves.
# A FRESH server so the /metrics counters start from zero (the closed forms assume it).
echo "== serve-smoke: cache-metering exactness =="
export MEMRA_SERVE_SPEC=0
if start_server "smoke=$MODEL"; then
  python3 tools/cache-meter-gate.py $BASE smoke --n 5 --k 256 --suffix 16 \
    && PASS "cache-metering accounting exact (per-request + /metrics + economics)" \
    || FAIL "cache-metering gate"
  stop_server
else
  FAIL "cache-metering server did not start"
fi
unset MEMRA_SERVE_SPEC

# 8. spec serving: same greedy prompt must produce IDENTICAL text to plain serving
if [ -f "$DRAFT" ]; then
  echo "== serve-smoke: spec serving (draft attached) =="
  start_server "smoke=$MODEL+$DRAFT" || exit 1
  SA=$(chat "Explain what a mutex is in one sentence." 64 | say)
  [ -n "$SA" ] && [ "$SA" = "$PLAIN_MUTEX" ] && PASS "spec == plain greedy text (serving exactness)" \
    || FAIL "spec-vs-plain text mismatch"

  # 9. SAMPLED TRUNCATION MATRIX (added 2026-08-05, lane/sampler-truncation-fix).
  # THE GAP THIS CLOSES: every check above runs temperature 0. A greedy-only serve battery
  # cannot see any sampled-path defect, and it did not see this one — memra-server shipped a
  # top_p/min_p bug that spliced token id 0 ("!") mid-word on the published OpenAI surface
  # (found by the memra-vs-llama head-to-head arm, not by a gate:
  # research/memra-vs-llama-daily-20260805/logs/posthoc-lsampler.txt). Root cause and the
  # unit-level gate arms: research/sampfix-20260805/.
  #
  # The assertion is deliberately NOT "text equals a golden": sampled output legitimately
  # varies with the model, and a golden here would rot. It is a DIFFERENTIAL test against a
  # baseline that is STRUCTURALLY IMMUNE to this defect class, which makes it model- and
  # prompt-independent with no threshold to tune.
  #
  # Why untruncated t=0.8 is a sound baseline: the whole defect class is bad masking in the
  # filtered device path. With no truncation the filter threshold `th` is exactly 0, so nothing
  # is ever masked, the row can never go all-(-inf), and the argmax can never fall through to
  # its smallest-index tie-break. The untruncated arm therefore CANNOT exhibit the bug, and the
  # rate of low-id/odd characters it produces on this prompt is the model's honest baseline.
  #
  # The check (per truncated arm, same prompt, same seed, same temperature):
  #   (a) seeded reproducibility — same seed + same shape must reproduce the text exactly
  #   (b) non-empty text
  #   (c) the truncated arm's count of '!' must not EXCEED the immune baseline's count.
  #       Truncation removes low-probability tail tokens, so a correct truncated draw can only
  #       be *less* surprising than the untruncated one — more '!' after truncating is the
  #       signature, and it is what every corrupt arm did (measured on the pre-fix binary:
  #       baseline 0, top_p 12, llama-shape 12, min_p 2).
  #   (d) plus the raw structural forms that are corruption regardless of any baseline: a '!!'
  #       run, or a '!' spliced directly before an alphanumeric (`!bash`, `gpu-r!ig`) — a token
  #       boundary no healthy tokenizer emits.
  # Run through the SPEC server on purpose: truncation interacts with the rejection-sampling
  # verify, and the bug lived in the spec full-accept bonus path specifically.
  echo "== serve-smoke: sampled truncation matrix (spec server) =="
  samp() { # $1 = extra json sampling fields
    curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"List three shell commands that inspect a file, one per line.\"}],
           \"max_tokens\":64,\"temperature\":0.8,\"seed\":7${1:+,$1}}" | say
  }
  nbang() { printf '%s' "$1" | tr -cd '!' | wc -c; }
  # the immune reference: temp 0.8, NO truncation => th==0 => nothing masked => cannot corrupt
  TRUNC_BASE=$(samp '')
  TRUNC_BASE_N=$(nbang "$TRUNC_BASE")
  echo "  (immune baseline: untruncated t0.8 seed=7, bangs=$TRUNC_BASE_N)"
  check_trunc() { # $1 = json fields, $2 = label
    local t1 t2 bangs
    t1=$(samp "$1"); t2=$(samp "$1")
    if [ -z "$t1" ]; then FAIL "trunc $2: empty text"; return; fi
    if [ "$t1" != "$t2" ]; then FAIL "trunc $2: not reproducible at a fixed seed"; return; fi
    bangs=$(nbang "$t1")
    case "$t1" in
      *'!!'*) FAIL "trunc $2: '!!' run = id-0 fallthrough ($bangs '!')"; return ;;
    esac
    if printf '%s' "$t1" | grep -qE '![[:alnum:]]'; then
      FAIL "trunc $2: '!' spliced before an alphanumeric = id-0 injection ($bangs '!')"; return
    fi
    if [ "$bangs" -gt "$TRUNC_BASE_N" ]; then
      FAIL "trunc $2: $bangs '!' vs immune-baseline $TRUNC_BASE_N — truncation cannot ADD tail tokens"
      return
    fi
    PASS "trunc $2 (reproducible, bangs=$bangs <= baseline $TRUNC_BASE_N)"
  }
  check_trunc '"top_k":40'                                  'top_k=40'
  check_trunc '"top_p":0.95'                                'top_p=0.95'
  check_trunc '"min_p":0.05'                                'min_p=0.05'
  check_trunc '"top_k":40,"top_p":0.95,"min_p":0.05'        'llama-default k40+p0.95+m0.05'
  stop_server

  # 10. SESSION-AFFINITY EXACTNESS (added 2026-08-05, lane/session-affinity).
  # THE GAP THIS CLOSES: checks 1-9 all send prompts that either stand alone or EXTEND the
  # previous one, so they exercise the prefix probes and never the affinity path. Affinity
  # resumes a parked session whose committed KV the new prompt does NOT extend verbatim
  # (the client rewrote earlier assistant turns), which is the one resume class that can
  # serve a request from state for tokens the request no longer contains. That is exactly
  # the failure a serving battery must own, and no gate outside it would notice: the lane's
  # own root-cause bug (checkpoint one token PAST the prompt end) made affinity decline
  # 100% of the time while every other check stayed green.
  #
  # WHAT IS ASSERTED, AND WHY IT IS NOT "resumed == cold". The lane's obvious gate — resume
  # the session, then compare against a full cold prime of the same request — was written
  # first and FAILED on this model at turn 2. Isolation (receipts in
  # research/session-affinity-20260805/RESULTS.md, "prefill chunking") showed that assertion
  # is not a property this engine has, on ANY reuse tier: with a per-turn cache_salt and
  # MEMRA_AFFINITY=0 — no reuse of any kind — the SAME prompt primed at MEMRA_PRIME_CHUNK
  # 2048 vs 32 produces DIFFERENT greedy text (that same turn 2, diverging at char 52).
  # Chunk boundaries alone flip a near-tie argmax, and every resume necessarily re-chunks the
  # prefill (rewind boundary + delta, instead of one full prime), so "resumed == cold" would
  # be gating chunked prefill's reduction order, not affinity. Asserting it here would have
  # wired a permanently-red gate into the battery and blamed affinity for it.
  #
  # What affinity DOES own, and what this checks:
  #   (a) DETERMINISM of the resume path: the same conversation replayed twice against
  #       resuming servers must produce identical text. A resume that mixed in state from
  #       tokens the request does not contain could not be stable — the 25-turn lane run
  #       confirms this at scale (25/25 across 3 independent reps).
  #   (b) LIVENESS: the affinity arm's log must show a rewind. Without it a binary where
  #       affinity never fires passes (a) trivially — measured, that is exactly what the
  #       pre-96beb3a6 binary did.
  #   (c) The DECLINE path is silent: no "affinity rewind failed" in the log. That is the
  #       one line that means state was accepted and then could not be restored.
  # Both arms replay prompts rebuilt from ONE recorded history, so a divergence at turn N
  # cannot cascade into turn N+1 (two independently-driven conversations would diverge into
  # uninterpretable rows).
  # Full 25-turn version + TTFT curve: research/session-affinity-20260805/.
  echo "== serve-smoke: session-affinity resume exactness =="
  AFFPY=/tmp/serve-smoke-affinity.py
  cat > $AFFPY <<'PY'
import json, sys, urllib.request
PORT, MODE, PATHF = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
SID = "smoke-affinity"
# Control tokens are NOT required here: the explicit tier (session_id) nominates directly,
# which keeps this check independent of any particular model's chat template.
SYS = ("You are a terse assistant. Answer in one short sentence.\n\n"
       "FACTS: copies overlap with compute; pinned buffers bound host memory; "
       "bytes per token set the budget.\n\n")

def ask(prompt, n=48):
    body = {"model": "smoke", "prompt": prompt, "max_tokens": n,
            "temperature": 0, "session_id": SID}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as f:
        d = json.load(f)
    return d["choices"][0]["text"]

def render(hist):
    s = SYS
    for role, text in hist:
        s += f"{role}: {text}\n"
    return s + "assistant:"

def rewrite(text):
    """THE REWRITE CLASS: delete a span from the INTERIOR of the answer, leaving its
    boundaries intact — a think-strip in miniature. Not a tail chop (that is a plain
    prefix relation, which the pre-affinity probes already handle)."""
    if len(text) < 40:
        return text
    lo = len(text) // 3
    return text[:lo] + text[lo + len(text) // 3:]

QS = ["Why does overlapping copies with compute matter?",
      "How do pinned buffers relate to that?",
      "What sets the byte budget?",
      "Summarize all three in one sentence."]

if MODE == "record":
    hist, out = [], []
    for q in QS:
        hist.append(("user", q))
        t = ask(render(hist))
        out.append(t)
        hist.append(("assistant", rewrite(t)))
    json.dump({"hist": hist, "texts": out}, open(PATHF, "w"))
else:  # replay: same prompts, rebuilt from the RECORDED history
    rec = json.load(open(PATHF))
    hist = [tuple(x) for x in rec["hist"]]
    bad = []
    for i in range(0, len(hist), 2):
        got = ask(render(hist[:i + 1]))
        want = rec["texts"][i // 2]
        # Burst overshoot: a spec burst may stop up to K tokens past max_tokens and the two
        # arms' bursts need not align, so the shorter text being a PREFIX of the longer is
        # the same tolerance serve-st-gate check 4 applies. Anything else is a divergence.
        if not (got.startswith(want) or want.startswith(got)):
            bad.append(i // 2)
    print("MISMATCH " + ",".join(map(str, bad)) if bad else "IDENTICAL")
PY
  # Both arms resume (MEMRA_AFFINITY=1): this is a REPEATABILITY test of the resume path,
  # not a resume-vs-cold test (see above for why the latter is not a property of the engine).
  # Affinity is owned by parked spec sessions. PP-2 deliberately defaults spec admission OFF,
  # so force the documented always-spec seam here; otherwise this arm only tests prefix-cache
  # determinism and can never exercise a rewind.
  export MEMRA_AFFINITY=1
  export MEMRA_SPEC_GATE=0
  if start_server "smoke=$MODEL+$DRAFT"; then
    python3 $AFFPY "$PORT" record /tmp/serve-smoke-affinity.json 2>/dev/null
    # NB: `grep -c` already prints 0 on no-match (and exits 1), so a `|| echo 0` fallback
    # would append a SECOND line and break the integer test. Default only if the file is gone.
    REWINDS=$(grep -c 'spec-affinity: rewound' /tmp/serve-smoke.log 2>/dev/null)
    FAILED_REWIND=$(grep -c 'affinity rewind failed' /tmp/serve-smoke.log 2>/dev/null)
    REWINDS=${REWINDS:-0}; FAILED_REWIND=${FAILED_REWIND:-0}
    stop_server
    # a FRESH server, so the second arm re-primes and re-parks from scratch: the replay
    # cannot be served by state the recording arm left behind.
    if start_server "smoke=$MODEL+$DRAFT"; then
      VERDICT=$(python3 $AFFPY "$PORT" replay /tmp/serve-smoke-affinity.json 2>/dev/null)
      REWINDS2=$(grep -c 'spec-affinity: rewound' /tmp/serve-smoke.log 2>/dev/null)
      REWINDS2=${REWINDS2:-0}
      stop_server
      [ "$VERDICT" = IDENTICAL ] \
        && PASS "affinity resume is deterministic across servers (4 rewritten turns)" \
        || FAIL "affinity resume not reproducible (turns $VERDICT)"
      [ "${REWINDS2:-0}" -gt 0 ] \
        && PASS "replay arm resumed too ($REWINDS2 rewind(s))" \
        || FAIL "replay arm never rewound — only one arm exercised the resume path"
    else
      FAIL "affinity replay arm did not start"
    fi
    # LIVENESS: no rewind means the arms agreed because affinity never ran.
    [ "${REWINDS:-0}" -gt 0 ] \
      && PASS "affinity fired ($REWINDS rewind(s) on a rewritten history)" \
      || FAIL "affinity never rewound — the resume path was not exercised"
    # A rewind that FAILED means state was accepted and then could not be restored.
    [ "${FAILED_REWIND:-0}" -eq 0 ] \
      && PASS "no failed rewinds" \
      || FAIL "$FAILED_REWIND affinity rewind(s) failed after being accepted"
  else
    FAIL "affinity arm did not start"
  fi
  unset MEMRA_AFFINITY MEMRA_SPEC_GATE
else
  echo "== serve-smoke: spec arm SKIP (no draft at $DRAFT)"
fi

# 11. GEMMA4 ARM (lane/gemma4-serve-gaps, 2026-08-07). THE GAP THIS CLOSES: every check
# above runs a qwen-class model, and no gate anywhere booted a gemma4 model under the
# DEFAULT scheduler — which is exactly where ONE request panicked the worker, the respawn
# re-panicked, and the process FATALed (decode_step_batch had no gemma4 arm; receipt
# research/gemma4-serve-20260807/raw/repro-panic-server-*.log). Assertions:
#   (a) 1 request on the DEFAULT scheduler returns 200 with clean content (gap-1 regression);
#   (b) generation STOPS at the turn end (finish=stop, no <turn|> in content — the
#       eog_ids() union) and no channel/turn tag ever reaches content (gap 2);
#   (c) a thinking-ON request separates: reasoning non-empty, content still clean;
#   (d) the server survived: /health green after all of it, zero panic lines in the log.
# same canonical path as local-ci.sh's G12 (MEMRA_G12_MODEL overrides there too)
GEMMA_MODEL="${GEMMA_MODEL:-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf}"
if [ -f "$GEMMA_MODEL" ]; then
  echo "== serve-smoke: gemma4 arm (default scheduler; the worker-panic regression) =="
  if start_server "smoke=$GEMMA_MODEL"; then
    gchat() { # extra-json-fields
      curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ok\"}],
             \"max_tokens\":64,\"temperature\":0${1:+,$1}}"
    }
    R=$(gchat '')
    echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
c = r["choices"][0]["message"].get("content") or ""
assert c.strip(), "empty content"
for tag in ("<turn|>", "<|channel>", "<channel|>", "<|turn>"):
    assert tag not in c, f"tag leak: {tag} in {c!r}"
assert r["choices"][0]["finish_reason"] == "stop", "did not stop at turn end"
' && PASS "gemma4 1-request on DEFAULT scheduler (clean content, stop at turn end)" \
  || FAIL "gemma4 default-scheduler request"
    R=$(gchat '"reasoning_effort":"low"')
    echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
reasoning = m.get("reasoning") or ""
c = m.get("content") or ""
assert reasoning.strip(), "thinking-on: empty reasoning"
for tag in ("<turn|>", "<|channel>", "<channel|>", "<|turn>"):
    assert tag not in c, f"tag leak: {tag} in {c!r}"
assert c.strip(), "thinking-on: empty content"
' && PASS "gemma4 thinking-on: thought/content separated, tags stay syntax" \
  || FAIL "gemma4 thought/content separation"
    curl -sf $BASE/health >/dev/null && PASS "gemma4 server alive after both requests" \
      || FAIL "gemma4 server died"
    if grep -q "panicked" /tmp/serve-smoke.log; then
      FAIL "gemma4 arm: panic in the server log"
    else
      PASS "gemma4 arm: zero panics in the server log"
    fi
    stop_server
  else
    FAIL "gemma4 server did not start"
  fi
else
  echo "== serve-smoke: gemma4 arm SKIP (no model at $GEMMA_MODEL)"
fi

# 12. Q35 COLDHOL ARM (lane/cx-coldfix, 2026-08-12). A cached-only smoke cannot see
# carried cold-prefill corruption. The frozen sold shape supplies two cold misses among
# 18 full-cache hits at c=4; every request must return exactly 60 tokens. v0.81.0's
# continuation-capable prime batch made both misses select EOS at token 26 while all hits
# remained clean. This arm also pins the immediate safety contract: routed-MoE carried
# chunks stay serial until a multi-chunk + serving-decode gate qualifies that path.
Q35_COLD_MODEL="${MEMRA_Q35_COLD_MODEL:-/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
if [ -f "$Q35_COLD_MODEL" ]; then
  echo "== serve-smoke: Q35 mixed c=4 cold-prefill exactness =="
  unset MEMRA_PRIME_BATCH MEMRA_PREFILL_TICK MEMRA_PRIME_BATCH_MAX_T \
    MEMRA_PRIME_BATCH_HOLD_MS
  export MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_PREFIX_CACHE_MB=4096 \
    MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 \
    MEMRA_MAX_SESSIONS=96
  if start_server "q35-coldfix=$Q35_COLD_MODEL"; then
    Q35_GATE_LOG=/tmp/serve-smoke-q35-cold-mixed.log
    if python3 tools/q35-cold-mixed-gate.py --base "$BASE" --model q35-coldfix \
        --namespace serve-smoke-q35-coldfix --timeout 600 >"$Q35_GATE_LOG" 2>&1; then
      PASS "Q35 mixed c=4: 20/20 requests reached exactly 60 tokens"
    else
      tail -1 "$Q35_GATE_LOG"
      FAIL "Q35 mixed c=4 exact-token regression"
    fi
    if grep -Eq '^\[prime-batch\].*carried=[1-9]' /tmp/serve-smoke.log; then
      FAIL "Q35 routed-MoE entered a carried prime batch"
    else
      PASS "Q35 routed-MoE carried prime batches remain gated"
    fi
    stop_server
  else
    FAIL "Q35 cold-prefill server did not start"
  fi
  unset MEMRA_SERVE_SPEC MEMRA_CTX MEMRA_PREFIX_CACHE_MB MEMRA_PREFIX_DEDUP \
    MEMRA_REUSE_POOL MEMRA_AFFINITY MEMRA_MAX_SESSIONS
else
  echo "== serve-smoke: Q35 coldhol arm SKIP (no model at $Q35_COLD_MODEL)"
fi

echo "serve-smoke: $FAILS failed"
[ $FAILS -eq 0 ]
