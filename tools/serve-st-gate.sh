#!/usr/bin/env bash
# serve-st gate (serve-st lane, 2026-08-04): an HF safetensors checkpoint DIR served by
# memra-server end-to-end, plus the CLI-vs-server exactness contract:
#
#   1. /models lists the ST model.
#   2. /v1/chat/completions returns coherent text through the checkpoint's own chat
#      template (tokenizer_config chat_template / chat_template.jinja).
#   3. EXACTNESS: the SAME checkpoint, SAME prompt through the SAME template, greedy:
#      run-gen's ST dir branch (tokenwise prime, eager decode) against the server
#      (batched prime + serving decode). The server arm runs
#      MEMRA_SERVE_SPEC=0: with spec on, Token events carry ONE id per flush (per spec
#      round since sse-cadence 2026-08-05; per burst before) so
#      the response `tokens` array is not the per-token stream (worker contract, not a
#      bug). The trailing EOS id is tolerated (the CLI stream includes it, server Token
#      events stop before it).
#      The two arms are two numeric configs by design (prefill GEMM at m=T against the
#      decode GEMV at m=1, the batched decode body against the eager chain), so their
#      streams are IDENTICAL only while no step is a near-tie. The check passes on
#      identical streams. Where they part (memra #1181: the block-128 FP8 class, whose
#      prime tile is not bit-identical to the m=1 path), it passes only when ALL of these
#      hold, and prints the step and the margin:
#        a. the step is a near-tie under the CLI's own logits: the CLI's logit for its
#           token minus its logit for the server's token is at most
#           MEMRA_PRIME_GATE_MARGIN (default 1.0, the margin bound of the batched-prime
#           gate #46 in crates/memra-engine/src/forward.rs). A flip on a wide margin
#           cannot come from FP composition and fails;
#        b. the served stream after that step is the CLI's own: run-gen teacher-forced
#           on the server's ids disagrees with them at the parting step and nowhere
#           else. This is stricter than the law (a second near-tie later in the window
#           fails and has to be priced by hand);
#        c. the difference is the numeric config and nothing else: a second server boot
#           on the CLI's config (MEMRA_SERVE_B1FAST=1 for the eager decode chain,
#           MEMRA_CONFIDENCE_TRACE for the tokenwise prime) produces the CLI's stream
#           exactly. THIS BOOT DEPENDS ON TWO DOORS: the trace door must keep priming
#           the prompt tokenwise (the check requires its file to have rows) and the
#           eager door must cover the plan. On a plan the eager door does not cover
#           (decode_batch.rs lists them: the Qwen3.5 MoE class and others) a parted
#           stream fails here; there is no same-config arm for it yet.
#      A server stream that ends where the CLI goes on has no server token to price and
#      fails. The law's drift ceiling (MEMRA_PRIME_GATE_MAXDIFF over the full row) is not
#      checked: no door reads the served logits.
#   4. ST SERVE-SPEC exactness (#68 closed 2026-08-04, quarantine LIFTED): the DEFAULT
#      server (spec bursts ON for MTP dir checkpoints) must produce the greedy text of
#      the plain serve oracle (MEMRA_SERVE_SPEC=0: the same worker and the same batched
#      program, speculation off) at a 400-token window, past the pre-fix corruption
#      onset (~250 tok). Two prompts: the short one above, and a long-form one that must
#      fill the window. The comparison is EQUALITY of text and completion_tokens. Both
#      arms serve one numeric class: the speculative verify forward runs the batched
#      class the plain decode serves (memra #1146), so a difference here is a
#      spec-exactness finding, not a near-tie between two programs, and the public
#      stream stops at the request budget in both arms, so no burst overshoot reaches
#      the client. The default arm must report speculative rounds, so the check cannot
#      pass by comparing plain with plain; a checkpoint with no embedded MTP tensors has
#      nothing to speculate with and the check is SKIPPED by name (the summary line says
#      so). Until memra #1158 removed it, the oracle was the legacy round-robin scheduler
#      (MEMRA_SERVE_BATCH=0, the per-session eager class) under a prefix match.
#      Root cause + receipts of the original quarantine: research/fp8ship-20260804/RESULTS.md
#      (the persistent draft graph replayed with dangling pool addresses; was never
#      ST-specific).
#
# Usage: tools/serve-st-gate.sh [st_dir]   (defaults to the local qwen3.5-4B BF16 ckpt)
# Scratch and every boot's server log go to one run directory, not fixed /tmp names: green
# removes it; a failed check or an early exit keeps it and prints its basename (saved stdout
# never carries a home path). ST_OUT=<dir> banks them there instead, relative to the caller.
# GPU: callers wrap in `flock /tmp/memra-5090.lock` per the box convention.
set -uo pipefail
# A relative ST_OUT is the caller's, not the repo root's.
case "${ST_OUT:-}" in ""|/*) ;; *) ST_OUT=$PWD/$ST_OUT ;; esac
cd "$(dirname "$0")/.."

ST="${1:-/data/ai-ml/hf-models/qwen35-4b-hf}"
[ -d "$ST" ] || { echo "serve-st-gate: SKIP (no checkpoint dir at $ST)"; exit 0; }
# WAS 8178 — the SAME port as tools/apikeys-gate.sh, with no occupancy check in either
# (GATE-INTEGRITY-20260819 A-16). Two gates on one number is not a hypothetical: run them
# concurrently, or leave one's server behind, and whichever binds second silently measures the
# first one's model. Moved to 8180 (unused across tools/) AND guarded, because a distinct
# default only removes the collision we know about.
PORT="${MEMRA_ST_PORT:-8180}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
. tools/port-guard.sh
memra_port_guard serve-st-gate "$PORT" MEMRA_ST_PORT || exit 1
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

# Build unconditionally — cargo is incremental (no-op when fresh). The `[ -x BIN ] ||`
# idiom silently ran STALE binaries when one merely existed (rotted gate, H100 law 3).
cargo build --release -p memra-server || exit 1
cargo build --release -p memra-engine --bin run-gen || exit 1

if [ -n "${ST_OUT:-}" ]; then
  mkdir -p "$ST_OUT" || exit 1
  ST_TMP=$(cd "$ST_OUT" && pwd)
else
  ST_TMP=$(mktemp -d "${TMPDIR:-/tmp}/serve-st-gate.XXXXXX") || exit 1
fi
ST_BOOT=0
ST_DONE=0
SLOG=$ST_TMP/serve-st-server-0.log
# Every exit: stop the server if one was started, then remove the run directory on green or
# name it otherwise.
finish() {
  rc=$?
  stop_server
  if [ "$ST_DONE" = 1 ] && [ "$FAILS" -eq 0 ]; then
    [ -n "${ST_OUT:-}" ] || rm -rf -- "$ST_TMP"
  else
    echo "serve-st-gate: logs kept in $(basename "$ST_TMP")"
  fi
  exit $rc
}
# No server, nothing to stop: an empty SPID must never reach kill (kill 0 is the process group).
stop_server() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null || true
  SPID=
}
trap finish EXIT

PROMPT="What is the capital of France? Answer in one short sentence."
NGEN=64

echo "== serve-st-gate: checkpoint $ST =="

# ---- CLI arm: run-gen ST dir branch, chat-templated greedy decode ----
CLI_LOG=$ST_TMP/serve-st-cli.log
MEMRA_CHAT=1 MEMRA_NGEN=$NGEN "${CARGO_TARGET_DIR:-target}"/release/run-gen "$ST" --prompt "$PROMPT" \
  > "$CLI_LOG" 2>&1 \
  || { echo "run-gen failed; log tail:"; tail -5 "$CLI_LOG"; exit 1; }
CLI_TOKENS=$(grep '^tokens: ' "$CLI_LOG" | tail -1 | sed 's/^tokens: //')
[ -n "$CLI_TOKENS" ] || { echo "run-gen printed no token stream"; exit 1; }


# ---- server arm: native shape (no MEMRA_COMPAT) so /v1/completions returns raw ids.
# MEMRA_SERVE_SPEC=0 for the token-exactness arm: spec bursts emit ONE Token event id
# per burst, so the response `tokens` array is only the per-token stream on the plain
# tokenwise path (worker contract). Spec-vs-plain identity is gated separately below.
start_server() {  # $1 = extra env (e.g. "MEMRA_SERVE_SPEC=0"), sets SPID and SLOG
  ST_BOOT=$((ST_BOOT+1))
  SLOG=$ST_TMP/serve-st-server-$ST_BOOT.log
  env $1 MEMRA_MODELS="st=$ST" MEMRA_ADDR=$ADDR "${CARGO_TARGET_DIR:-target}"/release/memra-server \
    > "$SLOG" 2>&1 &
  SPID=$!
  # 27B-class ST dirs CPU-dequant ~29 GB at load (~13 min); 240s was calibrated on the
  # 4B default ckpt and times out spuriously (fp8ship-20260804 official-27B run).
  # Belt and braces on the pre-flight guard: the healthy responder must BE our child.
  for _ in $(seq 600); do
    curl -sf $BASE/health >/dev/null 2>&1 \
      && { memra_port_owned serve-st-gate "$PORT" "$SPID" || return 1; return 0; }
    sleep 2
  done
  echo "server did not come up; log tail:"; tail -5 "$SLOG"; return 1
}
start_server "MEMRA_SERVE_SPEC=0" || exit 1

# 1. /models lists the ST checkpoint
curl -sf $BASE/models | grep -q '"st"' && PASS "/models lists the ST model" || FAIL "/models"

# 2. chat completion through the checkpoint's template: coherent text (Paris must appear
#    in content or the separated reasoning field), usage populated.
R=$(curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"st\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
       \"max_tokens\":400,\"temperature\":0}")
echo "$R" > "$ST_TMP/serve-st-chat.json"
echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
text = (m.get("content") or "") + (m.get("reasoning") or "")
assert text.strip(), "empty content+reasoning"
assert "paris" in text.lower(), f"incoherent answer: {text[:200]!r}"
assert r["usage"]["completion_tokens"] > 0, "no completion tokens"
' && PASS "chat completion coherent (Paris) via ST template" || FAIL "chat completion coherent"

# 3. exactness: server greedy token ids against CLI greedy token ids (same template render).
srv_tokens() {  # the 64-token greedy completion of the gate prompt, as a Python list
  curl -sf -m 300 $BASE/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"st\",\"prompt\":\"$PROMPT\",\"chat\":true,
         \"max_tokens\":$NGEN,\"temperature\":0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tokens"])'
}
# Exit 0 identical (EOS tolerated), 3 parted at a step both arms have a token for (prints
# "PART <step> <cli id> <server id>"), 1 anything else.
compare_streams() {  # $1 = CLI list, $2 = server list
  python3 - "$1" "$2" <<'EOF'
import ast, sys
cli = ast.literal_eval(sys.argv[1])
srv = ast.literal_eval(sys.argv[2])
# Server Token events stop BEFORE the EOS id; the CLI stream includes it.
if cli == srv or (len(cli) == len(srv) + 1 and cli[:-1] == srv):
    print(f"identical {len(srv)} ids (cli {len(cli)} incl. trailing eos)"
          if len(cli) != len(srv) else f"identical {len(srv)} ids")
    sys.exit(0)
div = next((i for i, (a, b) in enumerate(zip(cli, srv)) if a != b), min(len(cli), len(srv)))
print(f"DIVERGE at token {div}: cli={cli[div:div+5]} srv={srv[div:div+5]} "
      f"(lens {len(cli)}/{len(srv)})")
if div < len(cli) and div < len(srv):
    print(f"PART {div} {cli[div]} {srv[div]}")
    sys.exit(3)
sys.exit(1)
EOF
}
SRV_TOKENS=$(srv_tokens)
CMP=$(compare_streams "$CLI_TOKENS" "$SRV_TOKENS"); CMP_RC=$?
echo "$CMP" | grep -v '^PART '
if [ $CMP_RC -eq 0 ]; then
  PASS "CLI-vs-server greedy token streams identical"
elif [ $CMP_RC -eq 3 ]; then
  # The streams part at a step both arms have a token for. Two numeric configs may part on
  # a near-tie and on nothing else. One run-gen run teacher-forced on the SERVER's ids gives
  # the CLI's logit row at the parting step (the prefix before it is common, so it is the
  # row the CLI drew its own token from) and counts every step where the CLI's argmax is
  # not the server's token: rules a and b. Then the server on the CLI's config: rule c.
  read -r _ DIV CLI_ID SRV_ID <<< "$(echo "$CMP" | grep '^PART ')"
  MARGIN_BOUND="${MEMRA_PRIME_GATE_MARGIN:-1.0}"
  echo "$SRV_TOKENS" > "$ST_TMP/serve-st-srv.ids"
  N_SRV=$(echo "$SRV_TOKENS" | tr -cs '0-9' '\n' | grep -c .)
  WHY=
  MEMRA_CHAT=1 MEMRA_NGEN=$N_SRV MEMRA_FORCE_TOKENS_FILE="$ST_TMP/serve-st-srv.ids" \
    MEMRA_FORCE_LOGITS_AT=$DIV MEMRA_FORCE_LOGITS_FILE="$ST_TMP/serve-st-cli-row.f32" \
    "${CARGO_TARGET_DIR:-target}"/release/run-gen "$ST" --prompt "$PROMPT" \
    > "$ST_TMP/serve-st-cli-forced.log" 2>&1 \
    || WHY="run-gen teacher-forced on the server's ids failed"
  if [ -z "$WHY" ]; then
    python3 - "$ST_TMP/serve-st-cli-row.f32" "$DIV" "$CLI_ID" "$SRV_ID" "$MARGIN_BOUND" <<'EOF' \
      || WHY="the streams part at token $DIV on a margin above the near-tie bound"
import struct, sys
raw = open(sys.argv[1], "rb").read()
row = struct.unpack(f"<{len(raw) // 4}f", raw)
div, cli_id, srv_id, bound = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
top = max(range(len(row)), key=row.__getitem__)
margin = row[cli_id] - row[srv_id]
print(f"token {div} under the CLI's logits: its token {cli_id} = {row[cli_id]:.4f}, the server's "
      f"{srv_id} = {row[srv_id]:.4f}, margin {margin:.4f} (bound {bound})")
if top != cli_id:
    print(f"the CLI's argmax at token {div} of the forced run is {top}, not its own token {cli_id}")
    sys.exit(1)
sys.exit(0 if margin <= bound else 1)
EOF
  fi
  rm -f "$ST_TMP/serve-st-cli-row.f32"
  if [ -z "$WHY" ]; then
    TF=$(grep '^teacher-forced EXACTNESS' "$ST_TMP/serve-st-cli-forced.log" | tail -1)
    echo "the CLI on the server's ids: ${TF:-no EXACTNESS line}"
    case "$TF" in
      *"disagreements=1/"*" first_at=$DIV "*) ;;
      *) WHY="after token $DIV the served stream is not the CLI's argmax path (more than the one near-tie step differs)" ;;
    esac
  fi
  if [ -z "$WHY" ]; then
    stop_server
    TRACE=$ST_TMP/serve-st-confidence.jsonl
    case "$TRACE" in
      *[[:space:]]*) WHY="the run directory's path holds whitespace, which the same-config boot's environment cannot carry" ;;
    esac
  fi
  if [ -z "$WHY" ]; then
    if start_server "MEMRA_SERVE_SPEC=0 MEMRA_SERVE_B1FAST=1 MEMRA_CONFIDENCE_TRACE=$TRACE"; then
      SAME_TOKENS=$(srv_tokens)
      SAME=$(compare_streams "$CLI_TOKENS" "$SAME_TOKENS") \
        || WHY="the server on the CLI's numeric config (MEMRA_SERVE_B1FAST=1 and MEMRA_CONFIDENCE_TRACE's tokenwise prime) does not write the CLI's stream"
      echo "server on the CLI's numeric config: $(echo "$SAME" | grep -v '^PART ')"
      [ -s "$TRACE" ] \
        || WHY="MEMRA_CONFIDENCE_TRACE wrote no rows: the tokenwise-prime door the same-config boot relies on did not engage"
    else
      WHY="the same-config server boot did not come up"
    fi
  fi
  if [ -z "$WHY" ]; then
    PASS "CLI-vs-server streams part at token $DIV on a near-tie between two numeric configs; the served tail is the CLI's and the server on the CLI's config is identical"
  else
    FAIL "CLI-vs-server exactness: $WHY"
  fi
else
  FAIL "CLI-vs-server exactness"
fi

stop_server

# 4. ST serve-spec exactness (#68 closed, quarantine lifted): DEFAULT server (spec
#    bursts ON) vs the plain serve oracle (MEMRA_SERVE_SPEC=0: plain decode on the same
#    batched program). Two greedy prompts per arm at max_tokens 400: the short gate
#    prompt, which the model ends by itself, and a long-form prompt that must fill the
#    window (pre-fix corruption onset ~250 tokens). Text and completion_tokens must be
#    EQUAL: the engine may commit accepted drafts past the budget inside the session
#    cache, but the public stream stops at the request budget, so there is no overshoot
#    to tolerate. The default arm must report speculative rounds, or the check compared
#    plain with plain. A checkpoint with no embedded MTP tensors has nothing to
#    speculate with: the check is SKIPPED by name, and the summary line says so.
MAXTOK=400
WINDOW_PROMPT="Explain, step by step and in full detail, how a speculative decoding scheduler should decide between running a draft-and-verify burst and taking a batched decode step when many sessions share one GPU. Cover the throughput arithmetic, the latency consequences, and how hysteresis prevents mode thrash."
# The same test docs/ONBOARDING.md applies: any tensor named mtp.* in the checkpoint. A dir
# with neither an index nor a top-level safetensors file is "unknown" and runs the check.
HAS_MTP=$(python3 - "$ST" <<'EOF'
import glob, json, os, struct, sys

d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
shards = glob.glob(os.path.join(d, "*.safetensors"))
if os.path.exists(idx):
    names = list(json.load(open(idx)).get("weight_map", {}))
elif shards:
    names = []
    for f in shards:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            names += [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
else:
    print("unknown")
    raise SystemExit(0)
print("yes" if any(k.startswith("mtp.") for k in names) else "no")
EOF
)
grab_chat() {  # $1 = prompt, $2 = outfile (the whole response JSON)
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"st\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],
         \"max_tokens\":$MAXTOK,\"temperature\":0}" > "$2"
}
SKIPPED=""
if [ "$HAS_MTP" = no ]; then
  echo "  skip: check 4 (spec exactness): the checkpoint carries no embedded MTP tensors, so the default server does not speculate"
  SKIPPED=" (check 4 skipped: no MTP head)"
else
start_server "MEMRA_SERVE_SPEC=0" || exit 1
grab_chat "$PROMPT" "$ST_TMP/serve-st-plain-short.json"
grab_chat "$WINDOW_PROMPT" "$ST_TMP/serve-st-plain-window.json"
stop_server
start_server "" || exit 1
if grep -q "QUARANTINED" "$SLOG"; then
  FAIL "stale quarantine notice at load (#68 is closed; lift regressed)"
else
  PASS "no quarantine notice (dir checkpoints spec-eligible)"
fi
grab_chat "$PROMPT" "$ST_TMP/serve-st-spec-short.json"
grab_chat "$WINDOW_PROMPT" "$ST_TMP/serve-st-spec-window.json"
python3 - "$ST_TMP" "$MAXTOK" <<'EOF'
import json, sys

tmp, budget = sys.argv[1], int(sys.argv[2])

def load(arm, prompt):
    r = json.load(open(f"{tmp}/serve-st-{arm}-{prompt}.json"))
    m = r["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning") or "")
    return text, r["usage"]["completion_tokens"], (r["usage"].get("spec") or {}).get("rounds", 0)

bad = 0
for prompt in ("short", "window"):
    spec, n_spec, rounds = load("spec", prompt)
    plain, n_plain, plain_rounds = load("plain", prompt)
    if not spec.strip() or not plain.strip():
        print(f"{prompt}: EMPTY text (spec {len(spec)} chars, plain {len(plain)} chars)")
        bad += 1
        continue
    if plain_rounds:
        print(f"{prompt}: the MEMRA_SERVE_SPEC=0 oracle reported {plain_rounds} speculative rounds")
        bad += 1
    if not rounds:
        print(f"{prompt}: the default server reported no speculative rounds: plain was compared "
              f"with plain")
        bad += 1
    if prompt == "window" and n_plain != budget:
        print(f"window: the plain oracle stopped at {n_plain} of {budget} tokens: the window "
              f"prompt no longer fills the window")
        bad += 1
    if spec == plain and n_spec == n_plain:
        print(f"{prompt}: default (spec) text equals the plain serve oracle "
              f"({len(spec)} chars, {n_spec} tokens, {rounds} speculative rounds)")
        continue
    n = min(len(spec), len(plain))
    d = next((j for j in range(n) if spec[j] != plain[j]), n)
    print(f"{prompt}: DIVERGE at char {d}: spec={spec[d:d+40]!r} plain={plain[d:d+40]!r} "
          f"(chars {len(spec)}/{len(plain)}, tokens {n_spec}/{n_plain})")
    bad += 1
raise SystemExit(1 if bad else 0)
EOF
[ $? -eq 0 ] && PASS "default (spec) server == plain serve oracle (400-tok window)" \
             || FAIL "default (spec) server diverged from the plain serve oracle"
stop_server
fi

ST_DONE=1
echo "serve-st-gate: $FAILS failed$SKIPPED"
[ $FAILS -eq 0 ]
