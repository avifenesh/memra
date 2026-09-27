#!/usr/bin/env bash
# tools/spec-ctx-edge-gate.sh: the speculative context-edge gate (memra#659). A request whose budget
# spans its whole session cap (max_tokens omitted) used to commit its last speculative round past
# the session cache: the out-of-bounds KV row poisoned later verifies (the #87 NaN trap) and the
# deferred draft-KV fill asserted `mtp_kv_fill: scratch overflow`, which killed the GPU worker and,
# on the second panic, the process. This gate boots the real memra-server on an MTP model, drives
# open requests to their cap back to back, and asserts every request completes and the boot
# survives. Every verdict is one line, verbatim, in `$OUT/VERDICTS.txt`; every boot keeps its
# server log, its request bodies and `nvidia-smi --query-compute-apps` before and after.
#
# Arms:
#   on     MEMRA_ADMIT_BY_MEMORY=1, MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=$SCE_OPEN (64), default spec.
#          Four open requests then one bounded control. Each open request: 200, finish_reason
#          `length`, completion_tokens == $SCE_OPEN (the charged output; the 8 slack rows stay
#          free). The control: 200, completion_tokens == its max_tokens.
#   plain  the `on` boot's door with MEMRA_SERVE_SPEC=0: one open request, its message byte-equal
#          to the `on` arm's first (spec and plain emit the same tokens up to the same bound).
#   off    door OFF, MEMRA_CTX=$SCE_CTX (384), default spec. Three open requests run away to the
#          cap: each 200, finish_reason `length`, 0 < completion_tokens <= cap - prompt_tokens.
#   Every arm's boot: no `panicked`, no `argmax sentinel`, no `[worker] FATAL`, no respawn, no
#   `spec verify refused` line; the server is alive after the last request; `/health` 200; spec
#   rounds ran (`[spec-acc]` lines) on the spec arms.
#
# Usage: tools/spec-ctx-edge-gate.sh <model.gguf> <memra-server binary> [out_dir]
#   SCE_PORT (8193), SCE_ARMS (on,plain,off), SCE_OPEN (64), SCE_CTX (384).
#   The binary is an argument, not built here, so the red arm (the unfixed tree) and the green arm
#   run the same gate. Run under the rig lock (`flock /tmp/memra-5090.lock`, or the collector on a
#   PRO box); the gate boots three servers in sequence and never takes the lock itself.
#   Exit 0 when no verdict FAILED; 1 on any FAIL; 2 on setup.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODEL="${1:-}"; BIN="${2:-}"
[ -n "$MODEL" ] && [ -n "$BIN" ] || { echo "usage: $0 <model.gguf> <memra-server> [out_dir]"; exit 2; }
[ -f "$MODEL" ] || { echo "spec-ctx-edge-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$BIN" ] || { echo "spec-ctx-edge-gate: FATAL: $BIN is not an executable"; exit 2; }
PORT="${SCE_PORT:-8193}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
ARMS="${SCE_ARMS:-on,plain,off}"
OPEN="${SCE_OPEN:-64}"
CTX="${SCE_CTX:-384}"
OUT="${3:-/tmp/spec-ctx-edge-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
VERDICTS="$OUT/VERDICTS.txt"
: > "$VERDICTS"
READY_WAIT_S="${SCE_READY_WAIT_S:-420}"
PROMPT='Count from 1 to 2000, one number per line, and write nothing else.'

. tools/port-guard.sh
memra_port_guard spec-ctx-edge-gate "$PORT" SCE_PORT || exit 2

sha256sum "$BIN" > "$OUT/binary.sha256"
{ echo "commit $(git rev-parse HEAD)"; echo "model $MODEL"; echo "binary $BIN"
  nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader; date -u +%FT%TZ; } > "$OUT/source.txt"

PASSN=0; FAILN=0
verdict() { # <line ending in -> PASS|FAIL>
  echo "$1" | tee -a "$VERDICTS"
  case "$1" in *"-> PASS") PASSN=$((PASSN+1));; *) FAILN=$((FAILN+1));; esac
}
jget() { # <file> <dotted.path> -> value or ""
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
try:
    v=json.load(open(sys.argv[1]))
    for k in sys.argv[2].split('.'):
        v=v[int(k)] if isinstance(v,list) else v[k]
    print("" if v is None else v)
except Exception:
    print("")
PY
}
msg_sha() { # <file>: sha256 of the first choice's message object (content and reasoning), or ""
  python3 - "$1" <<'PY' 2>/dev/null
import hashlib,json,sys
try:
    m=json.load(open(sys.argv[1]))["choices"][0]["message"]
    print(hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest())
except Exception:
    print("")
PY
}
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$1" 2>&1; }

SPID=0; D=""
boot() { # <label> [ENV=VAL ...]
  local label=$1; shift
  D="$OUT/$label"; mkdir -p "$D"
  gpu_apps "$D/compute-apps-before.csv"
  echo "$*" > "$D/env.txt"
  env MEMRA_COMPAT=openai MEMRA_MODELS="sce=$MODEL" MEMRA_ADDR=$ADDR "$@" "$BIN" > "$D/server.log" 2>&1 &
  SPID=$!
  local t=0
  until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$BASE/readyz" 2>/dev/null)" = 200 ]; do
    if ! kill -0 "$SPID" 2>/dev/null; then echo "  [$label] server died during boot; log tail:"; tail -n 5 "$D/server.log"; return 1; fi
    t=$((t+1)); [ "$t" -gt $((READY_WAIT_S*10)) ] && { echo "  [$label] FATAL: not ready within ${READY_WAIT_S}s"; return 1; }
    sleep 0.1
  done
  memra_port_owned spec-ctx-edge-gate "$PORT" "$SPID" || return 1
  echo "  [$label] ready (pid $SPID)"
}
stop() { # SIGTERM only a memra-server this gate started (pid AND cwd), record its exit code
  local label=$1 rc
  if kill -0 "$SPID" 2>/dev/null; then
    if [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ]; then kill -TERM "$SPID" 2>/dev/null
    else echo "  [$label] REFUSING to signal pid $SPID: cwd is not this tree"; fi
  fi
  wait "$SPID" 2>/dev/null; rc=$?
  echo "$rc" > "$D/exit"
  gpu_apps "$D/compute-apps-after.csv"
  SPID=0
}
cleanup() { [ "$SPID" != 0 ] && kill -0 "$SPID" 2>/dev/null && [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ] && kill -TERM "$SPID" 2>/dev/null; true; }
trap cleanup EXIT

chat() { # <outprefix> [max_tokens]: non-stream, temperature 0; prints the HTTP code
  local mt="" code
  [ -n "${2:-}" ] && mt=",\"max_tokens\":$2"
  code=$(curl -s -m 300 -o "$1.body" -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"sce\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"stream\":false$mt}") || code=000
  echo "${code:-000}"
}
# census <label> <spec:1|0>: the boot survived with no failure line; spec rounds ran when expected
census() {
  local label=$1 want_spec=$2 alive=0 health panics sentinels fatal respawn refused specacc v=FAIL
  kill -0 "$SPID" 2>/dev/null && alive=1
  health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" 2>/dev/null); health=${health:-000}
  panics=$(grep -c 'panicked' "$D/server.log"); sentinels=$(grep -c 'argmax sentinel' "$D/server.log")
  fatal=$(grep -c -F '[worker] FATAL' "$D/server.log"); respawn=$(grep -c 'respawn attempt' "$D/server.log")
  refused=$(grep -c 'spec verify refused' "$D/server.log"); specacc=$(grep -c -F '[spec-acc]' "$D/server.log")
  if [ "$alive" = 1 ] && [ "$health" = 200 ] && [ "$panics" = 0 ] && [ "$sentinels" = 0 ] && [ "$fatal" = 0 ] \
     && [ "$respawn" = 0 ] && [ "$refused" = 0 ]; then
    if [ "$want_spec" = 0 ] || [ "$specacc" -gt 0 ]; then v=PASS; fi
  fi
  verdict "SCE ($label) boot census: alive=$alive health=$health panicked=$panics argmax_sentinel=$sentinels worker_fatal=$fatal respawn=$respawn verify_refused=$refused spec_acc_lines=$specacc (spec expected=$want_spec) -> $v"
}
in_arms() { case ",$ARMS," in *",$1,"*) return 0;; *) return 1;; esac; }

echo "== spec-ctx-edge-gate: arms=$ARMS port=$PORT open=$OPEN ctx=$CTX out=$OUT =="
ON_SHA=""

if in_arms on; then
  echo "--- arm on: door ON, open output $OPEN, default spec ---"
  if boot on MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS="$OPEN"; then
    for i in 1 2 3 4; do
      code=$(chat "$D/r$i"); fr=$(jget "$D/r$i.body" choices.0.finish_reason); ct=$(jget "$D/r$i.body" usage.completion_tokens)
      pt=$(jget "$D/r$i.body" usage.prompt_tokens)
      v=FAIL; [ "$code" = 200 ] && [ "$fr" = length ] && [ "$ct" = "$OPEN" ] && v=PASS
      verdict "SCE (on) r$i open request: http=$code finish_reason=$fr prompt_tokens=$pt completion_tokens=$ct expected=$OPEN -> $v"
      [ "$i" = 1 ] && ON_SHA=$(msg_sha "$D/r1.body")
    done
    code=$(chat "$D/control" 16); ct=$(jget "$D/control.body" usage.completion_tokens)
    v=FAIL; [ "$code" = 200 ] && [ "$ct" = 16 ] && v=PASS
    verdict "SCE (on) bounded control: http=$code completion_tokens=$ct expected=16 -> $v"
    census on 1
  else
    verdict "SCE (on) boot failed -> FAIL"
  fi
  stop on
fi

if in_arms plain; then
  echo "--- arm plain: door ON, open output $OPEN, MEMRA_SERVE_SPEC=0 ---"
  if boot plain MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS="$OPEN" MEMRA_SERVE_SPEC=0; then
    code=$(chat "$D/r1"); fr=$(jget "$D/r1.body" choices.0.finish_reason); ct=$(jget "$D/r1.body" usage.completion_tokens)
    sha=$(msg_sha "$D/r1.body")
    v=FAIL; [ "$code" = 200 ] && [ "$fr" = length ] && [ "$ct" = "$OPEN" ] && v=PASS
    verdict "SCE (plain) r1 open request: http=$code finish_reason=$fr completion_tokens=$ct expected=$OPEN -> $v"
    if [ -n "$ON_SHA" ]; then
      v=FAIL; [ -n "$sha" ] && [ "$sha" = "$ON_SHA" ] && v=PASS
      verdict "SCE (plain) message equals the on arm's r1: plain=${sha:0:16} spec=${ON_SHA:0:16} -> $v"
    fi
    census plain 0
  else
    verdict "SCE (plain) boot failed -> FAIL"
  fi
  stop plain
fi

if in_arms off; then
  echo "--- arm off: door OFF, MEMRA_CTX=$CTX, default spec ---"
  if boot off MEMRA_CTX="$CTX"; then
    for i in 1 2 3; do
      code=$(chat "$D/r$i"); fr=$(jget "$D/r$i.body" choices.0.finish_reason); ct=$(jget "$D/r$i.body" usage.completion_tokens)
      pt=$(jget "$D/r$i.body" usage.prompt_tokens)
      room=$(( CTX - ${pt:-0} ))
      v=FAIL; [ "$code" = 200 ] && [ "$fr" = length ] && [ -n "$ct" ] && [ "$ct" -gt 0 ] 2>/dev/null && [ "$ct" -le "$room" ] && v=PASS
      verdict "SCE (off) r$i open request to the cap: http=$code finish_reason=$fr prompt_tokens=$pt completion_tokens=$ct room=$room -> $v"
    done
    census off 1
  else
    verdict "SCE (off) boot failed -> FAIL"
  fi
  stop off
fi

echo "== spec-ctx-edge-gate: $PASSN PASS, $FAILN FAIL =="
if [ "$FAILN" = 0 ] && [ "$PASSN" -gt 0 ]; then echo "SPEC-CTX-EDGE GATE: ALL GREEN"; exit 0; fi
echo "SPEC-CTX-EDGE GATE: RED"; exit 1
