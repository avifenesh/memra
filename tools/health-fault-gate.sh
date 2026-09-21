#!/usr/bin/env bash
# tools/health-fault-gate.sh: serving-shape health, readiness and lifecycle gate (memra#524; the
# fault/lifecycle half of memra#526). Boots the real memra-server on the real model, injects the
# faults the server already has doors for, and asserts HTTP codes plus body fields on `/readyz`,
# `/health` (== `/livez`; there is no `/healthz` route) and the request path. Every verdict is one
# line, verbatim, in `$OUT/VERDICTS.txt`; every arm keeps its server log, its readiness samples
# and `nvidia-smi --query-compute-apps` before and after.
#
# Arms (pre-registered research/spill-b-20260919/DAY24.md section 5, restated by the lead for
# day 25):
#   a  readiness before the boot calibration probe completes reads not-ready. On a FIRST boot the
#      listener binds after the probe (docs/SERVING.md, phase table), so every pre-ready sample is
#      connection-refused (code 000); the arm asserts the log order (`[admit-cal] boot calibration
#      done:` before `[server] listening on`) and that no sample ever read 200 before that order
#      held. The probe window is observable over HTTP during a RESPAWN: arm d records the phase
#      sequence there (verdict line `a2`, `loading` then `warming`, memra#524 phase=warming).
#   b  readiness after the probe reads ready and the first request completes (200, finish_reason,
#      completion_tokens > 0), `/readyz` 200 before and after.
#   c  the probe-skipped boots (MEMRA_ADMIT_CALIBRATE=0, MEMRA_SERVE_SPEC=0, MEMRA_ADMIT_RESERVE_MB)
#      state what readiness means there: ready WITHOUT warmup, never `warming`. Recorded as the
#      DOCUMENTED behaviour of those doors, not as a pass.
#   d  a request that panics the worker (MEMRA_PANIC_AFTER=1, the one-shot fault door with its
#      FLAGS row; MEMRA_WORKER_RESPAWN=1) leaves `/health` truthful: 503 with the quoted payload
#      within seconds, never 200 while the worker is dead or reloading, 200 with generation 1 after
#      the respawn, and a request completes on the respawned worker.
#   e  a fatal fault of the gpu-watch class (memra#516) is not cleared by a timeout-only recovery:
#      a PATH-shadowed `nvidia-smi` (server process only) hangs ONE steady-state probe past
#      MEMRA_GPU_PROBE_TIMEOUT_S, the latch fires (`/health` 503, `[gpu-watch] CRITICAL`), the shim
#      answers again, and three probe intervals later the latch still holds. No new door: the shim
#      is the injection and MEMRA_GPU_WATCH_S / MEMRA_GPU_PROBE_TIMEOUT_S are documented knobs.
#   f  a graceful shutdown flips readiness first and drains: SIGTERM with a stream open ->
#      `/readyz` 503 `draining` + Retry-After, `/health` 200 `draining`, a new request 503 with
#      `code: draining`, the stream runs to `[DONE]` with a finish_reason, exit 0, `drain complete`.
#
# Usage: tools/health-fault-gate.sh [model.gguf]
#   HFG_PORT (8189; 8186 is serve-gemma4-batch-gate.sh's, revuto on #621; census of tools/ before choosing a default), HFG_ARMS (a,b,c,d,e,f; a and b share one boot), HFG_OUT (receipt dir).
#   Run under the rig lock (`flock /tmp/memra-5090.lock`, or the collector on a PRO box); the
#   gate boots seven servers in sequence and never takes the lock itself, like serve-smoke.
#   Exit 0 when no arm FAILED (DOCUMENTED arms do not fail the gate); 1 on any FAIL; 2 on setup.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "health-fault-gate: SKIP (no model at $MODEL)"; exit 0; }
PORT="${HFG_PORT:-8189}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
ARMS="${HFG_ARMS:-a,b,c,d,e,f}"
OUT="${HFG_OUT:-/tmp/health-fault-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
VERDICTS="$OUT/VERDICTS.txt"
: > "$VERDICTS"
NVSMI="$(command -v nvidia-smi || true)"
[ -n "$NVSMI" ] || { echo "health-fault-gate: FATAL: nvidia-smi not on PATH (arm e needs the real tool to shadow)"; exit 2; }
READY_WAIT_S="${HFG_READY_WAIT_S:-420}"

. tools/port-guard.sh
memra_port_guard health-fault-gate "$PORT" HFG_PORT || exit 2

# Build UNCONDITIONALLY (serve-smoke's rule: a gate that can run a stale binary is a rotted gate).
cargo build --release -p memra-server || { echo "health-fault-gate: build FAILED"; exit 2; }
BIN=target/release/memra-server
sha256sum "$BIN" > "$OUT/binary.sha256"
{ echo "commit $(git rev-parse HEAD)"; echo "model $MODEL"; "$NVSMI" --query-gpu=name,driver_version,power.limit --format=csv,noheader; date -u +%FT%TZ; } > "$OUT/source.txt"

PASSN=0; DOCN=0; FAILN=0
verdict() { # <line ending in -> PASS|DOCUMENTED|FAIL>
  echo "$1" | tee -a "$VERDICTS"
  case "$1" in
    *"-> PASS") PASSN=$((PASSN+1));;
    *"-> DOCUMENTED"*) DOCN=$((DOCN+1));;
    *) FAILN=$((FAILN+1));;
  esac
}
now_ms() { echo $(( ${EPOCHREALTIME/./} / 1000 )); }  # bash 5; `date +%s%3N` prints nanoseconds with this date(1)
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
gpu_apps() { "$NVSMI" --query-compute-apps=pid,process_name,used_memory --format=csv > "$1" 2>&1; }

SPID=0; D=""; T_LAUNCH=0; T_READY=0
# sample <route> <samples.csv>: one HTTP read, appended as t_ms,code,status,phase,detail
sample() {
  local body="$D/.sample.$$" code
  code=$(curl -s -o "$body" -w '%{http_code}' -m 2 "$BASE$1" 2>/dev/null) || code=000
  [ -n "$code" ] || code=000
  local status phase detail
  if [ -s "$body" ]; then
    status=$(jget "$body" status); phase=$(jget "$body" worker.phase); detail=$(jget "$body" detail)
  else status=""; phase=""; detail=""; fi
  echo "$(( $(now_ms) - T_LAUNCH )),$code,$status,$phase,\"${detail//\"/\'}\"" >> "$2"
  LAST_CODE=$code; LAST_STATUS=$status; LAST_PHASE=$phase; LAST_DETAIL=$detail
}
# boot <label> [ENV=VAL ...]: launch, poll /readyz at 100 ms until 200, record every sample.
boot() {
  local label=$1; shift
  D="$OUT/$label"; mkdir -p "$D"
  gpu_apps "$D/compute-apps-before.csv"
  T_LAUNCH=$(now_ms)
  env MEMRA_COMPAT=openai MEMRA_MODELS="hfg=$MODEL" MEMRA_ADDR=$ADDR "$@" "$BIN" > "$D/server.log" 2>&1 &
  SPID=$!
  echo "$*" > "$D/env.txt"
  : > "$D/readyz-samples.csv"
  local deadline=$(( T_LAUNCH + READY_WAIT_S * 1000 ))
  while :; do
    sample /readyz "$D/readyz-samples.csv"
    if [ "$LAST_CODE" = "200" ]; then T_READY=$(now_ms); break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "  [$label] server died during boot; log tail:"; tail -5 "$D/server.log"; return 1; fi
    if [ "$(now_ms)" -gt "$deadline" ]; then echo "  [$label] FATAL: not ready within ${READY_WAIT_S}s"; return 1; fi
    sleep 0.1
  done
  memra_port_owned health-fault-gate "$PORT" "$SPID" || return 1
  echo "  [$label] ready after $(( T_READY - T_LAUNCH )) ms (pid $SPID)"
}
# stop: SIGTERM only a memra-server this gate started (pid AND cwd), record its exit code.
stop() {
  local label=$1 rc
  if kill -0 "$SPID" 2>/dev/null; then
    if [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ]; then kill -TERM "$SPID" 2>/dev/null
    else echo "  [$label] REFUSING to signal pid $SPID: cwd is not this tree"; fi
  fi
  wait "$SPID" 2>/dev/null; rc=$?
  echo "$rc" > "$D/exit"
  echo "  [$label] exit $rc"
  gpu_apps "$D/compute-apps-after.csv"
  rm -f "$D/.sample.$$"
  SPID=0
}
cleanup() { [ "$SPID" != 0 ] && kill -0 "$SPID" 2>/dev/null && [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ] && kill -TERM "$SPID" 2>/dev/null; true; }
trap cleanup EXIT

# chat <max_tokens> <stream:true|false> <outprefix> [timeout_s]: writes .hdr .body, prints the code
chat() {
  local code
  code=$(curl -s -m "${4:-180}" -D "$3.hdr" -o "$3.body" -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"hfg\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about a mutex.\"}],\"max_tokens\":$1,\"temperature\":0,\"stream\":$2}") || code=000
  echo "${code:-000}"
}
completes() { # <prefix>: non-stream body has finish_reason in (stop,length) and completion_tokens>0
  local fr ct
  fr=$(jget "$1.body" choices.0.finish_reason 2>/dev/null); ct=$(jget "$1.body" usage.completion_tokens)
  case "$fr" in stop|length) [ -n "$ct" ] && [ "$ct" -gt 0 ] 2>/dev/null;; *) false;; esac
}
in_arms() { case ",$ARMS," in *",$1,"*) return 0;; *) return 1;; esac; }
lineno() { grep -n -m1 -F -- "$1" "$2" | cut -d: -f1; }

echo "== health-fault-gate: arms=$ARMS port=$PORT out=$OUT =="

# ---------------------------------------------------------------- a + b: armed boot
if in_arms a || in_arms b; then
  echo "--- arm a/b: armed boot (default doors: the calibration probe runs before ready) ---"
  if boot ab; then
    if in_arms a; then
      pre=$(awk -F, '$2!=200' "$D/readyz-samples.csv" | wc -l)
      c000=$(awk -F, '$2==000' "$D/readyz-samples.csv" | wc -l)
      c503=$(awk -F, '$2==503' "$D/readyz-samples.csv" | wc -l)
      early200=$(awk -F, '$2==200 && $3!="ready"' "$D/readyz-samples.csv" | wc -l)
      warm200=$(awk -F, '$2==200 && $4=="warming"' "$D/readyz-samples.csv" | wc -l)
      probe_ln=$(lineno '[admit-cal] boot calibration done:' "$D/server.log"); probe_ln=${probe_ln:-0}
      listen_ln=$(lineno '[server] listening on' "$D/server.log"); listen_ln=${listen_ln:-0}
      first_phase=$(awk -F, '$2==200{print $4; exit}' "$D/readyz-samples.csv")
      v=FAIL
      [ "$probe_ln" -gt 0 ] && [ "$listen_ln" -gt "$probe_ln" ] && [ "$early200" = 0 ] && [ "$warm200" = 0 ] && [ "$pre" -gt 0 ] && v=PASS
      verdict "HFG (a) readiness-before-probe: probe_done_line=$probe_ln listening_line=$listen_ln pre_ready_samples=$pre pre_ready_codes={000:$c000,503:$c503} ready_samples_not_ready=$early200 ready_while_warming=$warm200 first_ready_phase=$first_phase -> $v"
    fi
    if in_arms b; then
      sample /readyz "$D/readyz-b.csv"; before=$LAST_CODE/$LAST_STATUS
      t0=$(now_ms); code=$(chat 32 false "$D/first"); t1=$(now_ms)
      sample /readyz "$D/readyz-b.csv"; after=$LAST_CODE/$LAST_STATUS
      fr=$(jget "$D/first.body" choices.0.finish_reason); ct=$(jget "$D/first.body" usage.completion_tokens)
      v=FAIL
      [ "$before" = "200/ready" ] && [ "$code" = 200 ] && completes "$D/first" && [ "$after" = "200/ready" ] && v=PASS
      verdict "HFG (b) ready-then-first-request: readyz_before=$before request_http=$code finish_reason=$fr completion_tokens=$ct request_ms=$((t1-t0)) ready_to_completion_ms=$((t1-T_READY)) readyz_after=$after (N=1, not a timing claim) -> $v"
    fi
    stop ab
  else
    in_arms a && verdict "HFG (a) readiness-before-probe: boot failed -> FAIL"
    in_arms b && verdict "HFG (b) ready-then-first-request: boot failed -> FAIL"
    stop ab
  fi
fi

# ---------------------------------------------------------------- c: probe-skipped boots
if in_arms c; then
  echo "--- arm c: the probe-skipped boots (documented behaviour, not a pass) ---"
  run_c() { # <label> <expected skip line> ENV=VAL
    local label=$1 skip=$2; shift 2
    if boot "$label" "$@"; then
      skip_ln=$(lineno "$skip" "$D/server.log"); skip_ln=${skip_ln:-0}
      done_ln=$(lineno '[admit-cal] boot calibration done:' "$D/server.log"); done_ln=${done_ln:-0}
      warm=$(awk -F, '$4=="warming"' "$D/readyz-samples.csv" | wc -l)
      t0=$(now_ms); code=$(chat 32 false "$D/first"); t1=$(now_ms)
      v=FAIL
      [ "$skip_ln" -gt 0 ] && [ "$done_ln" = 0 ] && [ "$warm" = 0 ] && [ "$code" = 200 ] && completes "$D/first" && v="DOCUMENTED (ready without warmup; the first request pays the cold route)"
      verdict "HFG (c) probe-skipped $label [$*]: skip_line=$skip_ln probe_done_line=$done_ln warming_samples=$warm ready_ms=$((T_READY-T_LAUNCH)) first_request_http=$code first_request_ms=$((t1-t0)) (N=1) -> $v"
      stop "$label"
    else
      verdict "HFG (c) probe-skipped $label [$*]: boot failed -> FAIL"; stop "$label"
    fi
  }
  run_c c-calibrate0 'boot calibration disarmed (MEMRA_ADMIT_CALIBRATE=0)' MEMRA_ADMIT_CALIBRATE=0
  run_c c-spec0 'boot calibration skipped: spec serving disabled' MEMRA_SERVE_SPEC=0
  run_c c-reserve 'boot calibration skipped: MEMRA_ADMIT_RESERVE_MB teeth door' MEMRA_ADMIT_RESERVE_MB=1536
fi

# ---------------------------------------------------------------- d: worker panic, respawn
if in_arms d; then
  echo "--- arm d: a request panics the worker (MEMRA_PANIC_AFTER=1), the respawn rule (MEMRA_WORKER_RESPAWN=1) ---"
  if boot d MEMRA_PANIC_AFTER=1 MEMRA_WORKER_RESPAWN=1; then
    : > "$D/health-samples.csv"
    trig=$(chat 8 false "$D/trigger" 60)
    t_trig=$(now_ms)
    # 1. the flip: /health 503 with the quoted payload, within 10 s
    flip_ms=""; while [ $(( $(now_ms) - t_trig )) -lt 10000 ]; do
      sample /health "$D/health-samples.csv"
      if [ "$LAST_CODE" = 503 ] && [[ "$LAST_DETAIL" == *"worker thread panicked"* ]]; then flip_ms=$(( $(now_ms) - t_trig )); break; fi
      sleep 0.1
    done
    sample /readyz "$D/readyz-d.csv"; readyz_dead="$LAST_CODE/$LAST_STATUS/$LAST_PHASE"
    # 2. recovery: poll /readyz at 100 ms, record the phase sequence of the not-ready window
    : > "$D/readyz-recovery.csv"; t_rec0=$(now_ms); rec_ms=""
    while [ $(( $(now_ms) - t_rec0 )) -lt $(( READY_WAIT_S * 1000 )) ]; do
      sample /readyz "$D/readyz-recovery.csv"
      if [ "$LAST_CODE" = 200 ]; then rec_ms=$(( $(now_ms) - t_trig )); break; fi
      kill -0 "$SPID" 2>/dev/null || break
      sleep 0.1
    done
    phases=$(awk -F, '$2!=200{print $4}' "$D/readyz-recovery.csv" | uniq | paste -sd'>' -)
    n_load=$(awk -F, '$4=="loading"' "$D/readyz-recovery.csv" | wc -l)
    n_warm=$(awk -F, '$4=="warming"' "$D/readyz-recovery.csv" | wc -l)
    last_load=$(awk -F, '$4=="loading"{n=NR} END{print n+0}' "$D/readyz-recovery.csv")
    first_warm=$(awk -F, '$4=="warming"{print NR; exit}' "$D/readyz-recovery.csv"); first_warm=${first_warm:-0}
    untruthful=$(cat "$D/health-samples.csv" "$D/readyz-recovery.csv" | awk -F, '($2==200 && $3!="ok" && $3!="ready") || ($2!=200 && ($3=="ok" || $3=="ready"))' | wc -l)
    gen=$(curl -s -m 5 "$BASE/readyz" -o "$D/readyz-after.json" -w '' && jget "$D/readyz-after.json" worker.generation)
    after=$(chat 32 false "$D/after" 120)
    sample /health "$D/health-samples.csv"; health_after="$LAST_CODE/$LAST_STATUS"
    panic_ln=$(lineno '[worker] PANIC in the GPU worker thread' "$D/server.log"); panic_ln=${panic_ln:-0}
    respawn_ln=$(lineno '[worker] respawn attempt 1/1' "$D/server.log"); respawn_ln=${respawn_ln:-0}
    v=FAIL
    [ -n "$flip_ms" ] && [[ "$readyz_dead" == 503/* ]] && [ -n "$rec_ms" ] && [ "$gen" = 1 ] && [ "$after" = 200 ] && completes "$D/after" && [ "$health_after" = "200/ok" ] && [ "$untruthful" = 0 ] && [ "$panic_ln" -gt 0 ] && [ "$respawn_ln" -gt "$panic_ln" ] && v=PASS
    verdict "HFG (d) panic-respawn-truthful-health: trigger_http=$trig health_503_quoted_after_ms=${flip_ms:-none} readyz_while_dead=$readyz_dead untruthful_samples=$untruthful recovered_after_ms=${rec_ms:-none} generation_after=$gen request_after_http=$after health_after=$health_after log_panic_line=$panic_ln log_respawn_line=$respawn_ln -> $v"
    v2=FAIL
    [ "$n_load" -gt 0 ] && [ "$n_warm" -gt 0 ] && [ "$first_warm" -gt "$last_load" ] && v2=PASS
    verdict "HFG (a2) warming-phase-on-respawn: not_ready_phase_sequence=$phases loading_samples=$n_load warming_samples=$n_warm loading_then_warming=$([ "$first_warm" -gt "$last_load" ] && echo true || echo false) -> $v2"
    stop d
  else
    verdict "HFG (d) panic-respawn-truthful-health: boot failed -> FAIL"; verdict "HFG (a2) warming-phase-on-respawn: boot failed -> FAIL"; stop d
  fi
fi

# ---------------------------------------------------------------- e: fatal gpu-watch fault latches
if in_arms e; then
  echo "--- arm e: a hung nvidia-smi probe latches the gpu fault; answering again does not clear it ---"
  SHIM="$OUT/e-shim"; mkdir -p "$SHIM"; HANG="$SHIM/hang"; rm -f "$HANG"
  cat > "$SHIM/nvidia-smi" <<SH
#!/usr/bin/env bash
# health-fault-gate arm e: shadows nvidia-smi for the server under test only. While the flag
# file exists the probe hangs (exec, so the deadline kill lands on the sleep itself).
[ -e "$HANG" ] && exec sleep 600
exec "$NVSMI" "\$@"
SH
  chmod +x "$SHIM/nvidia-smi"
  if boot e PATH="$SHIM:$PATH" MEMRA_GPU_WATCH_S=2 MEMRA_GPU_PROBE_TIMEOUT_S=2; then
    : > "$D/health-samples.csv"
    on_ln=$(lineno '[gpu-watch] on: every 2s, probe deadline 2s' "$D/server.log"); on_ln=${on_ln:-0}
    ctrl_ok=0; for _ in 1 2 3; do sample /health "$D/health-samples.csv"; [ "$LAST_CODE/$LAST_STATUS" = "200/ok" ] && ctrl_ok=$((ctrl_ok+1)); sleep 2; done
    touch "$HANG"; t_hang=$(now_ms); latch_ms=""
    while [ $(( $(now_ms) - t_hang )) -lt 20000 ]; do
      sample /health "$D/health-samples.csv"
      if [ "$LAST_CODE" = 503 ] && [[ "$LAST_DETAIL" == *"nvidia-smi did not answer within 2s"* ]]; then latch_ms=$(( $(now_ms) - t_hang )); break; fi
      sleep 0.2
    done
    rm -f "$HANG"; t_clear=$(now_ms)
    sleep 8   # >= 3 probe intervals with the shim answering again
    sample /health "$D/health-samples.csv"; health_late="$LAST_CODE/$LAST_STATUS"; detail_late="$LAST_DETAIL"
    sample /readyz "$D/readyz-e.csv"; readyz_late="$LAST_CODE/$LAST_STATUS"
    req=$(chat 8 false "$D/req" 60)
    crit=$(grep -c -F '[gpu-watch] CRITICAL' "$D/server.log")
    late_ok=$(awk -F, -v t=$((t_clear - T_LAUNCH)) '$1>t && $2==200' "$D/health-samples.csv" | wc -l)
    v=FAIL
    [ "$on_ln" -gt 0 ] && [ "$ctrl_ok" = 3 ] && [ -n "$latch_ms" ] && [ "$health_late" = "503/unhealthy" ] && [[ "$detail_late" == *"did not answer"* ]] && [[ "$readyz_late" == 503/* ]] && [ "$crit" = 1 ] && [ "$late_ok" = 0 ] && v=PASS
    verdict "HFG (e) fatal-fault-not-cleared-by-timeout: watch_on_line=$on_ln control_200_samples=$ctrl_ok/3 latched_after_ms=${latch_ms:-none} shim_answering_again_for_ms=$(( $(now_ms) - t_clear )) health_after=$health_late readyz_after=$readyz_late health_200_after_clear=$late_ok critical_lines=$crit request_http_while_latched=$req (observed, bounded, not asserted) -> $v"
    stop e
  else
    verdict "HFG (e) fatal-fault-not-cleared-by-timeout: boot failed -> FAIL"; stop e
  fi
fi

# ---------------------------------------------------------------- f: graceful drain
if in_arms f; then
  echo "--- arm f: SIGTERM with a stream open flips readiness first and drains ---"
  if boot f MEMRA_DRAIN_S=60; then
    : > "$D/stream.body"
    curl -s -N -m 170 -D "$D/stream.hdr" -o "$D/stream.body" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"hfg","messages":[{"role":"user","content":"Write a long story about a lighthouse keeper."}],"max_tokens":256,"temperature":0,"stream":true}' &
    CPID=$!
    started=false; t_s=$(now_ms)
    while [ $(( $(now_ms) - t_s )) -lt 30000 ]; do
      [ "$(grep -c '^data: ' "$D/stream.body" 2>/dev/null)" -ge 2 ] 2>/dev/null && { started=true; break; }
      sleep 0.05
    done
    frames_at_term=$(grep -c '^data: ' "$D/stream.body")
    kill -TERM "$SPID"; t_term=$(now_ms)
    sample /readyz "$D/readyz-f.csv"; readyz_drain="$LAST_CODE/$LAST_STATUS"; readyz_detail="$LAST_DETAIL"
    ra=$(curl -s -m 2 -o /dev/null -D - "$BASE/readyz" | grep -i '^retry-after:' | tr -d '\r' | awk '{print $2}')
    sample /health "$D/health-f.csv"; health_drain="$LAST_CODE/$LAST_STATUS"
    newreq=$(chat 8 false "$D/during-drain" 20)
    new_code=$(jget "$D/during-drain.body" error.code)
    new_ra=$(grep -i '^retry-after:' "$D/during-drain.hdr" | tr -d '\r' | awk '{print $2}')
    wait "$CPID"; crc=$?
    t_stream_done=$(now_ms)
    frames=$(grep -c '^data: ' "$D/stream.body"); done_seen=$(grep -c '^data: \[DONE\]' "$D/stream.body")
    fr=$(grep '^data: {' "$D/stream.body" | sed 's/^data: //' | python3 -c '
import json,sys
fr=""
for l in sys.stdin:
    try:
        for c in json.loads(l).get("choices",[]):
            if c.get("finish_reason"): fr=c["finish_reason"]
    except Exception: pass
print(fr)')
    wait "$SPID" 2>/dev/null; rc=$?; echo "$rc" > "$D/exit"; SPID=0
    t_exit=$(now_ms)
    gpu_apps "$D/compute-apps-after.csv"
    rm -f "$D/.sample.$$"
    drain_ln=$(lineno '[server] drain complete in' "$D/server.log"); drain_ln=${drain_ln:-0}
    deadline_ln=$(lineno 'drain deadline' "$D/server.log"); deadline_ln=${deadline_ln:-0}
    v=FAIL
    [ "$started" = true ] && [ "$readyz_drain" = "503/not_ready" ] && [[ "$readyz_detail" == *draining* ]] && [ -n "$ra" ] && [ "$health_drain" = "200/draining" ] && [ "$newreq" = 503 ] && [ "$new_code" = draining ] && [ -n "$new_ra" ] && [ "$crc" = 0 ] && [ "$done_seen" = 1 ] && [ -n "$fr" ] && [ "$rc" = 0 ] && [ "$drain_ln" -gt 0 ] && [ "$deadline_ln" = 0 ] && v=PASS
    verdict "HFG (f) sigterm-drains-and-flips-readiness-first: stream_frames_at_sigterm=$frames_at_term readyz_during_drain=$readyz_drain retry_after_s=${ra:-none} health_during_drain=$health_drain new_request_http=$newreq new_request_code=$new_code new_request_retry_after_s=${new_ra:-none} stream_curl_rc=$crc stream_frames=$frames stream_done=$done_seen stream_finish_reason=$fr stream_finished_after_sigterm_ms=$((t_stream_done-t_term)) exit_code=$rc exit_after_sigterm_ms=$((t_exit-t_term)) drain_complete_line=$drain_ln deadline_hit_line=$deadline_ln -> $v"
  else
    verdict "HFG (f) sigterm-drains-and-flips-readiness-first: boot failed -> FAIL"; stop f
  fi
fi

echo "health-fault-gate: arms=$ARMS pass=$PASSN documented=$DOCN fail=$FAILN receipts=$OUT" | tee -a "$VERDICTS"
[ "$FAILN" = 0 ]
