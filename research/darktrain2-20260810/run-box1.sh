#!/usr/bin/env bash
# box1 driver for darktrain phase 2. One invocation is one bounded GPU-lock block.
set -uo pipefail

BLOCK=${1:-}
case "$BLOCK" in
  allocator|qos|checkpoint|refusal) ;;
  *) echo "usage: $0 allocator|qos|checkpoint|refusal" >&2; exit 2 ;;
esac

REPO=${REPO:-$HOME/memra-cx-grouped}
BIN=${BIN:-$REPO/target/release/memra-server}
WORK_ROOT=${WORK_ROOT:-$HOME/darktrain2}
PY=${PY:-$WORK_ROOT/venv/bin/python}
TRAINER=${TRAINER:-$WORK_ROOT/harness/train_lora_seam.py}
QOS=${QOS:-$WORK_ROOT/harness/qos_probe.py}
MODEL_ROOT=${MODEL_ROOT:-$HOME/step37/models/step-3.7-flash}
MODEL=${MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
MODEL_NAME=${MODEL_NAME:-stepfun/step-3.7-flash}
PORT=${PORT:-18421}
BASE=http://127.0.0.1:$PORT
STAMP=${DARKTRAIN2_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/receipts/$STAMP}
OUT=$RUN_ROOT/$BLOCK
GOLDEN=$WORK_ROOT/golden-response.bin
# Allocator calibration measured 18,250 MiB at steady state with the 16 GiB bank.
# Declare 19 GiB so the honest-job audit has real margin instead of a 182 MiB knife edge.
BG_BUDGET_MB=${BG_BUDGET_MB:-19456}
TRAIN_RESERVE_MB=${TRAIN_RESERVE_MB:-16384}
SERVER_PID=0
SAMPLER_PID=0

mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

fail() {
  echo "FATAL: $*"
  return 1
}

snapshot() {
  local path=$1 label=$2
  {
    echo "label=$label"
    echo "ts=$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  } >"$path" 2>&1
}

compute_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
}

cleanup() {
  if (( SAMPLER_PID > 0 )); then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=0
  fi
  if (( SERVER_PID > 0 )); then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=0
  fi
}

stop_server() {
  cleanup
  for _ in $(seq 1 60); do
    [[ -z $(compute_apps 2>/dev/null) ]] && return 0
    sleep 1
  done
  compute_apps || true
  fail "GPU processes remained after server shutdown"
}

start_sampler() {
  local path=$1
  nvidia-smi \
    --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 >"$path" 2>&1 &
  SAMPLER_PID=$!
}

stop_sampler() {
  if (( SAMPLER_PID > 0 )); then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=0
  fi
}

wait_ready() {
  local log=$1
  for _ in $(seq 1 900); do
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || {
      tail -80 "$log" || true
      return 1
    }
    sleep 1
  done
  tail -80 "$log" || true
  return 1
}

metrics_field() {
  local field=$1
  curl -sf "$BASE/metrics" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); p='$field'.split('.'); v=d; [None for k in p if not (v:=v.get(k) if isinstance(v,dict) else None)]; print(v if v is not None else '')"
}

wait_bg() {
  local wanted=$1 timeout_s=${2:-120}
  local deadline=$((SECONDS + timeout_s)) state
  while (( SECONDS < deadline )); do
    state=$(metrics_field bg.state 2>/dev/null || true)
    [[ $state == "$wanted" ]] && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 0.2
  done
  echo "wait_bg wanted=$wanted got=${state:-missing}"
  return 1
}

wait_launches() {
  local wanted=$1 timeout_s=${2:-120}
  local deadline=$((SECONDS + timeout_s)) launches
  while (( SECONDS < deadline )); do
    launches=$(metrics_field bg.launches 2>/dev/null || true)
    [[ ${launches:-0} =~ ^[0-9]+$ ]] && (( launches >= wanted )) && return 0
    sleep 0.2
  done
  return 1
}

wait_train_step() {
  local state_file=$1 wanted=$2 timeout_s=${3:-180}
  local deadline=$((SECONDS + timeout_s)) step
  while (( SECONDS < deadline )); do
    step=$(python3 - "$state_file" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1]))["step"])
except Exception:
    pass
PY
)
    [[ ${step:-} =~ ^[0-9]+$ ]] && (( step >= wanted )) && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 0.2
  done
  echo "wait_train_step wanted=$wanted got=${step:-missing}"
  return 1
}

proc_state() {
  local pid=$1
  python3 - "$pid" <<'PY'
import pathlib, sys
try:
    s = pathlib.Path(f"/proc/{sys.argv[1]}/stat").read_text()
    print(s[s.rfind(")") + 1:].split()[0])
except OSError:
    print("gone")
PY
}

wait_proc_state() {
  local pid=$1 wanted=$2 timeout_s=${3:-10}
  local deadline=$((SECONDS + timeout_s)) state
  while (( SECONDS < deadline )); do
    state=$(proc_state "$pid")
    [[ $state == "$wanted" ]] && return 0
    sleep 0.01
  done
  echo "wait_proc_state pid=$pid wanted=$wanted got=${state:-missing}"
  return 1
}

trainer_command() {
  local cell=$1 alloc_conf=$2 max_steps=${3:-1000000}
  local cmd="exec env CUDA_VISIBLE_DEVICES=0"
  if [[ -n $alloc_conf ]]; then
    cmd+=" PYTORCH_CUDA_ALLOC_CONF=$alloc_conf"
  fi
  cmd+=" $PY -u $TRAINER"
  cmd+=" --events $cell/trainer-events.jsonl"
  cmd+=" --state $cell/trainer-state.json"
  cmd+=" --checkpoint $cell/trainer-checkpoint.pt"
  cmd+=" --marker $cell/trainer-launched.marker"
  cmd+=" --reserve-mb $TRAIN_RESERVE_MB --rank 16 --max-steps $max_steps --log-every 5"
  cmd+=" >>$cell/trainer.stdout.log 2>&1"
  printf '%s' "$cmd"
}

start_server() {
  local cell=$1 mode=$2 alloc_conf=${3:-} yield_mode=${4:-stop} budget=${5:-$BG_BUDGET_MB}
  local log=$cell/server.log
  local -a bg_env=()
  if [[ $mode == bg ]]; then
    local command
    command=$(trainer_command "$cell" "$alloc_conf")
    bg_env+=("MEMRA_BG_JOB=$command")
    bg_env+=("MEMRA_BG_VRAM_MB=$budget")
    bg_env+=("MEMRA_BG_YIELD_MODE=$yield_mode")
  fi
  env -u MEMRA_SERVE_SPEC -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
    MEMRA_COMPAT=openai \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_CTX=262144 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PREFILL_TICK=2048 \
    "${bg_env[@]}" \
    "$BIN" >"$log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$log" || fail "server failed readiness for $cell"
}

warmup() {
  local cell=$1
  "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "warmup-$(basename "$cell")" \
    --requests 1 --max-tokens 8 --skip-exactness \
    --rows "$cell/warmup-rows.jsonl" --summary "$cell/warmup-summary.json"
}

assert_server_clean() {
  local log=$1
  local failures
  failures=$(grep -Ein "CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died" "$log" || true)
  if [[ -n $failures ]]; then
    echo "$failures"
    fail "server failure signature in $log"
  fi
}

preflight() {
  echo "block=$BLOCK stamp=$STAMP lock_acquired=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "source_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "qos_sha256=$(sha256sum "$QOS" | awk '{print $1}')"
  echo "python=$($PY -V 2>&1)"
  "$PY" -c 'import torch; print("torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda))'
  stat -c 'artifact=%n bytes=%s' "$MODEL" "$DRAFT"
  snapshot "$OUT/nvidia-smi-before.log" preflight
  local apps
  apps=$(compute_apps 2>/dev/null || true)
  [[ -z $apps ]] || { echo "$apps"; fail "box1 was not GPU-idle at lock acquisition"; }
}

run_allocator_arm() {
  local label=$1 alloc_conf=$2
  local cell=$OUT/$label
  mkdir -p "$cell"
  echo "allocator_cell=$label alloc_conf=${alloc_conf:-unset} start=$(date -u +%FT%TZ)"
  start_server "$cell" bg "$alloc_conf" stop "$BG_BUDGET_MB" || return 1
  snapshot "$cell/serve-resident-before-warmup.log" serve-resident
  warmup "$cell" || return 1
  wait_bg running 180 || { tail -100 "$cell/server.log"; return 1; }
  wait_train_step "$cell/trainer-state.json" 15 180 || {
    tail -100 "$cell/trainer.stdout.log" || true
    return 1
  }
  snapshot "$cell/training-running.log" training-running
  curl -sf "$BASE/metrics" >"$cell/metrics.json"
  sleep 3
  stop_server || return 1
  assert_server_clean "$cell/server.log" || return 1
  echo "allocator_cell=$label done=$(date -u +%FT%TZ)"
}

run_allocator() {
  run_allocator_arm default "" || return 1
  run_allocator_arm maxsplit "max_split_size_mb:128" || return 1
}

manual_park() {
  local cell=$1 pid=$2
  local start_ns end_ns state
  start_ns=$(date +%s%N)
  kill -STOP -- "-$pid"
  wait_proc_state "$pid" T 10 || return 1
  end_ns=$(date +%s%N)
  state=$(proc_state "$pid")
  python3 - "$pid" "$start_ns" "$end_ns" "$state" >"$cell/manual-park.json" <<'PY'
import json, sys
print(json.dumps({"pid": int(sys.argv[1]),
                  "sigstop_to_T_ms": (int(sys.argv[3])-int(sys.argv[2]))/1e6,
                  "state": sys.argv[4]}, sort_keys=True))
PY
}

run_qos_cell() {
  local arm=$1 rep=$2 create_golden=${3:-0}
  local cell=$OUT/rep${rep}-${arm}
  mkdir -p "$cell"
  echo "qos_cell=rep${rep}-${arm} start=$(date -u +%FT%TZ)"
  if [[ $arm == absent ]]; then
    start_server "$cell" absent || return 1
  else
    start_server "$cell" bg "" stop "$BG_BUDGET_MB" || return 1
  fi
  snapshot "$cell/serve-resident-before-warmup.log" serve-resident
  warmup "$cell" || return 1
  local pid=0 step_before=0
  if [[ $arm == absent ]]; then
    sleep 3
  else
    wait_bg running 180 || { tail -100 "$cell/server.log"; return 1; }
    wait_train_step "$cell/trainer-state.json" 10 180 || {
      tail -100 "$cell/trainer.stdout.log" || true
      return 1
    }
    pid=$(metrics_field bg.job_pid)
    [[ $pid =~ ^[0-9]+$ ]] || fail "missing trainer pid for $cell" || return 1
    step_before=$(python3 -c "import json; print(json.load(open('$cell/trainer-state.json'))['step'])")
    snapshot "$cell/training-running-before.log" training-running-before
    if [[ $arm == parked ]]; then
      manual_park "$cell" "$pid" || return 1
      snapshot "$cell/training-manually-parked.log" training-manually-parked
    fi
  fi
  start_sampler "$cell/gpu.csv"
  local -a golden_args=(--golden "$GOLDEN")
  (( create_golden == 1 )) && golden_args+=(--create-golden)
  local -a watch_args=()
  (( pid > 0 )) && watch_args+=(--watch-pid "$pid")
  "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "rep${rep}-${arm}" \
    --requests 8 --max-tokens 64 \
    --rows "$cell/qos-rows.jsonl" --summary "$cell/qos-summary.json" \
    "${golden_args[@]}" "${watch_args[@]}"
  local probe_rc=$?
  stop_sampler
  (( probe_rc == 86 )) && { echo "P0: served bytes changed in $cell"; return 86; }
  (( probe_rc == 0 )) || return "$probe_rc"
  curl -sf "$BASE/metrics" >"$cell/metrics-after.json"
  snapshot "$cell/after-qos.log" after-qos
  if [[ $arm != absent ]]; then
    [[ $(proc_state "$pid") == T ]] || fail "trainer not SIGSTOPped after $arm QoS" || return 1
    [[ $(metrics_field bg.state) == yielded ]] || fail "runner did not publish yielded" || return 1
    sleep 3
    wait_bg running 30 || return 1
    if [[ $(proc_state "$pid") == T ]]; then
      fail "trainer remained stopped after valley"
      return 1
    fi
    sleep 1
    local step_after
    step_after=$(python3 -c "import json; print(json.load(open('$cell/trainer-state.json'))['step'])")
    python3 - "$step_before" "$step_after" >"$cell/resume-progress.json" <<'PY'
import json, sys
before, after = map(int, sys.argv[1:])
print(json.dumps({"step_before": before, "step_after": after,
                  "advanced": after > before}, sort_keys=True))
raise SystemExit(0 if after > before else 1)
PY
    snapshot "$cell/training-resumed.log" training-resumed
  fi
  stop_server || return 1
  assert_server_clean "$cell/server.log" || return 1
  echo "qos_cell=rep${rep}-${arm} done=$(date -u +%FT%TZ)"
}

run_qos() {
  rm -f "$GOLDEN"
  run_qos_cell absent 1 1 || return $?
  run_qos_cell running 1 || return $?
  run_qos_cell parked 1 || return $?
  run_qos_cell running 2 || return $?
  run_qos_cell parked 2 || return $?
  run_qos_cell absent 2 || return $?
  run_qos_cell parked 3 || return $?
  run_qos_cell absent 3 || return $?
  run_qos_cell running 3 || return $?
  sha256sum "$GOLDEN" >"$OUT/golden-response.sha256"
}

last_checkpoint_step() {
  local events=$1
  python3 - "$events" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
saved = [row for row in rows if row.get("event") == "checkpoint_saved"]
print(saved[-1]["step"] if saved else "")
PY
}

run_checkpoint() {
  [[ -f $GOLDEN ]] || fail "QoS golden missing; run qos block first" || return 1
  local cell=$OUT/checkpoint
  mkdir -p "$cell"
  start_server "$cell" bg "" checkpoint "$BG_BUDGET_MB" || return 1
  warmup "$cell" || return 1
  wait_bg running 180 || return 1
  wait_train_step "$cell/trainer-state.json" 10 180 || return 1
  local pid step_before
  pid=$(metrics_field bg.job_pid)
  step_before=$(python3 -c "import json; print(json.load(open('$cell/trainer-state.json'))['step'])")
  snapshot "$cell/training-running-before.log" checkpoint-running
  "$QOS" --base "$BASE" --model "$MODEL_NAME" --label checkpoint-preempt \
    --requests 1 --max-tokens 64 --watch-pid "$pid" --golden "$GOLDEN" \
    --rows "$cell/qos-rows.jsonl" --summary "$cell/qos-summary.json"
  local probe_rc=$?
  (( probe_rc == 86 )) && { echo "P0: served bytes changed in checkpoint cell"; return 86; }
  (( probe_rc == 0 )) || return "$probe_rc"
  wait_bg preempted 30 || return 1
  snapshot "$cell/after-checkpoint-exit.log" checkpoint-process-exited
  curl -sf "$BASE/metrics" >"$cell/metrics-preempted.json"
  [[ $(proc_state "$pid") == gone ]] || fail "checkpointed trainer pid still exists" || return 1
  [[ -f $cell/trainer-checkpoint.pt ]] || fail "checkpoint file missing" || return 1
  sha256sum "$cell/trainer-checkpoint.pt" >"$cell/trainer-checkpoint.sha256"
  local saved_step
  saved_step=$(last_checkpoint_step "$cell/trainer-events.jsonl")
  [[ $saved_step =~ ^[0-9]+$ ]] || fail "no checkpoint_saved event" || return 1
  wait_launches 2 60 || return 1
  wait_bg running 60 || return 1
  wait_train_step "$cell/trainer-state.json" $((saved_step + 5)) 120 || return 1
  local resumed_pid step_after
  resumed_pid=$(metrics_field bg.job_pid)
  step_after=$(python3 -c "import json; print(json.load(open('$cell/trainer-state.json'))['step'])")
  python3 - "$step_before" "$saved_step" "$step_after" "$pid" "$resumed_pid" \
    >"$cell/checkpoint-resume.json" <<'PY'
import json, sys
before, saved, after, old_pid, new_pid = map(int, sys.argv[1:])
row = {"step_before_signal": before, "checkpoint_step": saved,
       "resumed_observed_step": after, "old_pid": old_pid, "new_pid": new_pid,
       "new_process": new_pid != old_pid, "resumed_past_checkpoint": after > saved}
print(json.dumps(row, sort_keys=True))
raise SystemExit(0 if row["new_process"] and row["resumed_past_checkpoint"] else 1)
PY
  snapshot "$cell/training-resumed.log" checkpoint-resumed
  curl -sf "$BASE/metrics" >"$cell/metrics-resumed.json"
  grep -q '"event": "checkpoint_loaded"' "$cell/trainer-events.jsonl" || {
    fail "relaunch did not log checkpoint_loaded"; return 1;
  }
  stop_server || return 1
  assert_server_clean "$cell/server.log" || return 1
}

run_refusal() {
  local baseline=$OUT/baseline-free refusal=$OUT/refusal
  mkdir -p "$baseline" "$refusal"
  start_server "$baseline" absent || return 1
  warmup "$baseline" || return 1
  sleep 3
  snapshot "$baseline/serve-resident.log" refusal-baseline
  local free_min
  free_min=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  echo "$free_min" >"$baseline/min-free-mib.txt"
  stop_server || return 1
  local refused_budget=$((free_min + 1024))
  echo "$refused_budget" >"$refusal/requested-budget-mib.txt"
  start_server "$refusal" bg "" stop "$refused_budget" || return 1
  sleep 4
  curl -sf "$BASE/metrics" >"$refusal/metrics.json"
  snapshot "$refusal/serve-resident.log" refusal-arm
  local state launches
  state=$(metrics_field bg.state)
  launches=$(metrics_field bg.launches)
  [[ $state == refused_vram ]] || fail "want refused_vram, got $state" || return 1
  [[ ${launches:-0} == 0 ]] || fail "refused trainer launch count=$launches" || return 1
  [[ ! -e $refusal/trainer-launched.marker ]] || fail "refused trainer actually launched" || return 1
  grep -q '\[darklane\] REFUSED' "$refusal/server.log" || fail "missing loud REFUSED log" || return 1
  stop_server || return 1
  assert_server_clean "$refusal/server.log" || return 1
}

run_locked() {
  trap cleanup EXIT INT TERM
  preflight || return 1
  case "$BLOCK" in
    allocator) run_allocator ;;
    qos) run_qos ;;
    checkpoint) run_checkpoint ;;
    refusal) run_refusal ;;
  esac
  local rc=$?
  cleanup
  snapshot "$OUT/nvidia-smi-after.log" final
  echo "block=$BLOCK rc=$rc lock_released=$(date -u +%FT%TZ)"
  return "$rc"
}

(
  flock -w 60 9 || { echo "LOCK_TIMEOUT"; exit 75; }
  run_locked
) 9>/tmp/memra-gpu.lock
