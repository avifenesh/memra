# shellcheck shell=bash
# Shared runner code for the kv-vmm Max-Q cell (DAY37 O1 replay at the r4 tree). Sourced by run.sh and run-timed.sh.
# Nothing here starts GPU work; the chains (chain-untimed.sh, chain-timed.sh) do, under the runner's lock hold.

K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$K/harness
MODEL=$(cd "$K/.." && pwd)/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
LANE_SHA=c6f9282c26e71349c0263b88dc7beb59c4a57797
MAIN_SHA=17dceb9817d050d0960a8e2d0be372921bfbee80
HARNESS_SHA=3476ea4dca27c4c3c738017b8429f507389f1223
RIG_LOCK=/tmp/memra-gpu.lock
EVALS=${KIT_EVALS_DIR:-/mnt/Data/work/evals}
CARD_LABEL=pro6000-maxq
# serve-smoke skipped these arms on both Workstation sittings (gate.log: "SKIP (no draft|no model at ...)"); they
# must stay absent so the smoke runs the same checks (31 ok lines per arm).
SMOKE_ABSENT=(/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
              /data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf
              /data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf)

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/chain.log"; }
refuse() {
  echo "REFUSED: $*" >&2
  if [ -n "${OUT:-}" ] && [ -d "$OUT" ]; then echo "$(date -u +%FT%TZ) REFUSED: $*" >> "$OUT/chain.log"; fi
  exit 2
}

# Re-exec once at low CPU and I/O priority (the trainer owns the other cards; no core pinning). FD 9 (the caller's
# window lock) survives the exec.
kit_nice() {
  if [ "${KIT_NICED:-}" != 1 ]; then
    export KIT_NICED=1
    exec nice -n 10 ionice -c2 -n7 bash "$0" "$@"
  fi
}

# The environment every chain step inherits. Inherited MEMRA_* names are dropped: the r4 server's env audit refuses
# names it does not know, and the cell's own names are set by the chains exactly as the registration sets them.
kit_env() {
  [[ "${CARD:-}" =~ ^[0-9]+$ ]] || refuse "set CARD=<nvidia-smi index of the window's card>"
  KIT_DROPPED=$(compgen -e | grep '^MEMRA_' | tr '\n' ' ')
  for v in $KIT_DROPPED; do unset "$v"; done
  unset CUDA_LAUNCH_BLOCKING CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  export CARD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
  export PATH=/usr/local/cuda/bin:$PATH NO_SCOPE=1 KIT_HARNESS=$H K MODEL
  export KIT_IDLE_WAIT_S=${KIT_IDLE_WAIT_S:-600}
}

new_out() {
  if [ -n "${OUT:-}" ]; then
    case "$OUT" in "$K"/out-2*) ;; *) refuse "OUT must be a $K/out-<stamp> dir from an earlier run";; esac
    [ -d "$OUT" ] || refuse "OUT=$OUT does not exist"
    RESUMED=1
  else
    OUT=$K/out-$(date -u +%Y%m%dT%H%M%SZ)$([ "${DRY_RUN:-0}" = 1 ] && echo -dry); RESUMED=0
    mkdir -p "$OUT"
  fi
  mkdir -p "$OUT/.scratch" "$OUT/telemetry" "$OUT/preflight" "$OUT/cut"
  # A stop of an earlier window is history, not this window's outcome.
  if [ -s "$OUT/STOP-REASON" ]; then cat "$OUT/STOP-REASON" >> "$OUT/stops-earlier.txt"; rm -f -- "$OUT/STOP-REASON"; fi
  export OUT
}

# The caller's GPU window: FD 9 must be open on $EVALS/gpu$CARD.lock and that file must be flock-held (an independent
# non-blocking flock fails).
check_window() { # $1 = refuse|report
  local want=$EVALS/gpu$CARD.lock have why=""
  have=$(readlink "/proc/$$/fd/9" 2>/dev/null)
  [ "$have" = "$want" ] || why="FD 9 is '${have:-closed}', not $want: run from the session that holds the card's window (exec 9>$want; flock 9)"
  if [ -z "$why" ] && flock -n "$want" true 2>/dev/null; then why="$want is not locked: the window is not held"; fi
  if [ -n "$why" ]; then
    [ "$1" = refuse ] && refuse "$why"
    echo "window check would refuse: $why"
  else
    echo "window: FD 9 holds $want"
  fi
}

check_card() { # $1 = refuse|report, $2 = label (untimed|timed|dry)
  local q name used apps
  q=$(nvidia-smi -i "$CARD" --query-gpu=index,name,memory.used,memory.total,power.limit,power.max_limit,power.default_limit,enforced.power.limit,temperature.gpu,fan.speed,driver_version --format=csv,noheader,nounits 2>&1) \
    || refuse "nvidia-smi -i $CARD failed: $q"
  echo "index,name,memory.used_mib,memory.total_mib,power.limit_w,power.max_limit_w,power.default_limit_w,enforced.power.limit_w,temperature.gpu_c,fan.speed_pct,driver_version" > "$OUT/preflight/card-$2.csv"
  echo "$q" >> "$OUT/preflight/card-$2.csv"
  nvidia-smi -i "$CARD" --query-gpu=name,power.limit,memory.total --format=csv > "$OUT/card.csv" 2>&1
  name=$(echo "$q" | cut -d, -f2 | sed 's/^ *//; s/ *$//')
  used=$(echo "$q" | cut -d, -f3 | tr -d ' ')
  apps=$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  echo "$apps" > "$OUT/preflight/compute-apps-$2.csv"
  local why=""
  [ "$name" = "$MAXQ_NAME" ] || why="card $CARD is '$name', not '$MAXQ_NAME'"
  [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 2000 ] || why="${why:+$why; }card $CARD has ${used} MiB used (the cell needs < 2000)"
  [ -z "$apps" ] || why="${why:+$why; }card $CARD has compute apps: $(echo "$apps" | tr '\n' ';')"
  if [ -n "$why" ]; then
    [ "$1" = refuse ] && refuse "$why"
    echo "card check would refuse: $why"
  fi
  POWER_LIMIT=$(echo "$q" | cut -d, -f5 | tr -d ' ')
  echo "card $CARD: $name, power.limit ${POWER_LIMIT} W, ${used} MiB used, driver $(echo "$q" | cut -d, -f11 | tr -d ' ')"
}

# One snapshot of the box (every card) for the regime record: what else ran beside the cell. No identifiers.
box_snapshot() { # $1 label
  nvidia-smi --query-gpu=index,name,power.limit,power.draw,temperature.gpu,fan.speed,memory.used,utilization.gpu --format=csv > "$OUT/preflight/box-$1.csv" 2>&1
  nvidia-smi --query-compute-apps=gpu_bus_id,used_memory --format=csv > "$OUT/preflight/box-compute-apps-$1.csv" 2>&1
  cut -d' ' -f1-3 /proc/loadavg > "$OUT/preflight/loadavg-$1.txt"
}

check_static() {
  command -v ss >/dev/null || command -v lsof >/dev/null || refuse "neither ss nor lsof on PATH: the gates cannot check their ports"
  local t; for t in python3 flock curl sha256sum nvidia-smi ionice; do command -v "$t" >/dev/null || refuse "$t missing"; done
  [ "$(cat "$K/bins/lane/source.commit")" = "$LANE_SHA" ] || refuse "bins/lane/source.commit is not $LANE_SHA"
  [ "$(cat "$K/bins/main/source.commit")" = "$MAIN_SHA" ] || refuse "bins/main/source.commit is not $MAIN_SHA"
  [ "$(cat "$H/SOURCE.commit")" = "$HARNESS_SHA" ] || refuse "harness/SOURCE.commit is not $HARNESS_SHA"
  (cd "$K/bins/lane" && sha256sum -c --quiet SHA256SUMS) > "$OUT/preflight/bins-lane.check" 2>&1 || refuse "lane bins fail SHA256SUMS: $(head -3 "$OUT/preflight/bins-lane.check")"
  (cd "$K/bins/main" && sha256sum -c --quiet SHA256SUMS) > "$OUT/preflight/bins-main.check" 2>&1 || refuse "main bin fails SHA256SUMS: $(head -3 "$OUT/preflight/bins-main.check")"
  (cd "$H" && sha256sum -c --quiet MANIFEST.sha256) > "$OUT/preflight/harness.check" 2>&1 || refuse "harness files fail MANIFEST.sha256: $(head -3 "$OUT/preflight/harness.check")"
  (cd "$K" && sha256sum -c --quiet KIT-MANIFEST.sha256) > "$OUT/preflight/kit.check" 2>&1 || refuse "kit files fail KIT-MANIFEST.sha256 (edited on the box?): $(head -3 "$OUT/preflight/kit.check")"
  cp "$K/KIT-MANIFEST.sha256" "$OUT/preflight/KIT-MANIFEST.sha256"
  local b; for b in "$K/bins/lane/memra-server" "$K/bins/lane/kv-tier-gate" "$K/bins/lane/vmm-call-cost" "$K/bins/main/memra-server"; do
    ldd "$b" > "$OUT/preflight/ldd-$(basename "$(dirname "$b")")-$(basename "$b").txt" 2>&1
    if grep -q 'not found' "$OUT/preflight/ldd-$(basename "$(dirname "$b")")-$(basename "$b").txt"; then refuse "$b: unresolved libraries"; fi
  done
  local p; for p in "${SMOKE_ABSENT[@]}"; do [ ! -e "$p" ] || refuse "$p exists: serve-smoke would run an arm the registration skipped"; done
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || refuse "less than 24 GB host memory available"
  [ "$(df -BG --output=avail "$K" | tail -1 | tr -dc 0-9)" -ge 20 ] || refuse "less than 20 GB free under $K"
  # serve-smoke runs target/release/memra-server from the harness root: the prebuilt lane binary, linked.
  mkdir -p "$H/target/release"; ln -sfn "$K/bins/lane/memra-server" "$H/target/release/memra-server"
  { echo "lane=$LANE_SHA main=$MAIN_SHA harness=$HARNESS_SHA"; cat "$K/bins/lane/SHA256SUMS" "$K/bins/main/SHA256SUMS"; } > "$OUT/preflight/sources.txt"
}

check_model() {
  [ -f "$MODEL" ] || refuse "no model at $MODEL"
  if [ -s "$OUT/preflight/model.sha256" ] && [ "$RESUMED" = 1 ] && [ "$(stat -c %Y "$MODEL")" -le "$(stat -c %Y "$OUT/preflight/model.sha256")" ]; then
    echo "model sha256 checked at $(stat -c %y "$OUT/preflight/model.sha256" | cut -c1-19) and the file is unchanged since"
  else
    sha256sum "$MODEL" > "$OUT/preflight/model.sha256.tmp"; mv "$OUT/preflight/model.sha256.tmp" "$OUT/preflight/model.sha256"
  fi
  [ "$(cut -d' ' -f1 "$OUT/preflight/model.sha256")" = "$WANT_MODEL" ] || refuse "model sha256 $(cut -d' ' -f1 "$OUT/preflight/model.sha256") is not $WANT_MODEL"
  echo "model sha256 ok ($WANT_MODEL)"
}

# The canonical memra lock, held by the runner for the whole window on FD 7. The collector and the boots inherit it
# (KIT_LOCK_FD; harness ports in tier-battery.py and boots-f.sh). Closing FD 7 at exit releases it.
take_lock() {
  exec 7>>"$RIG_LOCK"
  flock -n 7 || refuse "$RIG_LOCK is held by another memra session"
  export KIT_LOCK_FD=7
  echo "$(date -u +%FT%TZ) $RIG_LOCK held by the runner (pid $$, FD 7)" >> "$OUT/preflight/lock.txt"
}

TELEM_PID=""; WATCH_PID=""
start_watch() { # $1 part label, $2 deadline epoch
  local csv=$OUT/telemetry/card$CARD-$1-$(date -u +%Y%m%dT%H%M%SZ).csv
  setsid nvidia-smi -i "$CARD" --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,fan.speed,memory.used,utilization.gpu,clocks_event_reasons.active --format=csv -lms 250 > "$csv" 2> "$csv.err" &
  TELEM_PID=$!
  setsid python3 "$K/watchdog.py" --csv "$csv" --deadline "$2" --marker "KIT_RUN_ID=$KIT_RUN_ID" --log "$OUT/chain.log" --stopfile "$OUT/STOP-REASON" > "$OUT/.scratch/watchdog-$1.log" 2>&1 &
  WATCH_PID=$!
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [ -n "${KIT_RUN_ID:-}" ] && python3 "$K/watchdog.py" --kill-only --marker "KIT_RUN_ID=$KIT_RUN_ID" --log "$OUT/chain.log" >> "$OUT/chain.log" 2>&1
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null
  [ -n "$TELEM_PID" ] && kill "$TELEM_PID" 2>/dev/null
  wait 2>/dev/null
  nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/preflight/compute-apps-end.csv" 2>&1
  box_snapshot end
  # Scratch this runner created: the runner's own temp dir and serve-smoke's fixed /tmp files. Literal paths only;
  # the receipts under $OUT are the deliverable and stay.
  case "$OUT" in "$K"/out-2*) rm -rf -- "$OUT/.scratch";; esac
  rm -f -- /tmp/serve-smoke.log /tmp/serve-smoke-conc-1.json /tmp/serve-smoke-conc-2.json /tmp/serve-smoke-conc-3.json \
    /tmp/serve-smoke-affinity.py /tmp/serve-smoke-affinity.json
  exit "$rc"
}

# Reader and rule over the whole receipt root (both parts when the timed part has run).
read_and_rule() {
  ( cd "$H" && python3 research/spill-b-20260919/day37-read.py "$CARD_LABEL" "$OUT" gates-maxq ) > "$OUT/read.log" 2>&1
  echo "reader rc=$? ($(grep -c '^DAY37 ' "$OUT/read.log") clause lines)" >> "$OUT/chain.log"
  python3 "$K/day37-verdict.py" "$CARD_LABEL" "$OUT/read.log" > "$OUT/VERDICT.txt" 2>&1
  {
    echo "card_index=$CARD power_limit_w=${POWER_LIMIT:-?} name=$MAXQ_NAME"
    echo "sources lane=$LANE_SHA main=$MAIN_SHA harness=$HARNESS_SHA placement=MEMRA_KV_VMM_GROW=inline (pinned, PREREG.md)"
    [ -s "$OUT/STOP-REASON" ] && echo "stopped: $(cat "$OUT/STOP-REASON")"
  } >> "$OUT/VERDICT.txt"
  ( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -path './.scratch/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
}
