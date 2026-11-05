#!/usr/bin/env bash
# batch-oom-recover on ONE RTX PRO 6000 Blackwell Max-Q Workstation Edition (DAY49 addendum D, the cell that set
# docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md on the full-power Workstation Edition). Arms by binary: off = 2ccb038d1c
# as is (the Max-Q keeps the chunk's error arm), on = 2ccb038d1c plus maxq-batch-oom-recover.patch (is_pro6000_maxq
# joins the default). The registered units: health-fault-gate arm j twice from the on binary (j1, j2), then the
# serving shape's eight boots in the registered order, then the registered reader and maxq-verdict.py.
# Pre-registration: PREREG.md. Every change against the registered scripts: DEVIATIONS.md. No clause is timed, so
# the whole cell runs in a gap beside other cards' training; there is no run-timed.sh.
#
# Run on <box> by the session that holds the card's window (it holds /mnt/Data/work/evals/gpu<CARD>.lock on FD 9):
#   CARD=<nvidia-smi index> bash /mnt/Data/work/memra-maxq/batch-oom-recover/run.sh
# Optional env: TOTAL_TIMEOUT_S (default 5400; every step is bounded by what is left), LOCK_WAIT_S (default 120,
# the wait for /tmp/memra-gpu.lock), SKIP_EVAL_LOCK_CHECK=1 (do not insist on FD 9).
# Output: /mnt/Data/work/memra-maxq/batch-oom-recover/out-<UTC stamp>/ with VERDICT.txt, raw logs, 250 ms telemetry
# of the pinned card; the last line printed is "OUT <dir>". Exit 0 when a VERDICT line was written, 2 on a refusal,
# 3 when the run stopped (deadline, thermal, pin) and still wrote its NOT-READ verdict.
set -uo pipefail
umask 022

KIT=/mnt/Data/work/memra-maxq/batch-oom-recover
SCRATCH=/mnt/Data/work/memra-maxq/batch-oom-recover/scratch
MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MODEL_SHA=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
MODEL_BYTES=15705922304
MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
LOCK=/tmp/memra-gpu.lock
EVAL_LOCK_DIR=/mnt/Data/work/evals
GATE_PORT=8189
SERVE_PORT=18526
SRC=2ccb038d1c16358edde65582597e7b840e3f63fb
PATCH_SHA=6d88aae54863c8b5c65ab3e470a4ae8e093158995810acb8d93bc3002a979dbd
BIN_OFF=$KIT/bins/off/memra-server
BIN_ON=$KIT/bins/on/memra-server
BIN_OFF_SHA=711b4f674bdb501f2387374e5bd6b1a21ac86a45f4a38c9b32adde866c1a4ee3
BIN_ON_SHA=91f689817d9c92f415414e5980eca56617c31fc4ce76f0c962ea617c5247ec7e
SRC_OFF="$SRC"
SRC_ON="$SRC + maxq-batch-oom-recover.patch sha256=$PATCH_SHA"
TOTAL_TIMEOUT_S=${TOTAL_TIMEOUT_S:-5400}
GATE_CAP_S=1800     # one gate unit: three boots, each with the registered HFG_READY_WAIT_S=900
BOOTS_CAP_S=3600    # the eight serving boots (Workstation: 5.8 min)
CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64"
BOOTS=(O1-off:off O1-on:on O2-on:on O2-off:off P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain)

# CPU cap: the whole cell runs at nice 10 (never pinned: the trainers own their cores). FD 9 survives the exec.
if [ "${_BOR_NICED:-0}" != 1 ]; then exec env _BOR_NICED=1 nice -n 10 bash "$0" "$@"; fi

START=$(date +%s)
END=$(( START + TOTAL_TIMEOUT_S ))
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$KIT/out-$STAMP
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }
GUARD_PID=; STEP_PID=; LFD=; STOPPED=0; SCRATCH_MADE=0

note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
refuse() {
  note "REFUSED: $*"
  echo "VERDICT batch-oom-recover card=pro6000-maxq refused=[$*] -> NOT-RUN" > "$OUT/VERDICT.txt"
  exit 2
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  [ -n "$STEP_PID" ] && kill -TERM "$STEP_PID" 2>/dev/null
  # Never a signal to anything this kit did not start: guard.py reap matches executable under $KIT/bins and cwd $KIT.
  python3 -I "$KIT/guard.py" reap --kit "$KIT" --grace-s 20 >> "$OUT/run.log" 2>&1
  if [ -n "$GUARD_PID" ]; then kill -TERM "$GUARD_PID" 2>/dev/null; wait "$GUARD_PID" 2>/dev/null; fi
  # Scratch this run created, by literal path (TMPDIR of every child; the cell itself writes only logs under $OUT).
  # Only the lock holder made it, so a refused second run never deletes a live run's scratch.
  [ "$SCRATCH_MADE" = 1 ] && rm -rf /mnt/Data/work/memra-maxq/batch-oom-recover/scratch
  find "$OUT" -type f -name '.sample.*' -delete 2>/dev/null
  [ -n "$LFD" ] && exec {LFD}>&-
  note "exit rc=$rc"
  echo "OUT $OUT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

note "batch-oom-recover Max-Q kit: start out=$OUT total_timeout=${TOTAL_TIMEOUT_S}s"

# ---- refusals before anything touches the card
[ "$(cd "$(dirname "$0")" && pwd -P)" = "$KIT" ] || refuse "run.sh must run from $KIT"
[[ "${CARD:-}" =~ ^[0-9]+$ ]] || refuse "CARD must be an nvidia-smi index (got '${CARD:-}')"
for t in nvidia-smi python3 curl flock sha256sum timeout free df; do
  command -v "$t" >/dev/null || refuse "$t not on PATH"
done
command -v ss >/dev/null || command -v lsof >/dev/null || refuse "neither ss nor lsof on PATH"
q=$(nvidia-smi -i "$CARD" --query-gpu=name,memory.used,uuid,pci.bus_id,power.limit --format=csv,noheader,nounits 2>&1) \
  || refuse "nvidia-smi -i $CARD: $q"
IFS=, read -r c_name c_used c_uuid c_bus c_plimit <<< "$q"
c_name=$(trim "$c_name"); c_used=$(trim "$c_used"); c_uuid=$(trim "$c_uuid"); c_bus=$(trim "$c_bus"); c_plimit=$(trim "$c_plimit")
[ "$c_name" = "$MAXQ_NAME" ] || refuse "card $CARD is '$c_name', not '$MAXQ_NAME'"
[[ "$c_used" =~ ^[0-9]+$ ]] && [ "$c_used" -lt 2000 ] || refuse "card $CARD has $c_used MiB in use (needs < 2000)"
apps=$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
[ -z "$apps" ] || refuse "card $CARD has compute apps: $(echo "$apps" | tr '\n' ' ')"
EVAL_LOCK=$EVAL_LOCK_DIR/gpu$CARD.lock
if [ "${SKIP_EVAL_LOCK_CHECK:-0}" != 1 ]; then
  [ "$(readlink /proc/$$/fd/9 2>/dev/null)" = "$EVAL_LOCK" ] || refuse "FD 9 is not $EVAL_LOCK (the window holder's lock)"
  if flock -n "$EVAL_LOCK" true; then refuse "$EVAL_LOCK is open on FD 9 but not locked"; fi
fi
for p in $GATE_PORT $SERVE_PORT; do
  bash "$KIT/tools/port-guard.sh" check batch-oom-recover "$p" >> "$OUT/run.log" 2>&1 || refuse "port $p is busy or unobservable"
done
avail=$(df --output=avail -B1M "$KIT" | tail -1 | tr -d ' ')
[ "${avail:-0}" -ge 2048 ] || refuse "only ${avail} MiB free under $KIT"
[ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || refuse "less than 24 GB host memory available"
(cd "$KIT" && sha256sum -c --quiet KIT.sha256) >> "$OUT/run.log" 2>&1 || refuse "kit files differ from KIT.sha256"
[ "$(sha256sum "$BIN_OFF" | cut -d' ' -f1)" = "$BIN_OFF_SHA" ] || refuse "bins/off/memra-server sha256 mismatch"
[ "$(sha256sum "$BIN_ON" | cut -d' ' -f1)" = "$BIN_ON_SHA" ] || refuse "bins/on/memra-server sha256 mismatch"
for b in "$BIN_OFF" "$BIN_ON"; do
  missing=$(ldd "$b" 2>&1 | grep 'not found' || true)
  [ -z "$missing" ] || refuse "$b: $missing"
done
[ -f "$MODEL" ] || refuse "no model at $MODEL"
[ "$(stat -c %s "$MODEL")" = "$MODEL_BYTES" ] || refuse "model size $(stat -c %s "$MODEL") != $MODEL_BYTES (copy not finished?)"

# ---- the memra lock for the whole cell (AGENTS.md lock table: the PRO 6000 box lock)
exec {LFD}>>"$LOCK" || refuse "cannot open $LOCK"
flock -w "${LOCK_WAIT_S:-120}" "$LFD" || refuse "$LOCK busy for ${LOCK_WAIT_S:-120}s (another memra lane holds the box lock)"
note "lock $LOCK held on fd $LFD"

note "model sha256 check ($MODEL_BYTES B)"
msha=$(sha256sum "$MODEL" | cut -d' ' -f1)
echo "$msha  $MODEL" > "$OUT/model.sha256"
[ "$msha" = "$MODEL_SHA" ] || refuse "model sha256 $msha != $MODEL_SHA"
# The card can change while the model hashes: read it again.
used2=$(trim "$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits)")
[ "$used2" -lt 2000 ] || refuse "card $CARD now has $used2 MiB in use"
[ -z "$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader)" ] || refuse "card $CARD gained a compute app"

# ---- environment: the pinned card, no inherited MEMRA_* name, children's scratch under the kit
inherited=$(compgen -e | grep '^MEMRA_' || true)
echo "${inherited:-none}" > "$OUT/env-scrubbed.txt"
for v in $inherited; do unset "$v"; done
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD CARD
mkdir -p "$SCRATCH" && SCRATCH_MADE=1; export TMPDIR=$SCRATCH

# ---- records
nvidia-smi -i "$CARD" --query-gpu=index,name,uuid,pci.bus_id,power.limit,power.default_limit,power.max_limit,enforced.power.limit,memory.total,memory.used,driver_version,temperature.gpu,fan.speed,clocks.max.sm,clocks.max.memory --format=csv > "$OUT/card.csv"
nvidia-smi -i "$CARD" -q > "$OUT/card-q-before.txt" 2>&1
{ echo "host $(hostname)"; uname -a; echo "nproc $(nproc)"; cat /proc/loadavg; free -g; df -h "$KIT"; nvidia-smi -L; } > "$OUT/host.txt" 2>&1
{ echo "CARD=$CARD uuid=$c_uuid bus=$c_bus power_limit_w=$c_plimit"; echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES";
  echo "nice=$(nice) TMPDIR=$TMPDIR"; echo "eval_lock=$(readlink /proc/$$/fd/9 2>/dev/null || echo none)"; echo "memra_lock=$LOCK"; } > "$OUT/env.txt"
{ echo "off $BIN_OFF_SHA $SRC_OFF"; echo "on  $BIN_ON_SHA $SRC_ON"; } > "$OUT/binaries.sha256"
cp "$KIT/PROVENANCE.txt" "$KIT/KIT.sha256" "$OUT/"
note "card $CARD $c_name uuid=$c_uuid bus=$c_bus power.limit=${c_plimit} W used=${c_used} MiB"

# ---- telemetry (250 ms, this card only), thermal guard and pin check
python3 -I "$KIT/guard.py" watch --kit "$KIT" --out "$OUT" --card "$CARD" --uuid "$c_uuid" > "$OUT/guard.log" 2>&1 &
GUARD_PID=$!

step() { # <name> <cap_s> <log> cmd...: bounded by the step cap and what is left of TOTAL_TIMEOUT_S
  local name=$1 cap=$2 log=$3 rem t rc; shift 3
  if [ -e "$OUT/STOP" ]; then note "step $name not started: $(head -1 "$OUT/STOP")"; return 99; fi
  rem=$(( END - $(date +%s) - 120 ))
  if [ "$rem" -lt 60 ]; then echo "deadline: $name not started (${rem}s left)" >> "$OUT/STOP"; note "step $name not started: deadline"; return 98; fi
  t=$(( cap < rem ? cap : rem ))
  note "step $name start cap=${t}s"
  timeout -k 30 "$t" "$@" > "$log" 2>&1 &
  STEP_PID=$!
  # A stop (guard.py: thermal or pin; the deadline) ends the step at once; guard.py reaps its servers.
  while kill -0 "$STEP_PID" 2>/dev/null; do
    if [ -e "$OUT/STOP" ]; then pkill -TERM -P "$STEP_PID" 2>/dev/null; kill -TERM "$STEP_PID" 2>/dev/null; fi
    sleep 2
  done
  wait "$STEP_PID"; rc=$?
  STEP_PID=
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "deadline: $name stopped after ${t}s (rc=$rc)" >> "$OUT/STOP"; fi
  python3 -I "$KIT/guard.py" reap --kit "$KIT" --grace-s 20 >> "$OUT/run.log" 2>&1  # a step never leaves a server
  note "step $name rc=$rc $(tail -1 "$log" 2>/dev/null | cut -c1-200)"
  return "$rc"
}

# ---- the cell (chain.sh's order: the gates, then the boots)
mkdir -p "$OUT/gates" "$OUT/serve"
for g in j1 j2; do
  step "$g" "$GATE_CAP_S" "$OUT/gates/$g.log" env HFG_BIN="$BIN_ON" HFG_SOURCE="$SRC_ON" HFG_ARMS=j \
    HFG_OUT="$OUT/gates/$g" HFG_READY_WAIT_S=900 HFG_PORT=$GATE_PORT bash "$KIT/tools/health-fault-gate.sh" "$MODEL"
done
echo "j-vmm (vmm-green, vmm-red) not run: the on-demand KV planes are off on this card (kv_vmm.rs:77) and MEMRA_KV_ALLOCATOR is retired; DEVIATIONS.md D2" > "$OUT/gates/j-vmm.NOT-RUN.txt"
step boots "$BOOTS_CAP_S" "$OUT/serve/day49d-run.out" env WT="$KIT" BIN_OFF="$BIN_OFF" BIN_ON="$BIN_ON" \
  SRC_OFF="$SRC_OFF" SRC_ON="$SRC_ON" MODEL="$MODEL" MODEL_KEY=q38 BOOT_CTX= NO_SCOPE=1 FAULT=batch:1 PORT=$SERVE_PORT \
  CLIENT_EXTRA="$CLIENT_EXTRA" bash "$KIT/research/spill-b-20260919/day49d-run.sh" "$OUT/serve" "${BOOTS[@]}"

# ---- stop telemetry, then read
if [ -n "$GUARD_PID" ]; then kill -TERM "$GUARD_PID" 2>/dev/null; wait "$GUARD_PID" 2>/dev/null; GUARD_PID=; fi
nvidia-smi -i "$CARD" -q > "$OUT/card-q-after.txt" 2>&1
python3 -I "$KIT/maxq-verdict.py" --card pro6000-maxq --gate "$OUT/gates/j1" --gate "$OUT/gates/j2" \
  --serve "$OUT/serve" --run-dir "$OUT" --expect-name "$MAXQ_NAME" > "$OUT/verdict.log" 2>&1
grep '^VERDICT ' "$OUT/verdict.log" | tail -1 > "$OUT/VERDICT.txt"
[ -s "$OUT/VERDICT.txt" ] || echo "VERDICT batch-oom-recover card=pro6000-maxq reader-failed=[see verdict.log] -> NOT-READ" > "$OUT/VERDICT.txt"
[ -e "$OUT/STOP" ] && STOPPED=1
note "wall $(( $(date +%s) - START ))s; $(cat "$OUT/VERDICT.txt")"
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
du -sh "$OUT" >> "$OUT/run.log"
[ "$STOPPED" = 1 ] && exit 3
exit 0
