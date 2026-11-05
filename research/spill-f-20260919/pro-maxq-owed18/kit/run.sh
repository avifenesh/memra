#!/usr/bin/env bash
# OWED 18 (KV host-tier handoff: O_DIRECT export against buffered) on ONE RTX PRO 6000 Blackwell
# Max-Q card. It replays the BOX36 cell that set the Workstation default: run-sitting.sh's stage,
# b2-prompts, m1-proof and step 7 (handoff-1g, handoff-8g) at the deciding tree 114ef768d, with
# only the portability fixes listed in DEVIATIONS.md. Pre-registration: PREREG.md.
#
# Run on <box> by the session that holds the quiet-box window, nothing else on the box:
#   CARD=<nvidia-smi index> bash /mnt/Data/work/memra-maxq/handoff-direct/run.sh
# Optional: DEADLINE='YYYY-MM-DD HH:MM' (<local-tz> wall clock; default 2026-11-05 06:50):
# every step is bounded so the run, its reader and the copy back end by then.
# FORCE_8G=1 starts the 8 GiB cell even when its projection does not fit the time left.
# Output: /mnt/Data/work/memra-maxq/handoff-direct/out-<UTC stamp>/ (VERDICT.txt, raw logs,
# 250 ms card telemetry); the last line printed is "OUT <dir>".
set -uo pipefail
umask 022

KIT=/mnt/Data/work/memra-maxq/handoff-direct
S=/mnt/Data/work/memra-maxq/handoff-direct/scratch
MODELS=/mnt/Data/work/memra-maxq/models
ART_FILE=Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ART=$MODELS/$ART_FILE
ART_SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
ART_BYTES=18209036576
MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
LOCK=/tmp/memra-gpu.lock
PORT=18119
F=research/spill-f-20260919
BIN=$KIT/bins/box36
DEADLINE=${DEADLINE:-2026-11-05 06:50}
MARGIN_S=120          # reader, manifest and the copy back after the last cell
MIN_START_S=900       # below this the 1 GiB cell cannot finish (Workstation 444 s, plus setup)
REG_PROOF=600         # run-sitting.sh's collector timeouts; the window bounds them further
REG_1G=7200
REG_8G=10800
EIGHT_OVER_ONE=3.0    # 8 GiB wall / 1 GiB wall: BOX36 2.88; a slower prefill raises the 8 GiB share

LOCKED=0
STEP_PID=
OUT=

refuse() { echo "REFUSED: $*"; exit 2; }
stop() { echo "STOPPED at $1 (rc=$2); receipts kept in $OUT"; exit 3; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
remaining() { echo $(( END - MARGIN_S - $(date +%s) )); }

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  [ -n "$STEP_PID" ] && kill "$STEP_PID" 2>/dev/null
  [ -n "${TEL_PID:-}" ] && kill "$TEL_PID" 2>/dev/null
  if [ -n "${MAXQ_RUN_ID:-}" ]; then
    python3 "$KIT/kit-procs.py" reap --run-id "$MAXQ_RUN_ID" --spare "$$"
  fi
  if [ "$LOCKED" = 1 ]; then
    # Large scratch this run created, by literal path (the handoff file is up to 9.7 GB).
    rm -f /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2/handoff.bin \
          /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2/handoff.bin.tmp \
          /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2-prompts.jsonl
    find /mnt/Data/work/memra-maxq/handoff-direct/scratch -maxdepth 1 -type f -name '.m1-proof-*.bin' -delete
    rmdir /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2 /mnt/Data/work/memra-maxq/handoff-direct/scratch 2>/dev/null
  fi
  if [ -n "$OUT" ] && [ -d "$OUT" ]; then
    python3 "$KIT/maxq-verdict.py" "$OUT" --label pro6000-maxq > "$OUT/VERDICT.txt" 2>&1
    echo "== VERDICT"
    cat "$OUT/VERDICT.txt"
    echo "runner end $(date -u +%FT%TZ) rc=$rc" >> "$OUT/steps.txt"
    (cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -name runner.log -exec sha256sum {} + | sort -k2) > "$OUT/MANIFEST.sha256"
    echo "receipts $(du -sh "$OUT" | cut -f1); scratch removed: $([ -e "$S" ] && echo no || echo yes)"
    echo "OUT $OUT"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

step() {  # step NAME CMD...: raw output to $OUT/NAME.driver.log, bounded by the window
  local name=$1; shift
  local rem; rem=$(remaining)
  if [ "$rem" -lt 30 ]; then echo "== $name not started: ${rem}s left in the window"; return 124; fi
  echo "== $name start $(date -u +%FT%TZ) (${rem}s left in the window)"
  timeout --kill-after=15 "$rem" "$@" > "$OUT/$name.driver.log" 2>&1 &
  STEP_PID=$!
  wait "$STEP_PID"
  local rc=$?
  STEP_PID=
  echo "== $name end $(date -u +%FT%TZ) rc=$rc"
  echo "$name $(date -u +%FT%TZ) rc=$rc" >> "$OUT/steps.txt"
  return "$rc"
}
col() {  # a collector cell on the proven scratch: col NAME REGISTERED_TIMEOUT -- command...
  local name=$1 reg=$2; shift 3
  local ext=() t
  case " $* " in *@COLLECTOR_LOCK_FD@*) ext=(--external-lock) ;; esac
  t=$(( $(remaining) - 20 )); [ "$t" -gt "$reg" ] && t=$reg
  if [ "$t" -lt 60 ]; then echo "== $name not started: ${t}s left for the collector"; return 124; fi
  step "$name" python3 tools/tier-battery.py --rig pro-single --timeout "$t" "${ext[@]}" \
    --storage-root "$S" --storage-proof "$OUT/m1-proof/PROOF.json" --out "$OUT/$name" --execute "$@"
}

# ---------------------------------------------------------------- preflight (no GPU work)
[[ "${CARD:-}" =~ ^[0-9]+$ ]] || refuse "CARD=<nvidia-smi index> is required"
cd "$KIT" || refuse "no kit at $KIT"
for t in python3 nvidia-smi sha256sum flock ss findmnt timeout renice; do
  command -v "$t" > /dev/null || refuse "missing tool: $t"
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || refuse "python3 older than 3.10"
END=$(TZ=<local-tz> date -d "$DEADLINE" +%s) || refuse "DEADLINE '$DEADLINE' does not parse"
[ "$(remaining)" -ge "$MIN_START_S" ] || refuse "only $(remaining)s left before DEADLINE $DEADLINE <local-tz> (minus ${MARGIN_S}s); need ${MIN_START_S}s"

Q=$(nvidia-smi -i "$CARD" --query-gpu=index,name,pci.bus_id,memory.used,memory.total,power.limit,power.max_limit,enforced.power.limit,temperature.gpu,pstate,driver_version --format=csv,noheader,nounits) \
  || refuse "nvidia-smi -i $CARD failed"
IFS=, read -r _ C_NAME _ C_USED _ C_PL _ <<< "$Q"   # all fields go to card.json below
C_NAME=$(trim "$C_NAME"); C_USED=$(trim "$C_USED"); C_PL=$(trim "$C_PL")
[ "$C_NAME" = "$MAXQ_NAME" ] || refuse "card $CARD is \"$C_NAME\", not \"$MAXQ_NAME\""
[[ "$C_USED" =~ ^[0-9]+$ ]] && [ "$C_USED" -lt 2000 ] || refuse "card $CARD has ${C_USED} MiB in use (need under 2000)"
APPS=$(trim "$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)")
[ -z "$APPS" ] || refuse "card $CARD has compute apps: $APPS"

exec 9>> "$LOCK" || refuse "cannot open $LOCK"
flock -n 9 || refuse "$LOCK is held by another job"
LOCKED=1
[ -z "$(ss -Hltn "sport = :$PORT")" ] || refuse "port $PORT is in use"
AVAIL=$(df --output=avail -B1 "$KIT" | tail -n 1 | tr -d ' ')
[ "$AVAIL" -ge $(( 40 * 1024 ** 3 )) ] || refuse "only $AVAIL bytes free under $KIT (need 40 GiB: proof reserve 20 GiB, 9.7 GB handoff file)"
MEMAVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
[ "$MEMAVAIL_KB" -ge $(( 64 * 1024 * 1024 )) ] || refuse "MemAvailable ${MEMAVAIL_KB} kB < 64 GiB (SITTING.md)"
[ -f "$ART" ] || refuse "no artifact at $ART"
[ "$(stat -c %s "$ART")" = "$ART_BYTES" ] || refuse "artifact size $(stat -c %s "$ART") != $ART_BYTES"
(cd "$BIN" && sha256sum -c --quiet SHA256SUMS) || refuse "kit binaries do not match $BIN/SHA256SUMS"

# A stale scratch from an aborted run would fail the driver's first cycle; this run owns it now.
rm -f /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2/handoff.bin \
      /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2/handoff.bin.tmp \
      /mnt/Data/work/memra-maxq/handoff-direct/scratch/b2-prompts.jsonl
mkdir -p "$S/b2" || refuse "cannot create $S/b2"

SCRUBBED=$(compgen -e | grep '^MEMRA_' | tr '\n' ' ')
for v in $SCRUBBED; do unset "$v"; done
renice --priority 10 -p $$ > /dev/null   # CPU cap: nice 10 for the runner and every child, no pinning

# ---------------------------------------------------------------- receipts
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$KIT/out-$STAMP
PRIV=$KIT/private-$STAMP
mkdir "$OUT" || refuse "cannot create $OUT"
mkdir -m 700 "$PRIV" || refuse "cannot create $PRIV"
exec > >(tee -a "$OUT/runner.log" 9>&-) 2>&1
echo "runner start $(date -u +%FT%TZ) card=$CARD deadline='$DEADLINE' <local-tz> ($(remaining)s usable)" | tee "$OUT/steps.txt"
export MAXQ_RUN_ID=$STAMP MAXQ_SMI_CARD=$CARD MAXQ_LOCK_FD=9
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD

cp SOURCE_SHA "$OUT/commit.txt"
(cd "$BIN" && sha256sum kv-handoff-gate memra-server) > "$OUT/binaries.sha256"
python3 -c 'import json, sys; k = ["index", "name", "pci_bus_id", "memory_used_mib", "memory_total_mib", "power_limit_w", "power_max_limit_w", "enforced_power_limit_w", "temperature_c", "pstate", "driver_version"]; v = [x.strip() for x in sys.argv[1].split(",")]; print(json.dumps(dict(zip(k, v)), indent=1))' "$Q" > "$OUT/card.json"
nvidia-smi -i "$CARD" -q -d POWER,TEMPERATURE,CLOCK,PERFORMANCE > "$OUT/card-q.txt" 2>&1
{
  echo "utc $(date -u +%FT%TZ)"
  echo "kernel $(uname -r)"
  echo "nproc $(nproc)"
  echo "loadavg $(cat /proc/loadavg)"
  grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|Dirty):' /proc/meminfo
  cat /proc/swaps
  echo "python $(python3 --version 2>&1)"
  echo "memlock $(ulimit -l)"
  echo "nice $(ps -o ni= -p $$)"
  findmnt -T "$S" -o TARGET,SOURCE,FSTYPE,OPTIONS
} > "$OUT/host.txt" 2>&1
{
  echo "scrubbed MEMRA_* at start: ${SCRUBBED:-none}"
  env | grep -E '^(CUDA_|MAXQ_)' | sort
} > "$OUT/env.txt"
echo "card $CARD: $C_NAME, power.limit ${C_PL} W, ${C_USED} MiB used; nice $(ps -o ni= -p $$)"

# 250 ms telemetry of this card only, for the whole run; the owner's Max-Q thermal rule on top.
nvidia-smi -i "$CARD" --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,enforced.power.limit,temperature.gpu,fan.speed,memory.used,utilization.gpu,clocks_event_reasons.active,pcie.link.gen.current,pcie.link.width.current \
  --format=csv -lms 250 > "$OUT/card-telemetry.csv" 2> "$OUT/card-telemetry.err" 9>&- &
TEL_PID=$!
python3 kit-procs.py thermal-watch --csv "$OUT/card-telemetry.csv" --run-id "$MAXQ_RUN_ID" \
  --spare "$$,$TEL_PID" --out "$OUT/THERMAL-STOP.txt" > "$OUT/thermal-watch.log" 2>&1 9>&- &

# ---------------------------------------------------------------- the cell (run-sitting.sh 2, 3, 7)
step stage bash -c "cd '$MODELS' && echo '$ART_SHA  $ART_FILE' | sha256sum -c" || stop stage $?
step b2-prompts python3 $F/m1-b2-prompts.py --out "$S/b2-prompts.jsonl" \
  --check $F/m1-prereg/b2-prompts.manifest.json || stop b2-prompts $?
t=$(( $(remaining) - 20 )); [ "$t" -gt "$REG_PROOF" ] && t=$REG_PROOF
step m1-proof python3 tools/tier-battery.py --rig pro-single --timeout "$t" --out "$OUT/m1-proof" --execute \
  python3 $F/m1-nvme-proof.py --path "$S" --bind-bytes 1073741824 \
  --private-out "$PRIV/m1-proof.json" --public-out "$OUT/m1-proof/PROOF.json" || stop m1-proof $?

SCHED=$(python3 -c "print(','.join('buffered,direct' if k % 2 == 0 else 'direct,buffered' for k in range(10)))")
HO=(python3 "$F/m1-handoff-driver.py" run --gate "$BIN/kv-handoff-gate" --server "$BIN/memra-server"
    --artifact "$ART" --prompts "$S/b2-prompts.jsonl" --proof "$PRIV/m1-proof.json" --scratch "$S/b2"
    --host-mb 16384 --rig pro-single --lock-fd @COLLECTOR_LOCK_FD@ --io-schedule "$SCHED")
T1=$(date +%s)
col handoff-1g "$REG_1G" -- "${HO[@]}" --size-bytes 1073741824 --out "$OUT/handoff-1g/visits" || stop handoff-1g $?
E1=$(( $(date +%s) - T1 ))
PROJ=$(python3 -c "print(int($E1 * $EIGHT_OVER_ONE))")
REM=$(remaining)
echo "handoff-1g took ${E1}s; handoff-8g projected ${PROJ}s; ${REM}s left in the window"
if [ "${FORCE_8G:-0}" != 1 ] && [ $(( PROJ * 9 / 10 )) -gt "$REM" ]; then
  echo "handoff-8g not started: projected ${PROJ}s (${EIGHT_OVER_ONE} x the 1 GiB cell's ${E1}s) does not fit the ${REM}s left (FORCE_8G=1 overrides)" \
    | tee "$OUT/handoff-8g.SKIPPED"
else
  col handoff-8g "$REG_8G" -- "${HO[@]}" --size-bytes 8589934592 --tenant-pct 100 \
    --out "$OUT/handoff-8g/visits" || stop handoff-8g $?
fi
exit 0
