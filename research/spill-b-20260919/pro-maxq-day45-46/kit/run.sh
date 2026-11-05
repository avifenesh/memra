#!/usr/bin/env bash
# admit-w-release Max-Q deciding cell (PREREG.md): DAY45 2.2 and DAY46 2.2 replayed at their deciding trees, with
# portability fixes only (DEVIATIONS.md), on one RTX PRO 6000 Blackwell Max-Q Workstation Edition of <box>.
#
# usage (by the session that holds the GPU window and already holds /mnt/Data/work/evals/gpu2.lock on FD 9):
#   CARD=<idx> bash /mnt/Data/work/memra-maxq/admit-w-release/run.sh
# env: CARD             required, the nvidia-smi index of the window's card (the server sees only it)
#      PART             all (default: DAY45 half, then DAY46 half) | 1 (DAY45 half) | 2 (DAY46 half)
#      TOTAL_TIMEOUT_S  hard stop for the whole cell (default 13800 for all, 6600 for part 1, 9000 for part 2)
#      LOCK_WAIT_S      bounded wait for /tmp/memra-gpu.lock (default 300)
#      IDLE_WAIT_S      bounded per-boot idle wait inside day45-run.sh/day46-run.sh (default 600)
#      DRY_RUN=1        every check, the lock and the chains' own dry runs; no server boot, card occupancy only reported
# Writes ROOT/out-<UTC stamp>/: run.out (this log), card.csv, env.txt, model.sha256, telemetry.csv (250 ms, CARD only),
# compute-apps-card.csv, guard.log, day45/ and day46/ (the registered receipt layout: chain.log, run.log, card.csv,
# boots/<boot>/server.log, client.jsonl, ...; read.log, SUMMARY.txt), READINGS.txt, VERDICT.txt, MANIFEST.sha256.
# Never touches FD 9, another card, or a process it did not start. Deletes only ROOT/scratch-<stamp> on exit.
set -uo pipefail
ROOT=${KIT_ROOT:-/mnt/Data/work/memra-maxq/admit-w-release}
MODEL=${MODEL:-/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
RUN_LOCK=${KIT_LOCK:-/tmp/memra-gpu.lock}
MAXQ="RTX PRO 6000 Blackwell Max-Q Workstation"
PORT=${KIT_PORT:-18526}
CARD=${CARD:?usage: CARD=<idx> bash run.sh}
case $CARD in ''|*[!0-9]*) echo "REFUSED: CARD=$CARD is not an nvidia-smi index"; exit 2 ;; esac
PART=${PART:-all}
case $PART in
  all) parts="1 2"; tdef=13800 ;;
  1) parts=1; tdef=6600 ;;
  2) parts=2; tdef=9000 ;;
  *) echo "REFUSED: PART=$PART (all, 1 or 2)"; exit 2 ;;
esac
TOTAL_TIMEOUT_S=${TOTAL_TIMEOUT_S:-$tdef}
LOCK_WAIT_S=${LOCK_WAIT_S:-300}
DRY=${DRY_RUN:-0}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/out-$STAMP
SCRATCH=$ROOT/scratch-$STAMP
cd "$ROOT" || { echo "REFUSED: no kit at $ROOT"; exit 2; }
mkdir "$OUT" 2>/dev/null || { echo "REFUSED: $OUT exists or cannot be made (one run per second)"; exit 2; }
exec > >(tee -a "$OUT/run.out") 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
refuse() { say "REFUSED: $*"; echo "REFUSED: $*" > "$OUT/REFUSED.txt"; exit 3; }
CELL_PID=""; GUARD_PID=""
cleanup() {
  local rc=$?
  if [ -n "$GUARD_PID" ] && kill -0 "$GUARD_PID" 2>/dev/null; then kill -TERM "$GUARD_PID" 2>/dev/null; wait "$GUARD_PID" 2>/dev/null; fi
  if [ -n "$CELL_PID" ] && kill -0 -- "-$CELL_PID" 2>/dev/null; then
    say "cleanup: SIGTERM to the cell's process group $CELL_PID"
    kill -TERM -- "-$CELL_PID" 2>/dev/null
    for _ in $(seq 60); do kill -0 -- "-$CELL_PID" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$CELL_PID" 2>/dev/null && { say "cleanup: SIGKILL to group $CELL_PID"; kill -KILL -- "-$CELL_PID" 2>/dev/null; }
  fi
  case $STAMP in
    2[0-9][0-9][0-9][01][0-9][0-3][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) rm -rf -- "$ROOT/scratch-$STAMP" ;;
  esac
  say "exit rc=$rc OUT=$OUT"
  echo "OUT=$OUT"
}
trap cleanup EXIT
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
say "admit-w-release Max-Q kit: CARD=$CARD PART=$PART TOTAL_TIMEOUT_S=$TOTAL_TIMEOUT_S DRY_RUN=$DRY OUT=$OUT"

# 0. a clean MEMRA_* environment: the registered program is the binaries' defaults plus the arms' own env
cleared=$(compgen -e | grep '^MEMRA_' | tr '\n' ' ')
for v in $cleared; do unset "$v"; done
[ -n "$cleared" ] && say "cleared inherited env: $cleared"

# 1. the card: the Max-Q name, under 2000 MiB used, no compute app (CARD only)
q=$(nvidia-smi -i "$CARD" --query-gpu=name,memory.used --format=csv,noheader,nounits 2>&1) || refuse "nvidia-smi -i $CARD: $q"
name=${q%,*}; used=${q##*,}; used=${used// /}
case $name in *"$MAXQ"*) ;; *) refuse "card $CARD is '$name', not an $MAXQ Edition" ;; esac
apps=$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
if ! [[ $used =~ ^[0-9]+$ ]] || [ "$used" -ge 2000 ] || [ -n "$apps" ]; then
  if [ "$DRY" = 1 ]; then say "DRY_RUN: card $CARD has ${used} MiB used, apps=[${apps//$'\n'/; }]; a real run refuses this"
  else refuse "card $CARD has ${used} MiB used (needs < 2000) or compute apps [${apps//$'\n'/; }]"; fi
fi
nvidia-smi -i "$CARD" --query-gpu=index,name,pci.bus_id,uuid,power.limit,enforced.power.limit,power.default_limit,power.max_limit,memory.total,memory.used,temperature.gpu,fan.speed,driver_version --format=csv > "$OUT/card.csv"
say "card: $(tail -n 1 "$OUT/card.csv")"
# a read-only label of what shares the box (all cards, start and end only; no check reads it)
nvidia-smi --query-gpu=index,memory.used,power.draw,temperature.gpu --format=csv > "$OUT/box-state-start.csv" 2>&1

# 2. tools, host, port, disk
for t in python3 curl flock setsid sha256sum nvidia-smi nice; do command -v "$t" >/dev/null || refuse "no $t on PATH"; done
command -v ss >/dev/null || command -v lsof >/dev/null || refuse "neither ss nor lsof on PATH"
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -ge 24 ] || refuse "host memory available ${avail} GB < 24"
ss -ltn 2>/dev/null | grep -q ":$PORT " && refuse "port $PORT busy"
free_gb=$(df --output=avail -BG "$ROOT" | tail -n 1 | tr -dc 0-9); [ "$free_gb" -ge 5 ] || refuse "only ${free_gb} GB free under $ROOT"

# 3. the binaries and the model, by hash
( cd "$ROOT/bins" && sha256sum -c --quiet SHA256SUMS ) || refuse "kit binaries do not match bins/SHA256SUMS"
[ -f "$MODEL" ] || refuse "model absent: $MODEL"
say "hashing $MODEL"
h=$(sha256sum "$MODEL" | cut -d' ' -f1)
echo "$h  $MODEL" > "$OUT/model.sha256"
[ "$h" = "$WANT_MODEL" ] || refuse "model sha256 $h != $WANT_MODEL"
say "model sha256 OK ($h)"

# 4. the memra lock (one scored campaign per box; FD 8, never the caller's FD 9)
exec 8>"$RUN_LOCK"
say "waiting for $RUN_LOCK (flock -w $LOCK_WAIT_S)"
flock -w "$LOCK_WAIT_S" 8 || refuse "$RUN_LOCK not acquired within $LOCK_WAIT_S s"
say "lock acquired: $RUN_LOCK" | tee "$OUT/lock.txt"

# 5. the environment of every boot
mkdir -p "$SCRATCH/nv-cache" "$SCRATCH/tmp"
export ROOT OUT CARD MODEL PORT CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD RIG_LOCK_HELD=1 CELL_LOCK=none \
  IDLE_WAIT_S=${IDLE_WAIT_S:-600} READER_CARD=pro6000maxq CUDA_CACHE_PATH=$SCRATCH/nv-cache TMPDIR=$SCRATCH/tmp DRY_RUN=$DRY
env | grep -E '^(CARD|PART|PORT|CUDA_|MEMRA_|MODEL|RIG_LOCK_HELD|CELL_LOCK|IDLE_WAIT_S|READER_CARD|TMPDIR|DRY_RUN)=' | sort > "$OUT/env.txt"
{ echo "python3=$(python3 --version 2>&1)"; echo "kernel=$(uname -r)"; echo "host=$(hostname)"; } >> "$OUT/env.txt"

# 6. the cell, in its own process group at nice 10, under the guard (deadline, thermal rule, 250 ms telemetry)
say "cell start: parts=[$parts]"
nice -n 10 setsid bash "$ROOT/cell.sh" $parts < /dev/null &
CELL_PID=$!
python3 -I "$ROOT/lib/guard.py" --card "$CARD" --pgid "$CELL_PID" --out "$OUT" --deadline-s "$TOTAL_TIMEOUT_S" \
  > "$OUT/guard.out" 2>&1 &
GUARD_PID=$!
wait "$CELL_PID"; cell_rc=$?
CELL_PID_DONE=$CELL_PID
kill -0 -- "-$CELL_PID_DONE" 2>/dev/null || CELL_PID=""
say "cell end rc=$cell_rc"
kill -TERM "$GUARD_PID" 2>/dev/null; wait "$GUARD_PID" 2>/dev/null; GUARD_PID=""
for f in TIMEOUT.txt THERMAL-STOP.txt; do [ -f "$OUT/$f" ] && say "$f: $(cat "$OUT/$f")"; done

# 7. readers (a chain cut before its reader leaves no read.log), the verdict, a manifest
for p in $parts; do
  case $p in
    1) d=$OUT/day45; r=$ROOT/d45/research/spill-b-20260919/day45-read.py ;;
    2) d=$OUT/day46; r=$ROOT/d46/research/spill-b-20260919/day46-read.py ;;
  esac
  if [ -d "$d/boots" ] && [ ! -s "$d/read.log" ]; then python3 -I "$r" pro6000maxq "$d" > "$d/read.log" 2>&1; fi
done
nvidia-smi --query-gpu=index,memory.used,power.draw,temperature.gpu --format=csv > "$OUT/box-state-end.csv" 2>&1
python3 -I "$ROOT/verdict.py" --card pro6000maxq --root "$ROOT" --out "$OUT" > "$OUT/verdict.out" 2>&1
cat "$OUT/verdict.out"
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
exit "$cell_rc"
