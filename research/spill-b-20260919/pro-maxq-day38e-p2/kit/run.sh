#!/usr/bin/env bash
# park-compact on ONE RTX PRO 6000 Blackwell Max-Q card: the DAY38 addendum D cell (the deciding cell of the
# park-compaction default) rerun with binary arms at the fix-stack tip. Pre-registration: PREREG.md. Every change
# from the registered cell: DEVIATIONS.md. Sources and hashes: PROVENANCE.txt.
#
# Run on <box> by the session that holds the card's window (the caller holds its own card lock; this runner takes
# /tmp/memra-gpu.lock on FD 8 and never touches FD 9):
#   CARD=<nvidia-smi index> bash /mnt/Data/work/memra-maxq/park-compact/run.sh
# PART (default auto) splits the cell for windows of about 4 h:
#   1    = main-O1-off main-O1-on fault-batch fault-nobatch fault-nobatch-red   (run-part1.sh)
#   2    = main-O2-on main-O2-off, then the combined reading with the newest complete part 1   (run-part2.sh)
#   all  = the registered order in one run (needs a slot of about 5.5 h: set BUDGET_S)
#   auto = 2 when a complete part 1 without a combined reading exists, 1 when no complete part 1 exists,
#          and a refusal when the newest complete part 1 already has its combined reading
# PART1_OUT=<out dir> picks the part 1 that part 2 combines with. BUDGET_S (default 13800 = 3 h 50 min) bounds the
# run: a boot starts only when its projection fits the time left, and the running boot is killed at the budget.
# MODE=timed is run-timed.sh's quiet-box run of the timed readings (P3, P4); it is not for a window between
# other work.
# Output: /mnt/Data/work/memra-maxq/park-compact/out-<UTC stamp>/ (VERDICT.txt, SUMMARY.txt, raw boots, 250 ms
# telemetry of the card) and, after part 2, out-combined-<UTC stamp>/; the last line printed is "OUT <dir>".
set -uo pipefail
umask 022

KIT=/mnt/Data/work/memra-maxq/park-compact
S=/mnt/Data/work/memra-maxq/park-compact/scratch
MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
MODEL_SHA=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
MODEL_BYTES=15705922304
MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
LOCK=/tmp/memra-gpu.lock
LOCK_WAIT_S=${LOCK_WAIT_S:-120}
PORT=18526
MODE=${MODE:-gap}
PART=${PART:-auto}
if [ "$MODE" = timed ]; then BUDGET_S=${BUDGET_S:-21600}; else BUDGET_S=${BUDGET_S:-13800}; fi
# Projection of one main boot before one is measured on this run: the Workstation's 37.4 min (box-d run.log)
# times 600/280 W, the upper bound if the 122,880-token primes scaled with board power (PREREG.md section 5).
MAIN_EST_S=${MAIN_EST_S:-4800}
FAULT_EST_S=${FAULT_EST_S:-600}
MARGIN_S=180   # the reader, the manifest and the scratch cleanup after the last boot

OUT=
COMBINED=
BOOT_PID=
GUARD_PID=
LOCKED=0
refuse() { echo "REFUSED: $*"; exit 2; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
remaining() { echo $(( END - MARGIN_S - $(date +%s) )); }
own_servers() { local p; for p in $(pgrep -x memra-server); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$KIT" ] && echo "$p"; done; }

read_root() {  # read_root DIR: the registered reader plus the rule's VERDICT line, into DIR/VERDICT.txt
  local d=$1
  python3 "$KIT/maxq-verdict.py" "$d" --card pro6000-maxq --kit "$KIT" --card-index "$CARD" \
    --regime "$([ "$MODE" = timed ] && echo timed || echo gap)" > "$d/VERDICT.txt" 2>&1
  grep -E '^DAY38D (P1|P2) |^VERDICT ' "$d/VERDICT.txt" | tail -n 8
  (cd "$d" && find . -type f ! -name MANIFEST.sha256 ! -name runner.log -exec sha256sum {} + | sort -k2) > "$d/MANIFEST.sha256"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  [ -n "$BOOT_PID" ] && kill -TERM "$BOOT_PID" 2>/dev/null && wait "$BOOT_PID" 2>/dev/null
  local p
  for p in $(own_servers); do kill -TERM "$p" 2>/dev/null; done
  [ -n "$GUARD_PID" ] && kill -TERM "$GUARD_PID" 2>/dev/null && wait "$GUARD_PID" 2>/dev/null
  # Scratch this run created under the lock (pid files only; the cell writes no large file), by literal path.
  [ "$LOCKED" = 1 ] && rm -rf /mnt/Data/work/memra-maxq/park-compact/scratch
  if [ -n "$OUT" ] && [ -d "$OUT" ]; then
    nvidia-smi -i "$CARD" --query-gpu=index,name,memory.used,temperature.gpu,fan.speed,power.draw,power.limit --format=csv > "$OUT/card-end.csv" 2>&1
    echo "== reading $OUT"
    read_root "$OUT"
    if [ -n "$COMBINED" ] && [ -d "$COMBINED" ]; then
      echo "== combined reading $COMBINED"
      read_root "$COMBINED"
    fi
    echo "runner end $(date -u +%FT%TZ) rc=$rc" >> "$OUT/steps.txt"
    echo "receipts $(du -sh "$OUT" | cut -f1); scratch removed: $([ -e "$S" ] && echo no || echo yes); servers left: $(own_servers | wc -l)"
    if [ -n "$COMBINED" ] && [ -d "$COMBINED" ]; then echo "OUT $COMBINED"; else echo "OUT $OUT"; fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

# ------------------------------------------------------------------ preflight (no GPU work)
[[ "${CARD:-}" =~ ^[0-9]+$ ]] || refuse "CARD=<nvidia-smi index> is required"
cd "$KIT" || refuse "no kit at $KIT"
for t in python3 nvidia-smi sha256sum flock ss timeout curl free awk pgrep renice du; do
  command -v "$t" > /dev/null || refuse "missing tool: $t"
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' || refuse "python3 older than 3.12 (day38-parse.py needs 3.12 f-strings)"
case $PART in 1|2|all|auto|main) ;; *) refuse "PART=$PART (1, 2, all or auto)";; esac
[ "$MODE" = timed ] && [ "$PART" != main ] && refuse "MODE=timed runs PART=main (run-timed.sh)"
[ "$MODE" != timed ] && [ "$PART" = main ] && refuse "PART=main is run-timed.sh's"
[[ "$BUDGET_S" =~ ^[0-9]+$ ]] || refuse "BUDGET_S=$BUDGET_S"
END=$(( $(date +%s) + BUDGET_S ))

Q=$(nvidia-smi -i "$CARD" --query-gpu=name,memory.used,memory.total,uuid,pci.bus_id,power.limit,enforced.power.limit,power.default_limit,power.max_limit,driver_version,temperature.gpu,fan.speed --format=csv,noheader,nounits) \
  || refuse "nvidia-smi -i $CARD failed"
IFS=, read -r C_NAME C_USED C_TOTAL C_UUID C_BUS C_PL C_EPL C_DPL C_MPL C_DRV C_TEMP C_FAN <<< "$Q"
C_NAME=$(trim "$C_NAME"); C_USED=$(trim "$C_USED")
[ "$C_NAME" = "$MAXQ_NAME" ] || refuse "card $CARD is \"$C_NAME\", not \"$MAXQ_NAME\""
[[ "$C_USED" =~ ^[0-9]+$ ]] && [ "$C_USED" -lt 2000 ] || refuse "card $CARD has ${C_USED} MiB in use (need under 2000)"
APPS=$(trim "$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)")
[ -z "$APPS" ] || refuse "card $CARD has compute apps: $APPS"
if [ "$MODE" = timed ]; then
  # The timed readings need a quiet box: every card without a compute app (a read-only query of all cards).
  ALL=$(trim "$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader)")
  [ -z "$ALL" ] || refuse "not a quiet box (compute apps on other cards): $ALL"
fi
[ -z "$(ss -Hltn "sport = :$PORT")" ] || refuse "port $PORT is in use"
AVAIL=$(df --output=avail -B1 "$KIT" | tail -n 1 | tr -d ' ')
[ "$AVAIL" -ge $(( 5 * 1024 ** 3 )) ] || refuse "only $AVAIL bytes free under $KIT (need 5 GiB)"
MEMAVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
[ "$MEMAVAIL_KB" -ge $(( 32 * 1024 * 1024 )) ] || refuse "MemAvailable ${MEMAVAIL_KB} kB < 32 GiB"
sha256sum -c --quiet KIT-MANIFEST.sha256 || refuse "kit files do not match KIT-MANIFEST.sha256"
[ -f "$MODEL" ] || refuse "no model at $MODEL"
[ "$(stat -c %s "$MODEL")" = "$MODEL_BYTES" ] || refuse "model size $(stat -c %s "$MODEL") != $MODEL_BYTES"

# Which part.
part_done() { [ -f "$1/PART.txt" ] && grep -q "^part=$2 complete=1 kit=$(sha256sum KIT-MANIFEST.sha256 | cut -c1-16)\$" "$1/PART.txt"; }
newest_part1() { local d; for d in $(ls -1d "$KIT"/out-2* 2>/dev/null | sort -r); do part_done "$d" 1 && { echo "$d"; return 0; }; done; return 1; }
if [ "$PART" = auto ] || [ "$PART" = 2 ]; then
  P1=${PART1_OUT:-$(newest_part1)}
  if [ "$PART" = auto ]; then
    if [ -z "$P1" ]; then PART=1
    elif DONE=$(grep -ls "^part1=$P1\$" "$KIT"/out-combined-*/PART.txt); then
      refuse "the cell is complete: $(dirname "$(echo "$DONE" | head -n 1)")/VERDICT.txt (set PART=1 to run it again)"
    else PART=2; fi
  else
    [ -n "$P1" ] || refuse "PART=2 needs a complete part 1 (PART1_OUT or an out-*/PART.txt with part=1 complete=1)"
    part_done "$P1" 1 || refuse "$P1 is not a complete part 1 of this kit"
  fi
fi
case $PART in
  1) BOOTS=(main-O1-off:off main-O1-on:on fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red) ;;
  2) BOOTS=(main-O2-on:on main-O2-off:off) ;;
  all) BOOTS=(main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red) ;;
  main) BOOTS=(main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off) ;;
esac

# The model's sha256 before the first boot (each boot hashes it again into its own model.sha256).
echo "verifying the model sha256 ($MODEL_BYTES bytes)"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$MODEL_SHA" ] || refuse "model sha256 mismatch: $MODEL"

exec 8>> "$LOCK" || refuse "cannot open $LOCK"
flock -w "$LOCK_WAIT_S" 8 || refuse "$LOCK is held by another job (waited ${LOCK_WAIT_S}s)"
LOCKED=1

for v in $(compgen -e | grep '^MEMRA_'); do unset "$v"; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
renice -n 10 -p $$ > /dev/null   # CPU cap for the runner and every child; no core pinning

# ------------------------------------------------------------------ receipts
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$KIT/out-$STAMP
mkdir "$OUT" || refuse "cannot create $OUT"
mkdir -p "$S" || refuse "cannot create $S"
exec > >(tee -a "$OUT/runner.log") 2>&1
{
  echo "runner start $(date -u +%FT%TZ) mode=$MODE part=$PART card=$CARD budget_s=$BUDGET_S boots=[${BOOTS[*]}]"
  echo "caller fd9=$(readlink /proc/$$/fd/9 2>/dev/null || echo none) memra_lock=$LOCK (fd 8)"
} | tee "$OUT/steps.txt"
printf '{"index": %s, "name": "%s", "uuid": "%s", "pci_bus_id": "%s", "memory_used_mib": "%s", "memory_total_mib": "%s", "power_limit_w": "%s", "enforced_power_limit_w": "%s", "default_power_limit_w": "%s", "max_power_limit_w": "%s", "driver": "%s", "temperature_c": "%s", "fan_pct": "%s"}\n' \
  "$CARD" "$C_NAME" "$(trim "$C_UUID")" "$(trim "$C_BUS")" "$C_USED" "$(trim "$C_TOTAL")" "$(trim "$C_PL")" "$(trim "$C_EPL")" \
  "$(trim "$C_DPL")" "$(trim "$C_MPL")" "$(trim "$C_DRV")" "$(trim "$C_TEMP")" "$(trim "$C_FAN")" > "$OUT/card.json"
nvidia-smi -i "$CARD" --query-gpu=name,power.limit,memory.total --format=csv > "$OUT/card.csv"
nvidia-smi -i "$CARD" -q > "$OUT/card-q.txt" 2>&1
[ "$MODE" = timed ] && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv > "$OUT/box-start.csv"
cp SOURCE_SHA "$OUT/source.txt"
echo "power.limit=$(trim "$C_PL") W enforced=$(trim "$C_EPL") W default=$(trim "$C_DPL") W max=$(trim "$C_MPL") W"

python3 "$KIT/card-guard.py" --card "$CARD" --out "$OUT/telemetry.csv" --stop-file "$OUT/THERMAL-STOP.txt" \
  --pid-file "$S/boot.pid" < /dev/null > "$OUT/card-guard.log" 2>&1 &
GUARD_PID=$!

export WT=$KIT SMI_CARD=$CARD RIG_LOCK=$LOCK MODEL MODEL_KEY=q38 BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1 PORT \
  BIN_OFF=$KIT/bins/off/memra-server BIN_ON=$KIT/bins/on/memra-server RED_BIN=$KIT/bins/red/memra-server
main_proj=$MAIN_EST_S
main_measured=0
ok_boots=()
for spec in "${BOOTS[@]}"; do
  name=${spec%%:*}
  case $name in main-*) proj=$main_proj ;; *) proj=$FAULT_EST_S ;; esac
  rem=$(remaining)
  [ -e "$OUT/THERMAL-STOP.txt" ] && { echo "== $name not started: thermal stop"; break; }
  if [ "$rem" -lt "$proj" ]; then
    echo "== $name not started: ${rem}s left, projection ${proj}s" | tee -a "$OUT/steps.txt"
    break
  fi
  echo "== $name start $(date -u +%FT%TZ) (${rem}s left, projection ${proj}s)"
  t0=$(date +%s)
  timeout --kill-after=60 "$rem" bash research/spill-b-20260919/day38d-run.sh "$OUT" "$spec" < /dev/null &
  BOOT_PID=$!
  echo "$BOOT_PID" > "$S/boot.pid"
  wait "$BOOT_PID"
  rc=$?
  BOOT_PID=
  : > "$S/boot.pid"
  dur=$(( $(date +%s) - t0 ))
  echo "$name $(date -u +%FT%TZ) rc=$rc dur_s=$dur" >> "$OUT/steps.txt"
  echo "== $name end rc=$rc dur_s=$dur $(tail -n 1 "$OUT/boots/$name/client.log" 2>/dev/null | cut -c1-100)"
  if [ "$rc" = 124 ] || [ "$rc" = 137 ] || [ "$rc" = 143 ] || [ -e "$OUT/THERMAL-STOP.txt" ]; then
    # A boot cut by the budget or the thermal guard is not a reading: keep it aside, out of the reader's view.
    mkdir -p "$OUT/aborted"
    [ -d "$OUT/boots/$name" ] && mv "$OUT/boots/$name" "$OUT/aborted/$name"
    for f in "$OUT/boots/$name".*; do [ -e "$f" ] && mv "$f" "$OUT/aborted/"; done
    echo "== $name cut (rc=$rc); moved to $OUT/aborted/$name" | tee -a "$OUT/steps.txt"
    break
  fi
  if [ "$rc" = 0 ]; then
    ok_boots+=("$name")
    case $name in
      main-*)  # the measured main boot replaces the board-power bound; later boots keep the slowest measured
        p=$(( dur * 11 / 10 + 60 ))
        if [ "$main_measured" = 0 ] || [ "$p" -gt "$main_proj" ]; then main_proj=$p; fi
        main_measured=1 ;;
    esac
  fi
  # A boot that failed for another reason stays in place and is read as it reads (the registered chain went on).
done
[ "$MODE" = timed ] && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv > "$OUT/box-end.csv"

complete=1
for spec in "${BOOTS[@]}"; do
  name=${spec%%:*}
  printf '%s\n' "${ok_boots[@]}" | grep -qx "$name" || complete=0
done
[ -e "$OUT/THERMAL-STOP.txt" ] && complete=0
KITSUM=$(sha256sum KIT-MANIFEST.sha256 | cut -c1-16)
{
  echo "part=$PART complete=$complete kit=$KITSUM"
  echo "card=$CARD uuid=$(trim "$C_UUID") power_limit_w=$(trim "$C_PL")"
  echo "boots_ok=${ok_boots[*]}"
} > "$OUT/PART.txt"

if [ "$PART" = 2 ] && [ "$complete" = 1 ]; then
  COMBINED=$KIT/out-combined-$STAMP
  mkdir -p "$COMBINED/boots"
  cp -a "$P1/boots/." "$COMBINED/boots/"
  cp -a "$OUT/boots/." "$COMBINED/boots/"
  cat "$P1/run.log" "$OUT/run.log" > "$COMBINED/run.log"
  cp "$P1/card.json" "$COMBINED/part1-card.json"; cp "$OUT/card.json" "$COMBINED/part2-card.json"
  cp "$P1/VERDICT.txt" "$COMBINED/part1-VERDICT.txt" 2>/dev/null
  cp "$P1/telemetry.csv" "$COMBINED/part1-telemetry.csv"; cp "$OUT/telemetry.csv" "$COMBINED/part2-telemetry.csv"
  cp SOURCE_SHA "$COMBINED/source.txt"
  { echo "part=combined complete=1 kit=$KITSUM"; echo "part1=$P1"; echo "part2=$OUT"; } > "$COMBINED/PART.txt"
fi
exit 0
