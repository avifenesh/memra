#!/usr/bin/env bash
# moe-prefetch on the Max-Q: DAY59's deciding cell for MEMRA_MOE_PREFETCH=1 (research/spill-c-20260919/DAY59.md
# section 1; PREREG.md beside this file) replayed on ONE RTX PRO 6000 Blackwell Max-Q Workstation Edition of <box>.
# Run by the session that holds the window, nothing else on the box:
#   CARD=<idx> bash /mnt/Data/work/memra-maxq/moe-prefetch/run.sh
# It replaces day61-box.sh (the Workstation sitting driver): the same provenance, artifact SHA check and the four
# DAY59 cells (pfgates, pfserve, pftime, pfnaked) through the same day-40 runner and collector, one collector hold of
# /tmp/memra-gpu.lock per cell, with the prebuilt c60 binaries, no git and no build on the box. Every deviation from
# the registration is in DEVIATIONS.md. Writes out-<UTC stamp>/ (raw logs, 250 ms telemetry of that card only,
# readings, regime.txt, VERDICT.txt, DONE) and prints the out dir last.
# Exit: 0 a verdict was reached (qualifies or does-not-qualify), 1 void, 2 usage, 3 refused before any cell.
set -uo pipefail
ROOT=${MAXQ_ROOT:-/mnt/Data/work/memra-maxq/moe-prefetch}
MODELS=${MAXQ_MODELS:-/mnt/Data/work/memra-maxq/models}
TOTAL_S=${MAXQ_TOTAL_S:-1200}
: "${CARD:?usage: CARD=<nvidia-smi index> bash $ROOT/run.sh}"
case $CARD in ''|*[!0-9]*) echo "REFUSED: CARD must be one nvidia-smi index, got '$CARD'"; exit 2;; esac
case $TOTAL_S in ''|*[!0-9]*) echo "REFUSED: MAXQ_TOTAL_S must be seconds"; exit 2;; esac

# Hard backstop: the whole run, cells and readers included, ends by TOTAL_S + 120 s whatever state it is in. The
# normal bound is tighter: each cell's collector timeout is cut to the time left before TOTAL_S.
if [ -z "${MAXQ_INNER:-}" ]; then
    exec timeout --kill-after=30 $((TOTAL_S + 120)) env MAXQ_INNER=1 bash "$(readlink -f "$0")" "$@"
fi

MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
ART=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ART_SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
TREE=$ROOT/tree
L=$TREE/research/spill-c-20260919
BINS=$ROOT/bins/c60
LOCK=/tmp/memra-gpu.lock
RIG_LABEL=pro-maxq
SCRATCH=$ROOT/scratch
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/out-$STAMP
start=$(date +%s)
deadline=$((start + TOTAL_S))
mkdir -p "$OUT" "$SCRATCH" || { echo "REFUSED: cannot create $OUT"; exit 3; }
echo "$OUT" > "$ROOT/last-out.txt"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "moe-prefetch Max-Q run start $(date -u +%FT%TZ) card=$CARD total_s=$TOTAL_S out=$OUT"

cleanup() {
    local rc=$?
    # Anything still running from this kit's binaries is this run's (a timed-out or killed cell); stop it.
    if pgrep -f "$BINS/" > /dev/null 2>&1; then
        echo "cleanup: stopping leftover kit processes: $(pgrep -f "$BINS/" | tr '\n' ' ')"
        pkill -TERM -f "$BINS/" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "$BINS/" > /dev/null 2>&1 || break; sleep 1; done
        pkill -KILL -f "$BINS/" 2>/dev/null
    fi
    # Scratch: TMPDIR of every cell process. Literal paths only.
    case $ROOT in
        /mnt/Data/work/memra-maxq/moe-prefetch) rm -rf /mnt/Data/work/memra-maxq/moe-prefetch/scratch ;;
        /home/avifenesh/.cache/memra-maxq/moe-prefetch/test) rm -rf /home/avifenesh/.cache/memra-maxq/moe-prefetch/test/scratch ;;
    esac
    echo "exit=$rc end=$(date -u +%FT%TZ) elapsed=$(( $(date +%s) - start ))s" > "$OUT/DONE"
    echo "moe-prefetch Max-Q run exit=$rc elapsed=$(( $(date +%s) - start ))s"
    echo "$OUT"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

refuse() { echo "REFUSED: $*" | tee -a "$OUT/VERDICT.txt"; exit 3; }

# ---- preflight: the card, the box, the lock, the binaries, the artifact ---------------------------------------------
for t in nvidia-smi python3 sha256sum curl ss flock nice timeout; do
    command -v "$t" > /dev/null 2>&1 || refuse "missing tool $t"
done
name=$(nvidia-smi -i "$CARD" --query-gpu=name --format=csv,noheader 2>&1)
[ "$name" = "$MAXQ_NAME" ] || refuse "card $CARD reads '$name', not '$MAXQ_NAME'"
used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 | tr -d ' ')
case $used in ''|*[!0-9]*) refuse "card $CARD memory.used unreadable: '$used'";; esac
[ "$used" -lt 2000 ] || refuse "card $CARD uses $used MiB; the cell needs it under 2000 MiB"
card_apps=$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "$card_apps" = 0 ] || refuse "card $CARD has $card_apps compute app(s)"
# AGENTS.md (evidence discipline): one scored campaign per multi-card box; the timed cells need the other cards quiet.
box_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
box_note() { { echo "$(date -u +%FT%TZ) $1: box compute apps=$(box_apps)"
               nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>&1
               nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader 2>&1; } >> "$OUT/box-state.log"; }
box_note preflight
if [ "$(box_apps)" != 0 ] && [ "${MAXQ_ALLOW_BUSY_BOX:-0}" != 1 ]; then
    refuse "the box has $(box_apps) compute app(s) on other cards (box-state.log); the timed cells need a quiet box"
fi
exec 8>> "$LOCK" || refuse "cannot open $LOCK"
flock -n 8 || refuse "$LOCK is held by another job"
flock -u 8
exec 8>&-
MIN_AVAIL_GB=${MAXQ_MIN_AVAIL_GB:-48}   # the registered D40_MIN_AVAIL_GB (day61-box.sh:56); the knob exists for the laptop plumbing test
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
[ "$avail_kb" -ge $((MIN_AVAIL_GB * 1024 * 1024)) ] || refuse "MemAvailable ${avail_kb} kB is under the cell's ${MIN_AVAIL_GB} GiB"
( cd "$BINS" && sha256sum -c bins.sha256 ) > "$OUT/bins-check.log" 2>&1 || { cat "$OUT/bins-check.log"; refuse "kit binaries do not match bins.sha256"; }
for b in run-gen-c60 run-spec-c60 memra-server-c60; do [ -x "$BINS/$b" ] || refuse "missing $BINS/$b"; done

{
    echo "box start $(date -u +%FT%TZ) tree $(cat "$TREE/SOURCE_SHA") bins $(cat "$BINS/BUILD_SHA") card $CARD rig $RIG_LABEL"
    echo "kit: PREREG.md sha256 $(sha256sum "$ROOT/PREREG.md" | cut -d' ' -f1)  run.sh sha256 $(sha256sum "$ROOT/run.sh" | cut -d' ' -f1)"
    cat "$OUT/bins-check.log"
    nvidia-smi -i "$CARD" --query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,power.default_limit,power.max_limit,clocks.max.sm --format=csv
    echo "all cards:"
    nvidia-smi --query-gpu=index,name,power.limit,memory.used,temperature.gpu --format=csv
    nvidia-smi -i "$CARD" -q -d PERFORMANCE | sed -n '1,40p'
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader
    nproc; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket'
    grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo
    df -h "$MODELS" "$ROOT" | tail -n +1
    uname -r; python3 --version
    echo "cpu cap: nice -n 10 (no core pinning; DEVIATIONS.md)"
} 2>&1 | tee "$OUT/provenance.log"
plim_start=$(nvidia-smi -i "$CARD" --query-gpu=power.limit --format=csv,noheader 2>&1 | tr -d ' ')

printf '%s  %s\n' "$ART_SHA" "$ART" > "$ART.sha256"
t0=$(date +%s)
nice -n 10 sha256sum -c "$ART.sha256" > "$OUT/artifact-check.log" 2>&1
art_rc=$?
cat "$OUT/artifact-check.log" >> "$OUT/provenance.log"; cat "$OUT/artifact-check.log"
echo "artifact sha256 check rc=$art_rc took $(( $(date +%s) - t0 ))s" | tee -a "$OUT/provenance.log"
[ "$art_rc" = 0 ] && grep -qxF "$ART: OK" "$OUT/artifact-check.log" || refuse "ARTIFACT MISMATCH or missing ($ART)"

# ---- the cells (day61-box.sh lines 55-80, ported) --------------------------------------------------------------------
export TMPDIR=$SCRATCH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD D40_CARD=$CARD
export D40_RIG=pro-single D40_R=$OUT D40_TREE=$TREE D40_BINS=$BINS D40_ART=$ART
export D40_LOCK=$LOCK D40_MIN_AVAIL_GB=$MIN_AVAIL_GB
export D40_WAITS=6 D40_POLL_S=10 D40_LOCK_TRIES=3
cap=(nice -n 10)
cell_run() { # $1 cell  $2 script  $3 cap_s  [$4 timed]
    local cell=$1 script=$2 cap_s=$3 timed=${4:-} out n left to
    [ -f "$OUT/$cell.done" ] && { echo "$cell already done"; return 0; }
    left=$(( deadline - $(date +%s) - 20 ))
    if [ "$left" -lt 60 ]; then
        echo "$cell SKIPPED: ${left}s left before the run deadline" | tee -a "$OUT/box-driver.log"; return 1
    fi
    if [ -n "$timed" ]; then
        box_note "before $cell"
        if [ "$(box_apps)" != 0 ] && [ "${MAXQ_ALLOW_BUSY_BOX:-0}" != 1 ]; then
            echo "$cell SKIPPED: the box is not quiet (box-state.log)" | tee -a "$OUT/box-driver.log"; return 1
        fi
    fi
    to=$(( cap_s < left ? cap_s : left ))
    "${cap[@]}" env D40_CELL_SCRIPT="$L/$script" bash "$L/day40-run-cell.sh" "$cell" "$to"
    echo "$cell rc=$? timeout=${to}s $(date -u +%FT%TZ)" | tee -a "$OUT/box-driver.log"
    out=$OUT/$cell
    for n in $(seq 30 -1 1); do [ -f "$OUT/$cell-retry$n/CELL.jsonl" ] && { out=$OUT/$cell-retry$n; break; }; done
    python3 "$TREE/tools/tier-battery.py" --rig pro-single --validate "$out" > "$OUT/$cell-validate.log" 2>&1
    echo "$cell validate rc=$?" | tee -a "$OUT/box-driver.log"
    mkdir -p "$OUT/$cell"
    case $cell in
        pfgates) python3 "$L/day59-pf.py" gates "$OUT/$cell" --rig "$RIG_LABEL" ;;
        pfserve) python3 "$L/day59-pf.py" serve "$OUT/$cell" --rig "$RIG_LABEL" ;;
        pftime|pfnaked) python3 "$L/day59-pf.py" time "$OUT/$cell" --rig "$RIG_LABEL" --shape "$cell" ;;
    esac > "$OUT/$cell/reading.log" 2>&1
    echo "$cell reader rc=$?" | tee -a "$OUT/box-driver.log"
    cat "$OUT/$cell/reading.log"
    if [ -f "$out/command.gpu.csv" ]; then
        { echo "== $cell"; python3 "$L/c-regime.py" "$out" 2>&1; } >> "$OUT/regime.txt"
    fi
    touch "$OUT/$cell.done"
}
# Caps per cell: about 10x the Workstation's wall (42 s, 31 s, 146 s, 139 s), each cut to the time left.
cell_run pfgates day59-cell.sh 420
cell_run pfserve day59-cell.sh 420
cell_run pftime day59-cell.sh 900 timed
cell_run pfnaked day59-cell.sh 900 timed

# ---- the verdict -----------------------------------------------------------------------------------------------------
plim_end=$(nvidia-smi -i "$CARD" --query-gpu=power.limit --format=csv,noheader 2>&1 | tr -d ' ')
echo "power.limit card $CARD: start=$plim_start end=$plim_end" | tee -a "$OUT/provenance.log"
{
    for c in pfgates pfserve pftime pfnaked; do cat "$OUT/$c/reading.log" 2>/dev/null || echo "($c: no reading)"; done
    echo
    cat "$OUT/regime.txt" 2>/dev/null
    echo "power.limit card $CARD: start=$plim_start end=$plim_end"
    echo
    python3 "$L/maxq-verdict.py" "$OUT" --rig "$RIG_LABEL" --card "$CARD" --power-limit "$plim_start"
} > "$OUT/VERDICT.txt" 2>&1
tail -n 1 "$OUT/VERDICT.txt"
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -name run.log ! -name DONE -print0 | sort -z | xargs -0 sha256sum ) > "$OUT/MANIFEST.sha256" 2>&1
case $(tail -n 1 "$OUT/VERDICT.txt") in
    *"-> qualifies"|*"-> does-not-qualify") exit 0 ;;
    *) exit 1 ;;
esac
