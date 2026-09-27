#!/usr/bin/env bash
# Day 65: the cell `pin` (DAY65.md section 1, OWED C12) on a host with two or more L3 domains (the Ryzen 9 9950X
# class), one RTX PRO 6000 Blackwell Workstation Edition. In order: provenance (tree, the artifact's SHA-256 against
# the registered value, the card, CPU and memory), the builds (day63-box-build.sh in a separate build worktree, labels
# c60 and i15), then the cell through the day-40 runner under /tmp/memra-gpu.lock (bounded waits, never signals a
# holder), the collector's --validate and the registered reader, both teed. The cell pins each run to its arm's 12
# CPUs, which is the run's CPU budget; the runner is not pinned. A rerun skips a cell with a .done marker. No host,
# id or price here.
# usage: D65_BUILDS="c60=<sha> i15=<sha>" day65-box.sh   (paths below; the lead stages the artifact and the two
# worktrees)
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
WT=/root/wt-c                     # the sitting tree: lane/spill-c-20260919 at its tip
BWT=/root/wt-c-build              # `git -C /root/wt-c worktree add --detach /root/wt-c-build`
R=/root/spill-receipts/c-day65
ART=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ART_SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
L=$WT/research/spill-c-20260919
: "${D65_BUILDS:?set D65_BUILDS to the label=sha list of DAY65 section 1}"
mkdir -p "$R"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n*.json -whitespace\n' > "$R/.gitattributes"
{
    echo "box start $(date -u +%FT%TZ) tree $(git -C "$WT" rev-parse HEAD)"
    git -C "$WT" status --porcelain --untracked-files=no | head
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
    nvidia-smi -q -d PERFORMANCE | sed -n '1,40p'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    nproc; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket'
    grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo
    df -h "$(dirname "$ART")" "$R" | tail -n +1
} 2>&1 | tee "$R/provenance.log"
printf '%s  %s\n' "$ART_SHA" "$ART" > "$ART.sha256"
sha256sum -c "$ART.sha256" 2>&1 | tee -a "$R/provenance.log"
grep -q ': FAILED' "$R/provenance.log" && { echo "ARTIFACT MISMATCH, stopping" | tee -a "$R/provenance.log"; exit 1; }

need=()
for spec in $D65_BUILDS; do
    [ -x "$R/bins/run-gen-${spec%%=*}" ] || need+=("$spec")
done
if [ ${#need[@]} -gt 0 ]; then
    bash "$L/day63-box-build.sh" "$BWT" "$R" "${need[@]}" 2>&1 | tee -a "$R/box-driver.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
fi
for b in run-gen-c60 run-gen-i15; do
    [ -x "$R/bins/$b" ] || { echo "missing $b, stopping" | tee -a "$R/box-driver.log"; exit 1; }
done

echo "cpu cap: each run pinned by the cell to its arm's 12 CPUs (DAY65 section 1); the runner unpinned" | tee -a "$R/provenance.log"

export D40_RIG=pro-single D40_R=$R D40_TREE=$WT D40_BINS=$R/bins D40_ART=$ART
export D40_LOCK=/tmp/memra-gpu.lock D40_MIN_AVAIL_GB=48
cell_run() { # $1 cell  $2 script  $3 timeout
    local cell=$1 script=$2 to=$3 out n
    [ -f "$R/$cell.done" ] && { echo "$cell already done"; return 0; }
    env D40_CELL_SCRIPT="$L/$script" bash "$L/day40-run-cell.sh" "$cell" "$to"
    echo "$cell rc=$? $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
    out=$R/$cell
    for n in $(seq 30 -1 1); do [ -f "$R/$cell-retry$n/CELL.jsonl" ] && { out=$R/$cell-retry$n; break; }; done
    python3 "$WT/tools/tier-battery.py" --rig pro-single --validate "$out" > "$R/$cell-validate.log" 2>&1
    echo "$cell validate rc=$?" | tee -a "$R/box-driver.log"
    mkdir -p "$R/$cell"
    case $cell in
        pin) python3 "$L/day65-read.py" "$R/$cell" --rig pro-single ;;
    esac > "$R/$cell/reading.log" 2>&1
    echo "$cell reader rc=$?" | tee -a "$R/box-driver.log"
    touch "$R/$cell.done"
}
cell_run pin day65-cell.sh 5400
echo "box done $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
