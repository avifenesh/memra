#!/usr/bin/env bash
# Day 52: the MoE slot cache door's target-card sitting (DAY52.md), one RTX PRO 6000 Blackwell Workstation Edition.
# In order: provenance (tree, both artifacts' SHA-256 against the registered values, the card, CPU and memory),
# the builds (day52-box-build.sh in a separate build worktree, so the sitting tree stays at the lane tip), then the
# cells, each its own collector hold under /tmp/memra-gpu.lock through the day-40 runner (bounded waits, never
# signals a holder): attrib (rung 0, day40-cell.sh), ladder (day52-cell.sh), hashlock, spec, decide
# (day51-cell.sh). After each cell: the collector's --validate and the registered reader, both teed.
# Every runner goes through the same CPU cap as the 5090 cells (a 1200% systemd scope; where the box has no
# systemd, the runner is pinned to 12 cores with taskset; which one ran is recorded). No host, id or price here.
# usage: day52-box.sh   (paths below; the lead stages the artifacts and the two worktrees)
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
WT=/root/wt-c                     # the sitting tree: lane/spill-c-20260919 at the tip named in DAY52 section 2
BWT=/root/wt-c-build              # `git -C /root/wt-c worktree add --detach /root/wt-c-build`
R=/root/spill-receipts/c-day52
ART=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ART_OTHER=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ART_SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
ART_OTHER_SHA=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
L=$WT/research/spill-c-20260919
: "${D52_BUILDS:?set D52_BUILDS to the label=sha list of DAY52 section 2}"
mkdir -p "$R"
printf '*.log -whitespace\n*.txt -whitespace\n*.snap -whitespace\n' > "$R/.gitattributes"
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
printf '%s  %s\n' "$ART_OTHER_SHA" "$ART_OTHER" > "$ART_OTHER.sha256"
sha256sum -c "$ART.sha256" "$ART_OTHER.sha256" 2>&1 | tee -a "$R/provenance.log"
grep -q ': FAILED' "$R/provenance.log" && { echo "ARTIFACT MISMATCH, stopping" | tee -a "$R/provenance.log"; exit 1; }

if ! grep -q '^BUILDS-DONE' "$R/builds.log" 2>/dev/null; then
    # shellcheck disable=SC2086
    bash "$L/day52-box-build.sh" "$BWT" "$R" $D52_BUILDS || exit 1
fi

if systemd-run --scope -q -p CPUQuota=1200% true 2>/dev/null; then
    cap=(systemd-run --scope -q -p CPUQuota=1200%); echo "cpu cap: systemd scope CPUQuota=1200%" | tee -a "$R/provenance.log"
else
    cap=(taskset -c 0-11); echo "cpu cap: taskset -c 0-11 (no systemd scope on this box)" | tee -a "$R/provenance.log"
fi

export D40_RIG=pro-single D40_R=$R D40_TREE=$WT D40_BINS=$R/bins D40_ART=$ART D40_ART_OTHER=$ART_OTHER
export D40_LOCK=/tmp/memra-gpu.lock D40_MIN_AVAIL_GB=48 D51_MOE_ENV="MEMRA_MOE_RESIDENT=0"
for spec in attrib:day40-cell.sh:7200 ladder:day52-cell.sh:14400 hashlock:day51-cell.sh:1800 \
            spec:day51-cell.sh:3600 decide:day51-cell.sh:3600; do
    IFS=: read -r cell script to <<< "$spec"
    [ -f "$R/$cell.done" ] && { echo "$cell already done"; continue; }
    "${cap[@]}" env D40_CELL_SCRIPT="$L/$script" bash "$L/day40-run-cell.sh" "$cell" "$to"
    echo "$cell rc=$? $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
    # The collector's out dir is $cell, or $cell-retryN after a busy-lock attempt; the cell's ev is always $cell/ev.
    out=$R/$cell
    for n in $(seq 30 -1 1); do [ -f "$R/$cell-retry$n/CELL.jsonl" ] && { out=$R/$cell-retry$n; break; }; done
    python3 "$WT/tools/tier-battery.py" --rig pro-single --validate "$out" > "$R/$cell-validate.log" 2>&1
    echo "$cell validate rc=$?" | tee -a "$R/box-driver.log"
    case $cell in
        attrib) python3 "$L/day40-attrib.py" "$R/$cell" --rig pro-single ;;
        ladder) python3 "$L/day52-views.py" "$R/$cell" --rig pro-single ;;
        *) python3 "$L/day51-decide.py" "$cell" "$R/$cell" --rig pro-single ;;
    esac > "$R/$cell/reading.log" 2>&1
    echo "$cell reader rc=$?" | tee -a "$R/box-driver.log"
    touch "$R/$cell.done"
done
# DAY52 section 4 (OWED C6, DAY53.md): the host-tier failure and identity gates on the verify digest v3 server,
# the 27B, device prefix budget 256 MB (day 23's target-card shape), each gate taking /tmp/memra-gpu.lock itself.
for cell in unit-server failure-default-off failure-plain-off failure-default-on identity-default-off identity-default-on; do
    [ -f "$R/c6-$cell.done" ] && { echo "c6 $cell already done"; continue; }
    MEMRA_GPU_LOCK=/tmp/memra-gpu.lock "${cap[@]}" bash "$L/day53-cell.sh" "$cell" "$ART_OTHER" "$R/bins/memra-server-v3" "$R/c6" 256
    echo "c6 $cell rc=$? $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
    touch "$R/c6-$cell.done"
done
echo "box done $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
