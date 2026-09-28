#!/usr/bin/env bash
# Day 94 (DAY94.md section 1, day89-box.sh for the cell where285, receipts c-day94): where the 285K class's gen-only gap sits (the cell where285; after DAY88.md section 5's cells promo, promo-res and promo-spec, the MoE slot cache
# door's promotion, phase 1), one RTX PRO 6000 Blackwell Workstation Edition, on the 285K class and on a 9950X. In order: provenance (tree, the artifact's SHA-256 against the registered
# value, the card, CPU and memory), the builds (day88-box-build.sh in a separate build worktree, labels i24 and i25, run-gen only),
# then the three cells (day89-cell.sh), one collector hold under
# /tmp/memra-gpu.lock through the day-40 runner (bounded waits, never signals a holder), the collector's --validate
# and the registered reader, both teed. The runner goes through the CPU cap (a 1200% systemd scope; where the box has
# no systemd, it is pinned to 12 cores with taskset; which one ran is recorded). A rerun skips a cell with a .done
# marker. No host, id or price here.
# No profiler in these cells. After the three cells, the reader (day88-read.py) reads them together into
# $R/reading.log.
# usage: D94_BUILDS="i24=<sha> i25=<sha>" day94-box.sh   (paths below; the lead stages the artifact and the two worktrees)
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
WT=/root/wt-c                     # the sitting tree: lane/spill-c-20260919 at its tip
BWT=/root/wt-c-build              # `git -C /root/wt-c worktree add --detach /root/wt-c-build`
R=/root/spill-receipts/c-day94
ART=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
ART_SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
L=$WT/research/spill-c-20260919
: "${D94_BUILDS:?set D94_BUILDS to the label=sha list of DAY94 section 1}"
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
for spec in $D94_BUILDS; do
    [ -x "$R/bins/run-gen-${spec%%=*}" ] || need+=("$spec")
done
if [ ${#need[@]} -gt 0 ]; then
    bash "$L/day88-box-build.sh" "$BWT" "$R" "${need[@]}" 2>&1 | tee -a "$R/box-driver.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
fi
for b in run-gen-i24 run-gen-i25; do
    [ -x "$R/bins/$b" ] || { echo "missing $b, stopping" | tee -a "$R/box-driver.log"; exit 1; }
done

if systemd-run --scope -q -p CPUQuota=1200% true 2>/dev/null; then
    cap=(systemd-run --scope -q -p CPUQuota=1200%); echo "cpu cap: systemd scope CPUQuota=1200%" | tee -a "$R/provenance.log"
else
    cap=(taskset -c 0-11); echo "cpu cap: taskset -c 0-11 (no systemd scope on this box)" | tee -a "$R/provenance.log"
fi

export D40_RIG=pro-single D40_R=$R D40_TREE=$WT D40_BINS=$R/bins D40_ART=$ART
export D40_LOCK=/tmp/memra-gpu.lock D40_MIN_AVAIL_GB=48
cell_run() { # $1 cell  $2 script  $3 timeout
    local cell=$1 script=$2 to=$3 out n
    [ -f "$R/$cell.done" ] && { echo "$cell already done"; return 0; }
    "${cap[@]}" env D40_CELL_SCRIPT="$L/$script" bash "$L/day40-run-cell.sh" "$cell" "$to"
    echo "$cell rc=$? $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
    out=$R/$cell
    for n in $(seq 30 -1 1); do [ -f "$R/$cell-retry$n/CELL.jsonl" ] && { out=$R/$cell-retry$n; break; }; done
    python3 "$WT/tools/tier-battery.py" --rig pro-single --validate "$out" > "$R/$cell-validate.log" 2>&1
    echo "$cell validate rc=$?" | tee -a "$R/box-driver.log"
    mkdir -p "$R/$cell"
    touch "$R/$cell.done"
}
# DAY94 section 1: the host's core types. The kernel's hybrid PMU lists first; else the CPUs with the highest max
# MHz (lscpu). The cell needs a P-core list distinct from the wide list, or it does not run.
{ echo "cpu_core=$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo absent)"
  echo "cpu_atom=$(cat /sys/devices/cpu_atom/cpus 2>/dev/null || echo absent)"
  lscpu -e=CPU,CORE,MAXMHZ,MINMHZ 2>&1; } | tee "$R/cores.log"
pcores=$(cat /sys/devices/cpu_core/cpus 2>/dev/null)
if [ -z "$pcores" ]; then
    pcores=$(lscpu -e=CPU,MAXMHZ --noheadings 2>/dev/null | awk '{print $2, $1}' | sort -rn \
        | awk 'NR == 1 {m = $1} $1 == m {print $2}' | sort -n | paste -sd, -)
    echo "pcores from lscpu max MHz: $pcores" | tee -a "$R/cores.log"
fi
if [ "${cap[0]}" = taskset ]; then wide=0-11; else wide=0-$(($(nproc) - 1)); fi
echo "wide=$wide pcores=$pcores" | tee -a "$R/cores.log" "$R/provenance.log"
expand() { # a CPU list (0-3,8) as sorted numbers
    python3 -c 'import sys
s = set()
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-")
    s.update(range(int(a), int(b or a) + 1))
print(" ".join(map(str, sorted(s))))' "$1"
}
if [ -z "$pcores" ] || [ "$(expand "$pcores")" = "$(expand "$wide")" ]; then
    echo "NO P-CORE LIST DISTINCT FROM THE WIDE ONE, stopping" | tee -a "$R/box-driver.log"; exit 1
fi
export D94_WIDE=$wide D94_PCORES=$pcores
cell_run where285 day94-cell.sh 9000
python3 "$L/day94-read.py" "$R" --rig pro-single > "$R/reading.log" 2>&1
echo "reader rc=$?" | tee -a "$R/box-driver.log"
echo "box done $(date -u +%FT%TZ)" | tee -a "$R/box-driver.log"
