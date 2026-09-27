#!/usr/bin/env bash
# Day 86 section 1c: the load that brings a 9950X host to DAY86 section 1b's condition before the cell slow86, when
# the lanes have no other sitting queued there. Not a scored cell and it decides nothing: door runs of run-gen-i22
# (the door's 16 GiB pinned pool and the 35B, the argv of the cells) back to back until D86_LOAD_HOURS have passed,
# each run under /tmp/memra-gpu.lock taken per run (bounded wait; another lane's sitting may take the card between
# runs, and never signals a holder), each run's gen-only seconds and /proc/vmstat compaction deltas logged, so the log
# also shows when the host's door runs turn slow. The runner is the CPU cap the box drivers use.
# usage: D86_LOAD_HOURS=<h> bash day86-load.sh   (builds run-gen-i22 from 4b378a064 into its own receipts dir first)
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
WT=/root/wt-c
BWT=/root/wt-c-build
R=/root/spill-receipts/c-day86-load
ART=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
LOCK=/tmp/memra-gpu.lock
L=$WT/research/spill-c-20260919
: "${D86_LOAD_HOURS:?set D86_LOAD_HOURS to the hours of load}"
mkdir -p "$R/ev"
printf '*.log -whitespace\n*.tsv -whitespace\n*.txt -whitespace\n' > "$R/.gitattributes"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/load-driver.log"; }
{ echo "load start $(date -u +%FT%TZ) tree $(git -C "$WT" rev-parse HEAD) hours=$D86_LOAD_HOURS"; uptime
  lscpu | grep -E 'Model name|^CPU\(s\)'; grep -E 'MemTotal|MemAvailable' /proc/meminfo; } > "$R/provenance.log" 2>&1
if [ ! -x "$R/bins/run-gen-i22" ]; then
    bash "$L/day63-box-build.sh" "$BWT" "$R" i22=4b378a064 2>&1 | tee -a "$R/load-driver.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "build failed, stopping"; exit 1; }
fi
if systemd-run --scope -q -p CPUQuota=1200% true 2>/dev/null; then cap=(systemd-run --scope -q -p CPUQuota=1200%)
else cap=(taskset -c 0-11); fi
vm() { grep -E '^(compact_isolated|compact_fail|compact_stall|pgmigrate_fail|pgmigrate_success) ' /proc/vmstat | awk '{printf "%s%s=%s", (NR > 1 ? " " : ""), $1, $2}'; }
end=$(( $(date +%s) + $(awk -v h="$D86_LOAD_HOURS" 'BEGIN {printf "%d", h * 3600}') ))
printf 'label\tstart\tend\trc\tgen_s\tvmstat_before\tvmstat_after\n' > "$R/runs.tsv"
n=0
while [ "$(date +%s)" -lt "$end" ]; do
    n=$((n + 1)); label=load-r$n
    exec 9> "$LOCK"
    if ! flock -w 1800 9; then log "$label: lock not taken in 1800 s, retrying"; exec 9>&-; continue; fi
    started=$(date -u +%FT%TZ); before=$(vm)
    "${cap[@]}" env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986 "$R/bins/run-gen-i22" "$ART" 55 88 13 \
        --experts-via-tier --expert-bank-host-bytes=17179869184 > "$R/ev/current.log" 2>&1
    rc=$?
    after=$(vm); ended=$(date -u +%FT%TZ)
    exec 9>&-
    gen=$(grep -oE 'generated 32 tokens in [0-9.]+s' "$R/ev/current.log" | grep -oE '[0-9.]+' | tail -1)
    # The full log (22077 trace lines) is kept for a failed run and for every 50th run; the others keep their
    # summary lines only, so hours of load stay small.
    if [ "$rc" -ne 0 ] || [ $((n % 50)) -eq 1 ]; then
        mv "$R/ev/current.log" "$R/ev/$label.log"
    else
        grep -vE '\[expert-host-slru\] key=' "$R/ev/current.log" | grep -E 'MATCH|generated|STEADY-STATE window|fill complete|error|panicked' > "$R/ev/$label.sum"
        rm -f "$R/ev/current.log"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$started" "$ended" "$rc" "${gen:--}" "$before" "$after" >> "$R/runs.tsv"
    [ "$rc" -eq 0 ] || log "$label rc=$rc"
done
log "load done: $n runs; $(awk -F'\t' 'NR > 1 && $4 != 0' "$R/runs.tsv" | wc -l) failed"
