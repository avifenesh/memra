#!/usr/bin/env bash
# Post-result mechanism diagnostic. Its times never enter the fixed-C verdict.
set -euo pipefail

study_root=${STUDY_ROOT:?set STUDY_ROOT to the pinned Qwen research directory}
cd "$study_root"
mkdir -p phase-diagnostic
finish() {
    rc=$?
    if [[ -n ${monitor:-} ]]; then
        kill "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
    fi
    printf "exit=%s utc=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > phase-diagnostic/exit.txt
}
trap finish EXIT
grep -q '"status": "completed"' heldout-pair-v2/status.json

for cycle in 0 1; do
    if [[ $cycle == 0 ]]; then
        order=(off prob cut)
    else
        order=(cut prob off)
    fi
    for arm in "${order[@]}"; do
        case "$arm" in
            off) pmin=0 ;;
            prob) pmin=0.00000001 ;;
            cut) pmin=0.15 ;;
        esac
        name="r${cycle}-${arm}"
        active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        if [[ -n $active ]]; then
            echo "another GPU process is active" >&2
            exit 1
        fi
        printf '%s\n' \
            "model=models/qwen/target.gguf" \
            "binary=binaries-confidence/qwen-prefix-study" \
            "workload=heldout-workloads-v2/qwen-c-heldout-scenario-0.txt" \
            "arm=fixed:3" "seed=20760000" "max_new=512" "ctx=32768" \
            "temperature=0.7" "top_k=20" "top_p=0.95" \
            "MEMRA_SPEC_PMIN=$pmin" "MEMRA_SPEC_PMIN0=0" \
            "MEMRA_SPEC_PHASE=1" "MEMRA_SPEC_PHASE_SYNC=1" \
            > "phase-diagnostic/$name.command.txt"
        nvidia-smi \
            --query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm \
            --format=csv --loop-ms=250 \
            > "phase-diagnostic/$name.gpu.csv" 2>&1 &
        monitor=$!
        rc=0
        MEMRA_SPEC_ADAPT=0 MEMRA_SPEC_ADAPT_FLOOR=1 MEMRA_SPEC_CAPMAX=7 \
            MEMRA_SPEC_PMIN="$pmin" MEMRA_SPEC_PMIN0=0 \
            MEMRA_SPEC_PMIN_INROUND=0 MEMRA_SPEC_STATS=1 \
            MEMRA_SPEC_PHASE=1 MEMRA_SPEC_PHASE_SYNC=1 \
            timeout 2400 binaries-confidence/qwen-prefix-study \
                models/qwen/target.gguf embedded \
                heldout-workloads-v2/qwen-c-heldout-scenario-0.txt \
                "phase-diagnostic/$name" fixed:3 20760000 512 32768 0.7 \
                > "phase-diagnostic/$name.log" 2>&1 || rc=$?
        kill "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
        unset monitor
        printf '%s\n' "$rc" > "phase-diagnostic/$name.exit"
        if (( rc != 0 )); then
            tail -25 "phase-diagnostic/$name.log" >&2
            exit "$rc"
        fi
        echo "PHASE_RUN_COMPLETE $name"
    done
done
