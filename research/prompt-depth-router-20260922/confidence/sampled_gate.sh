#!/usr/bin/env bash
set -euo pipefail

study_root=${STUDY_ROOT:?set STUDY_ROOT to the pinned Qwen research directory}
cd "$study_root"
trap 'rc=$?; printf "exit=%s utc=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > sampled-gate.exit; exit "$rc"' EXIT
test -f job-v2.exit
grep -q '^exit=0 ' job-v2.exit

MEMRA_CHAT=1 MEMRA_PROMPT_FILE="$study_root/oracle-code-prompt.txt" \
    MEMRA_NGEN=128 MEMRA_SPEC_K=3 MEMRA_SPEC_TEMP=0 \
    MEMRA_SPEC_ADAPT=0 MEMRA_SPEC_PMIN=0.15 MEMRA_SPEC_PMIN0=0 \
    binaries-confidence/run-spec models/qwen/target.gguf \
    > oracle-v2-c015.log 2>&1
grep -q '^=== SELF-CONSISTENCY PASS ===$' oracle-v2-c015.log
echo ORACLE_PASS c015

for arm in off c015 c030 c030zero; do
    case "$arm" in
        off) pmin=0; pmin0=0 ;;
        c015) pmin=0.15; pmin0=0 ;;
        c030) pmin=0.30; pmin0=0 ;;
        c030zero) pmin=0.30; pmin0=1 ;;
    esac
    MEMRA_CHAT=1 MEMRA_PROMPT_FILE="$study_root/oracle-code-prompt.txt" \
        MEMRA_NGEN=128 MEMRA_SPEC_K=3 MEMRA_SPEC_TEMP=0.7 \
        MEMRA_TOP_K=20 MEMRA_TOP_P=0.95 MEMRA_SEED=20740010 \
        MEMRA_SPEC_ADAPT=0 MEMRA_SPEC_PMIN="$pmin" MEMRA_SPEC_PMIN0="$pmin0" \
        MEMRA_SPEC_PMIN_INROUND=0 \
        binaries-confidence/run-spec models/qwen/target.gguf \
        > "sampled-oracle-$arm.log" 2>&1
    grep -q '^=== SELF-CONSISTENCY PASS ===$' "sampled-oracle-$arm.log"
    grep -q 'PASS (seeded rerun identical)' "sampled-oracle-$arm.log"
    printf "SAMPLED_REPRO_PASS %s\n" "$arm"
done
