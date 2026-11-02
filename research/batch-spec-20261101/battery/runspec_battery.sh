#!/usr/bin/env bash
# run-spec battery for the round split. usage: runspec_battery.sh RUN_SPEC_BIN OUTDIR
# Cells (q9 + owntrim draft, probe prompt, NGEN=128):
#   greedy     K=1..8 self-consistency vs the plain target
#   sampled    K=1..8 seeded reproducibility, T=0.8 seed 1234 (prints the sampled streams)
#   adapt      K=4 greedy, MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 (adaptive K,
#              p-min cut, zero-draft rounds / zero-round fold)
#   replay     K=3 greedy, MEMRA_SPEC_REPLAY=1 (legacy replay arm)
set -u
BIN=$1
OUT=$2
mkdir -p "$OUT"
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
PROMPT=$HOME/.cache/memra-batchspec/tools/fast-gate/prompts/probe.txt
unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY MEMRA_SPEC_TEMP MEMRA_SEED
run() {
    local name=$1
    shift
    env MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PROMPT" "$@" \
        timeout 1500 "$BIN" "$Q9" > "$OUT/$name.log" 2>&1
    echo "$name rc=$?" | tee -a "$OUT/rc.txt"
}
run greedy MEMRA_SPEC_TEMP=0
run sampled MEMRA_SPEC_TEMP=0.8 MEMRA_SEED=1234
run adapt MEMRA_SPEC_TEMP=0 MEMRA_SPEC_K=4 MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1
run replay MEMRA_SPEC_TEMP=0 MEMRA_SPEC_K=3 MEMRA_SPEC_REPLAY=1
