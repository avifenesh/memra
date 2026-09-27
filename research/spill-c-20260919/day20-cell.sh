#!/usr/bin/env bash
# Day 20 cell (lane/spill-c-20260919, research/spill-c-20260919/DAY20.md, pre-registered before any run).
# One collector lock hold (tools/tier-battery.py --external-lock passes the lock fd as $2).
# Cell tapladder20 (before) / tapladder20b (after): the standalone DFlash2 entry point
# (`dspark_q38_gate`, `HybridModel::generate_spec_dspark`) on the Qwen3.8-27B NVFP4/Q5K MTP GGUF with
# the q38 DFlash2 export, one word-list prompt per rung (day20-prompt.py, ascending), ngen $D20_NGEN,
# MEMRA_SPEC_STATS=1 (the acceptance line) and MEMRA_ALLOC_TRACE=1 (the allocation trace memra#365
# names; `[alloc-trace] <bytes> bytes from dflash.rs:<line>` is the tap receipt, `[dflash-taps]` the
# bounded one). Every rung is its own process; a refusal or OOM exits on its own and the next rung runs.
# Environment (set by the driver, never hosts or ids): D20_R receipts root, D20_OUT the collector's --out
# dir (ev/ lands inside it), D20_BINS binary dir, D20_TREE worktree, D20_LOCK the rig lock path, D20_GGUF
# the target, D20_DRAFT the drafter export dir, D20_WORDS the rung word counts, D20_NGEN, D20_SCRATCH the
# prompt scratch dir (cleaned by the driver).
# usage: day20-cell.sh tapladder20|tapladder20b <lockfd>
set -uo pipefail
cell=$1; fd=$2
: "${D20_R:?}" "${D20_BINS:?}" "${D20_TREE:?}" "${D20_LOCK:?}" "${D20_GGUF:?}" "${D20_DRAFT:?}" "${D20_WORDS:?}" "${D20_SCRATCH:?}"
NGEN=${D20_NGEN:-32}
EV=${D20_OUT:-$D20_R/$cell}/ev
mkdir -p "$EV" "$D20_SCRATCH"
cd "$D20_TREE" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$D20_LOCK" --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
sha256sum "$D20_BINS/dspark_q38_gate" | tee "$EV/binary.sha256"
sha256sum "$D20_GGUF" "$D20_DRAFT/config.json" "$D20_DRAFT/model.safetensors" | tee "$EV/artifact.sha256"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
: > "$EV/marks.tsv"
case $cell in tapladder20|tapladder20b) ;; *) echo "unknown cell $cell"; exit 2 ;; esac
for words in $D20_WORDS; do
    pd=$D20_SCRATCH/w$words; mkdir -p "$pd"
    python3 research/spill-c-20260919/day20-prompt.py "$words" "$pd/prompt.txt" | tee -a "$EV/prompts.txt"
    sha256sum "$pd/prompt.txt" >> "$EV/prompts.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/w$words.smi-before" 2>&1
    mark "w$words start"
    env CUDA_VISIBLE_DEVICES=0 MEMRA_PROMPT_DIR="$pd" MEMRA_SPEC_STATS=1 MEMRA_ALLOC_TRACE=1 \
        timeout 1500 "$D20_BINS/dspark_q38_gate" "$D20_GGUF" "$D20_DRAFT" "$NGEN" > "$EV/w$words.log" 2>&1
    rc=$?; echo "$rc" > "$EV/w$words.exit"
    mark "w$words exit rc=$rc"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/w$words.smi-after" 2>&1
    # the receipt lines, extracted (the full log stays)
    grep -E "^\[dspark-q38-gate\]|^\[dspark-q38\] acceptance|^\[dflash-taps\]|EXACT|DIVERGED|dspark_q38_gate:|out of memory|OUT_OF_MEMORY|Error|error|refused|panicked" "$EV/w$words.log" | head -40 > "$EV/w$words.receipt"
    # the tap allocation line: the largest allocation traced from dflash.rs
    grep -E "^\[alloc-trace\] [0-9]+ bytes from .*dflash.rs" "$EV/w$words.log" | sort -t' ' -k2 -n -r | head -3 >> "$EV/w$words.receipt"
done
echo "$cell cell done"
