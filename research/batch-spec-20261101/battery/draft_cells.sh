#!/usr/bin/env bash
# Pooled rows draft exactness + speed cells (decode-batch-bench --pooled-draft), q9 + owntrim.
# usage: draft_cells.sh BENCH OUTDIR
set -u
BENCH=$1
OUT=$2
mkdir -p "$OUT"
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
sha256sum "$BENCH" > "$OUT/bin.sha256"
cell() {
    local name=$1
    shift
    echo "== $name: $*" | tee -a "$OUT/cells.txt"
    env MEMRA_MTP_DRAFT="$DRAFT" "$@" > "$OUT/$name.log" 2>&1
    echo "$name rc=$?" | tee -a "$OUT/cells.txt"
    grep -E "ROWS-STEP|ROUND-DRAFT|DRAFT-TIME|member|session" "$OUT/$name.log" | tee -a "$OUT/cells.txt"
}
cell greedy-ragged "$BENCH" "$Q9" --pooled-draft 3,3,2,4 --ctx 512 --reps 5
cell mixed-ragged "$BENCH" "$Q9" --pooled-draft 3,2,4,3 --ctx 512 --reps 5 --temp 0.8
cell sampled-pmin env MEMRA_SPEC_PMIN=0.3 "$BENCH" "$Q9" --pooled-draft 4,4,3,2,4,4,3,1 --ctx 512 --reps 5 --temp 0.8 --all-sampled
cell greedy-pmin env MEMRA_SPEC_PMIN=0.5 "$BENCH" "$Q9" --pooled-draft 4,4,4,4,3,3,2,2 --ctx 512 --reps 5
cell filtered "$BENCH" "$Q9" --pooled-draft 3,3,3,3 --ctx 512 --reps 5 --temp 0.8 --top-p 0.9 --all-sampled
cell penalized "$BENCH" "$Q9" --pooled-draft 3,3,3,3 --ctx 512 --reps 5 --temp 0.8 --pen --all-sampled
cell eager-greedy env MEMRA_SPEC_NOGRAPH=1 "$BENCH" "$Q9" --pooled-draft 3,2,4,3 --ctx 512 --reps 5
cell eager-sampled env MEMRA_SPEC_NOGRAPH=1 "$BENCH" "$Q9" --pooled-draft 3,2,4,3 --ctx 512 --reps 5 --temp 0.8 --all-sampled
cell b2-k3 "$BENCH" "$Q9" --pooled-draft 3,3 --ctx 2048 --reps 5 --temp 0.8
cell b4-k3 "$BENCH" "$Q9" --pooled-draft 3,3,3,3 --ctx 2048 --reps 5 --temp 0.8
cell b8-k3 "$BENCH" "$Q9" --pooled-draft 3,3,3,3,3,3,3,3 --ctx 2048 --reps 5 --temp 0.8
cell b16-mixed "$BENCH" "$Q9" --pooled-draft 3,3,3,3,3,3,3,3,4,2,3,1,3,3,3,3 --ctx 256 --reps 5 --temp 0.8
echo "cells done" | tee -a "$OUT/cells.txt"
