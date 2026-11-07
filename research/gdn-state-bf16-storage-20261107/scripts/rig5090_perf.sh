#!/usr/bin/env bash
# RTX 5090 Laptop measurements for the bf16 GDN state storage lane, same binary, the class
# switched by MEMRA_GDN_STATE_BF16 (0 = fp32, 1 = bf16). Interleaved A B B A A B per cell
# (LAW:interleaved-ab), GPU temperature and SM clock logged per run.
#   - batched decode tick: decode-batch-bench, Qwen3.8-27B, depth 2048 at 1/4/8/16 sessions
#     and depth 12288 at 4 sessions, greedy, token fingerprints per run;
#   - single-stream decode and prefill: run-gen, 128 generated tokens.
set -u
R=$HOME/.cache/memra-gdnbf16-real
NEW=$R/bins/dev1-120a/release
OUT=${1:?outdir}
mkdir -p "$OUT"
G=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/perf.log"; }
therm(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | tr -d ' '; }
dbb() { # tag bf16 seqs depth
  local f=$OUT/dbb-$1.log
  MEMRA_GDN_STATE_BF16=$2 MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1500 "$NEW/decode-batch-bench" "$G" \
    --seqs $3 --chunk $3 --ctx $4 --steps 16 --reps 3 > "$f" 2>&1
  echo "rc=$?" >> "$f"
  log "dbb $1 bf16=$2 s=$3 d=$4 $(grep CHUNKSWEEP "$f" | cut -c1-200) fp=$(grep TOKFP "$f" | sed 's/.*\[//' | md5sum | cut -c1-12) $(grep -m1 -iE 'out of memory|error' "$f" | cut -c1-120) | $(therm)"
}
for cell in 1:2048 4:2048 8:2048 16:2048 4:12288; do
  s=${cell%%:*}; d=${cell#*:}
  i=0
  for arm in 0 1 1 0 0 1; do
    i=$((i + 1)); dbb "s$s-d$d-r$i" $arm $s $d
  done
done
PROMPT="Write a Python function that returns the n-th Fibonacci number, then explain its time complexity in two sentences."
i=0
for arm in 0 1 1 0 0 1; do
  i=$((i + 1)); f=$OUT/rungen-r$i-bf16$arm.log
  MEMRA_GDN_STATE_BF16=$arm MEMRA_CHAT=1 MEMRA_NGEN=128 timeout 900 "$NEW/run-gen" "$G" --prompt "$PROMPT" > "$f" 2>&1
  log "run-gen r$i bf16=$arm $(grep -E '^generated [0-9]+ tokens' "$f" | tail -1) | $(grep -E '^prefill [0-9]+ tok' "$f" | tail -1) | $(therm)"
done
log PERF_DONE
