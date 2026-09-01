#!/usr/bin/env bash
# lane/graph-launch-guard-sweep-20260831: the LEGAL settle for the local-ci --perf red.
#
# THE RED: qwen9b-plain-short 126.97 tok/s vs cross-day median 139.31 (-8.86%), on a
# window the battery itself recorded as DIRTY TWICE (persistent colbert co-resident,
# window_clean=false). Correctness stage fully green.
#
# THE FORM THAT CAN TELL A REGRESSION FROM A POISONED WINDOW (per tools/local-ci.sh's
# own protocol and the v0.71.0 worked example, perf-ab.sh): same thermal window,
# interleaved A/B/A/B, N=5 each, GPU held under ONE flock the whole time.
#   A = base b78b439bc (the lane's merge base)
#   B = lane candidate (guard sweep merged with origin/main c4145956b)
# Cell = qwen9b-plain-short, byte-identical config to perf-cells.json.
# Verdict rule: compare A-median to B-median FROM THIS RUN ONLY.
set -uo pipefail
cd /home/avifenesh/projects/memra/wt-graph-guard-sweep

BASE=/tmp/wt-guard-base-ab/target/release/run-gen
CAND=target/release/run-gen
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
PROMPT=research/gemma4-bringup/e4b-chat-watercycle-ids.txt
LOGD=/tmp/guard-perf-ab-logs
N=5

[ -x "$BASE" ] || { echo "perf-ab: no baseline binary at $BASE"; exit 2; }
[ -x "$CAND" ] || { echo "perf-ab: no candidate binary at $CAND"; exit 2; }
[ -f "$MODEL" ] || { echo "perf-ab: no model at $MODEL"; exit 2; }
mkdir -p "$LOGD"

echo "== perf-ab settle: qwen9b-plain-short, N=$N interleaved, one lock hold =="
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed 's/^/   entry: /'
echo "   other GPU compute apps at entry:"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/     /'
echo

: > "$LOGD/A.toks"; : > "$LOGD/B.toks"

flock -w 7200 /tmp/memra-5090.lock env \
  LOGD="$LOGD" N="$N" BASE="$BASE" CAND="$CAND" MODEL="$MODEL" PROMPT="$PROMPT" \
  bash -s <<'INNER'
rep() {
    local arm="$1" bin="$2" i="$3" log="$LOGD/$arm-$i.log"
    # shellcheck disable=SC2046
    MEMRA_NGEN=128 timeout 420 "$bin" "$MODEL" $(cat "$PROMPT") > "$log" 2>&1
    local rc=$? toks t
    toks=$(grep -oE "= [0-9.]+ tok/s" "$log" | tail -1 | grep -oE "[0-9.]+")
    t=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')
    printf '  %-4s rep%d: %-8s tok/s  (exit %d, %s)\n' "$arm" "$i" "${toks:-NONE}" "$rc" "$t"
    [ -n "$toks" ] && echo "$toks" >> "$LOGD/$arm.toks"
}
echo "  [warmup rep into the steady thermal state, discarded]"
rep warm "$CAND" 0 > /dev/null 2>&1
for i in $(seq 1 "$N"); do
    rep A "$BASE" "$i"
    rep B "$CAND" "$i"
done
INNER
echo

med() { sort -g "$1" | awk '{a[NR]=$1} END{if(NR==0){print 0;exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
AM=$(med "$LOGD/A.toks"); BM=$(med "$LOGD/B.toks")
echo "== VERDICT (this window only, N=$N each, interleaved, exclusive lock) =="
echo "   A base b78b439bc median: $AM tok/s   [$(tr '\n' ' ' < "$LOGD/A.toks")]"
echo "   B lane candidate median: $BM tok/s   [$(tr '\n' ' ' < "$LOGD/B.toks")]"
awk -v a="$AM" -v b="$BM" 'BEGIN{printf "   B vs A: %+.2f%%\n", (b-a)/a*100}'
