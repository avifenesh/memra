#!/usr/bin/env bash
# cells.sh <probe binary> <out root> <cell>...   cell = name:ENV=V,ENV=V (ENV list may be empty; DECODE_BATCH = --decode batch)
# One gpulock lease per cell; runs the argmax-margin-probe 12-row window on board-2048, gemma-4-26B-A4B.
set -u
probe=$(realpath "$1"); out=$(realpath -m "$2"); shift 2
model=${MODEL:-<models>/gemma-4-26B_q4_0-it.gguf}
cd "$HOME/.cache/memra-1225/wt" || exit 1
mkdir -p "$out"
for cell in "$@"; do
  name=${cell%%:*}; envs=${cell#*:}
  dir=$out/$name; mkdir -p "$dir"
  envargs=(NVIDIA_TF32_OVERRIDE=0)
  decode=()
  IFS=',' read -r -a kv <<< "$envs"
  for x in "${kv[@]}"; do
    [ -n "$x" ] || continue
    if [ "$x" = DECODE_BATCH ]; then decode=(--decode batch); else envargs+=("$x"); fi
  done
  echo "$(date '+%F %T') cell=$name env=${envargs[*]} decode=${decode[*]}" | tee -a "$out/cells.txt"
  start=$(date +%s)
  gpulock run --wait 3h --ttl 60m --for "memra #1225 probe cell $name" -- \
    env "${envargs[@]}" "$probe" "$model" research/e2e/prompts/board-2048.txt 12 "${decode[@]}" \
    > "$dir/probe.log" 2>&1
  rc=$?
  table=$(grep -E '^[0-9]{4} ' "$dir/probe.log" | sha256sum | cut -c1-16)
  echo "$(date '+%F %T') cell=$name rc=$rc wall_s=$(( $(date +%s) - start )) table_sha=$table" | tee -a "$out/cells.txt"
done
echo CELLS-DONE >> "$out/cells.txt"
