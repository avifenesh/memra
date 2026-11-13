#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1257): fill the page cache until MemFree is
# under LIMIT_MB (default 3000) or CAP_S seconds pass (default 300). Like memra#1221's
# fill_cache.sh, but with READERS parallel readers (default 4) over the model files, largest
# first, since one reader on the test unit left MemFree at 32 GB after 120 s (most of its first files were
# already cached). Reads only; writes nothing and changes no setting; the kernel reclaims these
# pages on demand. Prints MemFree/MemAvailable/Cached before and after.
# usage: fill_cache2.sh [LIMIT_MB [CAP_S [READERS]]]
LIM=${1:-3000}; CAP=${2:-300}; NR=${3:-4}
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';'; }
echo "$(date -Is) fill2: before $(mi)"
L=$(mktemp)
ls -S $HOME/models/*.gguf $HOME/models/*/*.gguf $HOME/models/*/*/*.gguf $HOME/models/*/*.safetensors \
  $HOME/models/*/*/*.safetensors 2>/dev/null > "$L"
pids=()
for i in $(seq 0 $((NR - 1))); do
  ( awk -v n="$NR" -v i="$i" 'NR % n == i' "$L" | while read -r f; do cat "$f" > /dev/null; done ) &
  pids+=($!)
done
waitfor --until "awk '/^MemFree/{exit !(\$2/1024 < $LIM)}' /proc/meminfo" --every 2s --timeout "${CAP}s" --quiet > /dev/null 2>&1
for p in "${pids[@]}"; do kill $(ps -o pid= --ppid "$p") "$p" 2>/dev/null; done
wait 2>/dev/null
rm -f "$L"
echo "$(date -Is) fill2 done: $(mi)"
