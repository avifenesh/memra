#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: fill the page cache by reading model files (largest
# first) until MemFree is under LIMIT_MB (default 3000) or CAP_S seconds have passed (default 120),
# so the driver's raw free reading is low while MemAvailable stays high: the condition in which the
# pre-capture pool trim fires. Nothing is written and no system setting is touched; the kernel
# reclaims these pages on demand. On these units MemFree settles at 6-10 GB once the cache is full
# (it fell 36 -> 6.6 GB and then stopped falling), so the cap is what ends most fills; whether the
# trim fired is read from each cell's own `[spec] pre-capture pool trim` line.
# usage: fill_cache.sh [LIMIT_MB [CAP_S]]
LIM=${1:-3000}; CAP=${2:-120}
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';'; }
echo "$(date -Is) fill: before $(mi)"
( for f in $(ls -S $HOME/models/*.gguf $HOME/models/*/*.gguf $HOME/models/*/*/*.gguf $HOME/models/*/*.safetensors 2>/dev/null); do
    cat "$f" > /dev/null
  done ) &
rp=$!
waitfor --until "awk '/^MemFree/{exit !(\$2/1024 < $LIM)}' /proc/meminfo" --every 3s --timeout "${CAP}s" --quiet > /dev/null 2>&1
kill $(ps -o pid= --ppid $rp) $rp 2>/dev/null
wait $rp 2>/dev/null
echo "$(date -Is) fill done: $(mi)"
