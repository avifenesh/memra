#!/bin/bash
# Peak ANON sampler for the q4e-loader cell scopes.
#
# RSS is the wrong instrument here and cgroup memory.peak is worse: the artifact mmap
# contributes 60+ GB of RssFile whose page-cache charge can belong to a different cgroup
# entirely, and memory.peak saturates at the cap because reclaimable file cache always
# expands to fill it. The kernel OOM line that opened memra issue #48 reported anon-rss, so
# anon is what this lane reports. One line per new maximum; the last line per scope is the
# peak.
out=/root/realgate/loaderout/anon-peak.tsv
[ -f "$out" ] || printf "# utc\tscope\tanon_bytes\tanon_GiB\n" > "$out"
declare -A peak
while :; do
  for d in /sys/fs/cgroup/system.slice/q4e-*.scope; do
    [ -d "$d" ] || continue
    n=$(basename "$d" .scope)
    a=$(awk '/^anon /{print $2}' "$d/memory.stat" 2>/dev/null)
    [ -n "$a" ] || continue
    cur=${peak[$n]:-0}
    if [ "$a" -gt "$cur" ]; then
      peak[$n]=$a
      printf "%s\t%s\t%s\t%.1f\n" "$(date -u +%FT%TZ)" "$n" "$a" "$(awk "BEGIN{print $a/1073741824}")" >> "$out"
    fi
  done
  sleep 2
done
