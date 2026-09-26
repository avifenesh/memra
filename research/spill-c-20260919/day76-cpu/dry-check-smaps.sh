#!/usr/bin/env bash
# DAY76: the cell's smaps capture exercised against a real ELF named run-gen-* (the control-flow dry check's stub is a
# script, whose process name is bash): a copy of python3 named run-gen-p76 sleeps 2 s; the capture line of
# day76-cell.sh, copied here verbatim, finds it by name and saves its smaps. Prints the saved file's VMA count.
set -euo pipefail
D=$(mktemp -d /tmp/c76smaps.XXXXXX)
trap 'rm -rf "$D"' EXIT
cp "$(readlink -f /usr/bin/python3)" "$D/run-gen-p76"
"$D/run-gen-p76" -c "import time; time.sleep(2)" &
sleep 0.3
EV=$D; label=o1-di-r1
p=$(ps -eo pid,comm | awk '$2 ~ /^run-gen/ {print $1}' | head -1)
[ -n "$p" ] && cat "/proc/$p/smaps" > "$EV/$label.smaps" 2>&1
wait
echo "found pid: ${p:+yes}; smaps VMAs: $(grep -c '^Size:' "$EV/$label.smaps")"
