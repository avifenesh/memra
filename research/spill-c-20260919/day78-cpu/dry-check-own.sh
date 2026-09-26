#!/usr/bin/env bash
# DAY78: the cell's `own_run_gen` (copied verbatim from day78-cell.sh) against real ELFs named run-gen-p78: one started
# below this shell, one started in its own session (not below this shell). It must name the first and never the second.
set -uo pipefail
D=$(mktemp -d /tmp/c78own.XXXXXX)
trap 'rm -rf "$D"' EXIT
cp "$(readlink -f /usr/bin/python3)" "$D/run-gen-p78"
own_run_gen() {
    local p q
    for p in $(ps -eo pid,comm | awk '$2 == "run-gen-p78" {print $1}'); do
        q=$p
        while [ -n "$q" ] && [ "$q" -gt 1 ]; do
            [ "$q" = "$$" ] && { echo "$p"; return 0; }
            q=$(awk '{print $4}' "/proc/$q/stat" 2>/dev/null)
        done
    done
    return 1
}
# A double fork: the subshell exits at once, so this process is reparented and is not below this shell.
( setsid "$D/run-gen-p78" -c "import time; time.sleep(3)" < /dev/null > /dev/null 2>&1 & )
sleep 0.3
foreign=$(ps -eo pid,args | grep "[r]un-gen-p78 -c" | awk '{print $1}' | head -1)
( "$D/run-gen-p78" -c "import time; time.sleep(3)" ) &
sleep 0.5
mine=$(ps -eo pid,ppid,args | awk -v s=$! '$2 == s && /run-gen-p78/ {print $1}')
[ -z "$mine" ] && mine=$(pgrep -P $! 2>/dev/null)
found=$(own_run_gen)
echo "foreign=${foreign:-none} mine=${mine:-?} found=${found:-none} -> $([ -n "$foreign" ] && [ "$found" != "$foreign" ] && [ -n "$found" ] && echo ok || echo WRONG)"
wait
