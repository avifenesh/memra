#!/usr/bin/env bash
# Runs ON unit-a inside this lane's lease (memra#1331 / PR #1350's own GB10 cell): arm ARM as
# a systemd user service shaped like deploy/systemd/memra-server.service's supervision
# (Type=notify, WatchdogSec=180, Restart=on-failure, TimeoutStartSec=600), in memra#1307's MTP
# serving (262,144 context, 32 sessions), with MEMRA_PANIC_AFTER=1: the first completed request
# panics the worker once, and the in-process respawn reloads. Read out: NRestarts and the main
# PID before and after, /readyz reached again (and when), every `withholding WATCHDOG=1` line,
# systemd's watchdog lines, and the respawn's reload steps with their durations (server lines
# timestamped by tsrun.py). usage: unit-a-wd.sh ARM
set -u
ARM=$1
R=$HOME/.cache/memra-1255/gb10
B=$R/bins/$ARM
OUT=$R/runs/wd-$ARM${TIMED:+-t}
U=memra1255-wd-$ARM${TIMED:+-t}
mkdir -p "$OUT"
echo "$$" >> "$R/pids.txt"
S=$OUT/summary.txt
TS=$B/research/spec-capture-headroom-20261112/tools/tsrun.py
Q=$HOME/models/memra-q38
G8=$Q/Qwen3.8-27B-NVFP4-MTP-GGUF
{ date -Is; hostname; sha256sum "$B/memra-server" | sed "s|$HOME|~|"; free -m | head -n 2; } > "$S"
systemctl --user reset-failed "$U" 2>/dev/null
: > "$OUT/server.log"
systemd-run --user --no-block --unit "$U" \
  -p Type=notify -p NotifyAccess=all -p WatchdogSec=180 -p TimeoutStartSec=600 \
  -p Restart=on-failure -p RestartSec=10 -p StartLimitIntervalSec=3600 -p StartLimitBurst=4 \
  -p TimeoutStopSec=60 -p "StandardOutput=append:$OUT/server.log" -p "StandardError=append:$OUT/server.log" \
  --setenv=MEMRA_COMPAT=openai --setenv=MEMRA_ADDR=127.0.0.1:8090 --setenv=MEMRA_KV_HOST_MB=0 \
  --setenv=MEMRA_TICK_TRACE=1 --setenv=MEMRA_VISION_DIR="$HOME/models/Qwen3.8-27B-NVFP4" \
  --setenv=MEMRA_CTX=262144 --setenv=MEMRA_MAX_SESSIONS=32 \
  --setenv=MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  --setenv=MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" \
  --setenv=MEMRA_PREFIX_STABLE_BOUNDARY=1 --setenv=MEMRA_PANIC_AFTER=1 \
  --setenv=PATH="/usr/local/cuda/bin:/usr/bin:/bin" \
  python3 "$TS" "$B/memra-server" >> "$S" 2>&1
show() { systemctl --user show "$U" -p ActiveState -p SubState -p NRestarts -p MainPID -p Result | tr '\n' ' '; }
t0=$(date +%s)
if ! waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --timeout 12m --every 5s --quiet; then
  echo "NOT READY within 12m: $(show)" >> "$S"
else
  t1=$(date +%s)
  echo "first ready after $((t1 - t0)) s: $(show)" >> "$S"
  echo "$(show)" > "$OUT/before.txt"
  if [ "${TIMED:-0}" = 1 ]; then
    # Land the panic about 70 s after a ping the worker answered live, so that the respawn's
    # reload (about 125 s here) spans the next two ping slots: the case where main withholds
    # pings past WatchdogSec (the issue's worst phase). Slots are the pinger's: its armed line
    # plus multiples of 90 s.
    ta=$(grep -m1 'watchdog armed' "$OUT/server.log" | cut -d' ' -f1)
    target=$(python3 -c 'import sys,math; ta,tr=float(sys.argv[1]),float(sys.argv[2]); k=math.ceil((tr+2-ta)/90); print(f"{ta+90*k+68:.0f}")' "$ta" "$t1")
    echo "timed: armed $ta, ready $t1, send at $target (a live ping slot + 68 s)" >> "$S"
    now=$(date +%s); [ "$target" -gt "$now" ] && sleep $((target - now))
  fi
  curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"q38","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16,"temperature":0}' > "$OUT/req1.json" 2>&1
  echo "request 1 done $(date -Is): $(head -c 300 "$OUT/req1.json")" >> "$S"
  # the panic lands after the completed request; give the worker a moment to fall
  waitfor --until 'grep -q "PANIC in the GPU worker thread" '"$OUT/server.log" --timeout 2m --every 2s --quiet \
    && echo "panic seen $(date -Is)" >> "$S"
  t2=$(date +%s)
  if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --timeout 15m --every 5s --quiet; then
    echo "ready again after $(( $(date +%s) - t2 )) s from the panic: $(show)" >> "$S"
  else
    echo "NOT ready again within 15m: $(show)" >> "$S"
  fi
  # one more request on the respawned (or restarted) worker
  curl -s -m 300 http://127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"q38","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16,"temperature":0}' > "$OUT/req2.json" 2>&1
  echo "request 2: $(head -c 300 "$OUT/req2.json")" >> "$S"
fi
echo "final: $(show)" >> "$S"
systemctl --user stop "$U"
waitfor --until "! systemctl --user is-active -q $U" --timeout 2m --every 5s --quiet
journalctl --user -u "$U" --no-pager -o short-iso-precise > "$OUT/journal.txt" 2>&1
grep -iE 'watchdog|restart|Main process exited|Failed|Started|Starting' "$OUT/journal.txt" | cut -c1-300 >> "$S"
echo "withholding lines: $(grep -c 'withholding WATCHDOG=1' "$OUT/server.log")" >> "$S"
grep -E '\[worker\] (PANIC|respawn attempt|loading model)|\[sd-notify\]|engine ready|READY|listening|calibrat' "$OUT/server.log" | cut -c1-260 > "$OUT/steps.txt"
echo "WD-DONE $ARM $(date -Is)" >> "$S"
