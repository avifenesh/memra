#!/bin/bash
# Clean proof run (no build on the pinned CPUs). Writes under $OUT.
set -u
OUT=$WORK/darklane/proof2
mkdir -p "$OUT"
cd "$OUT" || exit 1
PIN="taskset -c 24-31,56-63 nice -n 10"
P="python3 -I $WORK/darklane/vfork_stop_probe.py"
{
  echo "kernel: $(uname -srvm)"
  echo "sh: $(readlink -f /bin/sh) $(dpkg-query -W -f='${Package} ${Version}' dash)"
  echo "bash: $(dpkg-query -W -f='${Package} ${Version}' bash)"
  echo "glibc: $(ldd --version | head -1)"
  echo "python: $(python3 --version)"
  echo "cpus: 16 pinned (taskset), nice 10; no build on them during the probe"
  echo "loadavg_start: $(cat /proc/loadavg)"
  echo "date: $(date -Is)"
} > env.txt
$PIN strace -f -qq -e trace=vfork,fork,clone,clone3,execve -o strace-dash.txt sh -c 'sleep 0.001; true'
$PIN strace -f -qq -e trace=vfork,fork,clone,clone3,execve -o strace-bash.txt bash -c 'sleep 0.001; true'
# Permanence: hold each miss 3 s, then show a re-stop right after SIGCONT misses again.
$PIN $P sh 'while :; do sleep 0.01; done' 3000 3 3 > hold-dash-sleep.jsonl 2> hold-dash-sleep.err
# Rate: 1000 stops each, misses counted (0.5 s wait per stop), no hold.
for shell in sh bash; do
  $PIN $P $shell 'while :; do sleep 0.01; done' 1000 1000 0 > rate-$shell-sleep.jsonl 2> rate-$shell-sleep.err
  $PIN $P $shell 'while :; do /bin/true; done' 1000 1000 0 > rate-$shell-true.jsonl 2> rate-$shell-true.err
done
echo "loadavg_end: $(cat /proc/loadavg)" >> env.txt
echo done > DONE
