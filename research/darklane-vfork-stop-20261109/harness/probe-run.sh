#!/bin/bash
# Proof run for the darklane stop-acknowledgement flake. Writes under $OUT.
set -u
OUT=$WORK/darklane/proof
mkdir -p "$OUT"
cd "$OUT" || exit 1
PIN="taskset -c 24-31,56-63 nice -n 10"
{
  echo "host_kernel: $(uname -srvm)"
  echo "sh: $(readlink -f /bin/sh) $(dpkg-query -W -f='${Package} ${Version}' dash)"
  echo "bash: $(dpkg-query -W -f='${Package} ${Version}' bash)"
  echo "glibc: $(ldd --version | head -1)"
  echo "date: $(date -Is)"
} > env.txt

# 1. dash runs external commands with vfork (bash forks). Two commands so the shell must spawn the first.
$PIN strace -f -qq -e trace=vfork,fork,clone,clone3,execve -o strace-dash.txt sh -c 'sleep 0.001; true'
$PIN strace -f -qq -e trace=vfork,fork,clone,clone3,execve -o strace-bash.txt bash -c 'sleep 0.001; true'

# 2. the stop/ack race, the test's own job and a constantly forking job, dash vs bash.
$PIN python3 -I $WORK/darklane/vfork_stop_probe.py sh 'while :; do sleep 0.01; done' 3000 3 > probe-dash-sleep.jsonl 2> probe-dash-sleep.err
$PIN python3 -I $WORK/darklane/vfork_stop_probe.py sh 'while :; do /bin/true; done' 3000 3 > probe-dash-true.jsonl 2> probe-dash-true.err
$PIN python3 -I $WORK/darklane/vfork_stop_probe.py bash 'while :; do sleep 0.01; done' 3000 3 > probe-bash-sleep.jsonl 2> probe-bash-sleep.err
$PIN python3 -I $WORK/darklane/vfork_stop_probe.py bash 'while :; do /bin/true; done' 3000 3 > probe-bash-true.jsonl 2> probe-bash-true.err
echo done > DONE
