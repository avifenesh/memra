#!/usr/bin/env bash
# GB10 box: run GPU jobs strictly one after another. usage: queue.sh SCRIPT [args] (appends to the queue
# lock: a job waits for the lock file, so two jobs never load the weights at once)
set -u
exec 9> $HOME/hy4flash/pr9/gpu.lock
flock 9
"$@"
