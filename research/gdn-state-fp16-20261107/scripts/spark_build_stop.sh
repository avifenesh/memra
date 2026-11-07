#!/usr/bin/env bash
# Runs ON a GB10 unit: stop this lane's build (spark_build.sh, its cargo and every rustc/nvcc
# child under the lane target dir) by pid, then report what is left.
set -u
for p in $(ps -eo pid,args | awk '/[s]park_build\.sh|[c]argo build --release --locked/ && /gdnfp16|spark_build|cargo build/ {print $1}'); do
  kill "$p" 2>/dev/null
done
for p in $(ps -eo pid,args | awk '/[m]emra-gdnfp16\/(target|src)/ {print $1}'); do
  kill "$p" 2>/dev/null
done
ps -eo pid,args | awk '/[m]emra-gdnfp16|[s]park_build/ {print "left:", $0}' | cut -c1-140
