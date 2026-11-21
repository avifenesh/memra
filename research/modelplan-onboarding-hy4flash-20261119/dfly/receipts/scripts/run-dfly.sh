#!/usr/bin/env bash
# GB10 box: one drafter oracle run. usage: run-dfly.sh TAG REQUEST DTYPE [extra args]
set -u
tag=$1 req=$2 dtype=$3; shift 3
D=$HOME/hy4flash/pr10
log=$HOME/hy4flash/logs/pr10-dfly-$tag.log
echo "START $(date '+%F %T') tag=$tag req=$req dtype=$dtype args=$*" > $log
systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-28G} -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $HOME/hy4flash/pr7/venv/bin/python -I $D/lane/dfly/make_dfly_oracle.py $HOME/hy4flash/hf-stage \
  $HOME/hy4flash/pr7/requests/$req.json $D/target/$req $D/dtapes/$tag --vllm-src $D/vllm-src --dtype $dtype "$@" >> $log 2>&1
echo "EXIT rc=$? $(date '+%F %T')" >> $log
