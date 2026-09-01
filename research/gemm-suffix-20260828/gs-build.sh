#!/bin/bash
# lane/gemm-suffix: build ONE binary from the lane tip + the suffix patch. Own worktree,
# own target dir; touches no other lane's checkout.
set -u
source $HOME/.cargo/env
cd /root/wt-gemmsuffix
OUT=/root/gs-build.txt; : > $OUT
{
  echo "=== BUILD start=$(date -Is)"
  echo "    git log -1 = $(git log -1 --format='%H %s')"
  echo "    dirty       = $(git status --porcelain | tr '\n' ' ')"
  echo "    patch sha256= $(sha256sum /root/gemmsuffix.patch | cut -d' ' -f1)"
  echo "    tree diff sha256= $(git diff | sha256sum | cut -d' ' -f1)"
} >> $OUT
cargo build --release -p memra-server > /root/gs-build.log 2>&1
RC=$?
echo "    rc=$RC end=$(date -Is)" >> $OUT
if [ $RC -ne 0 ]; then tail -40 /root/gs-build.log >> $OUT; echo "GS-BUILD-FAILED" >> $OUT; exit 4; fi
cp -f target/release/memra-server /root/memra-server.gsuffix
{
  echo "    bin=/root/memra-server.gsuffix md5=$(md5sum /root/memra-server.gsuffix | cut -c1-12)"
  echo "    BINARY FINGERPRINT (strings, never cargo's Finished line):"
  echo "      gemm-prime ENGAGED  = $(strings -a /root/memra-server.gsuffix | grep -c 'gemm-prime. ENGAGED')"
  echo "      gemm-prime WALK     = $(strings -a /root/memra-server.gsuffix | grep -c 'gemm-prime. WALK')"
  echo "      base= discriminator = $(strings -a /root/memra-server.gsuffix | grep -c 'ENGAGED t=')"
  echo "      suffix door name    = $(strings -a /root/memra-server.gsuffix | grep -c 'MEMRA_STEP_GEMM_PRIME_SUFFIX')"
  echo "      tsend canary name   = $(strings -a /root/memra-server.gsuffix | grep -c 'MEMRA_STEP35_PRIME_BATCH_TSEND')"
  echo "      rewound receipt     = $(strings -a /root/memra-server.gsuffix | grep -c 'plain-affinity: rewound to')"
  df -h / | tail -1
  echo "GS-BUILD-DONE $(date -Is)"
} >> $OUT
